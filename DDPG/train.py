"""
DDPG 训练脚本（Pendulum-v1）

一共四个网络：
    amodel       actor  在线网络，负责产动作 μ(s)
    amodel_fix   actor  目标网络，负责算 TD 目标里的 a' = μ'(s')
    cmodel       critic 在线网络，负责打分 Q(s,a)
    cmodel_fix   critic 目标网络，负责算 TD 目标里的 Q'(s', a')

一次梯度更新（critic 和 actor 同一频率，共用同一个 batch）：
    1. trainq : 用 TD 目标更新 critic
    2. traina : 用 critic 对 μ(s) 的评分反传，更新 actor
    3. soft_update : 两个目标网络朝在线网络各挪一小步
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from agent import Agent
from model import CriticNetwork, ActorNetwork, load_checkpoint
from replay_buffer import ReplayBuffer

MEMORY_SIZE = 100_000
BATCH_SIZE = 64
WARMUP_SIZE = 2000          # 前 2000 步用纯随机动作把回放池填起来，这期间不做梯度下降
FREQ = 4                    # 每 4 个环境步做一次梯度更新
GAMMA = 0.99
TAU = 0.005                 # 目标网络软更新系数
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
MAX_EPISODE = 300
EVAL_EVERY = 20
SAVE_THRESHOLD = -200.0     # Pendulum 随机策略约 -1200，训练到 -200 以内算不错

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pendulum_ddpg.pt")



def soft_update(model, model_fix, tau=TAU):
    """θ_target ← (1-τ)·θ_target + τ·θ_online"""
    with torch.no_grad():
        for target_param, param in zip(model_fix.parameters(), model.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


def batch_to_tensors(batch, device):
    """把 buffer 采出来的 (s, a, r, s', d) 转成同一设备上的 float32 张量。

    buffer 里存的是 numpy 数组 / python 标量，形状分别是：
        s (B, obs_dim)   a (B, act_dim)   r (B,)   s' (B, obs_dim)   d (B,)
    转成张量后 r 和 d 要补一维成 (B, 1)，才能和 Q 值逐元素对齐。
    """
    s, a, r, s2, d = batch
    s = torch.as_tensor(np.asarray(s), dtype=torch.float32, device=device)
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    r = torch.as_tensor(np.asarray(r), dtype=torch.float32, device=device).unsqueeze(1)
    s2 = torch.as_tensor(np.asarray(s2), dtype=torch.float32, device=device)
    d = torch.as_tensor(np.asarray(d), dtype=torch.float32, device=device).unsqueeze(1)
    return s, a, r, s2, d


def trainq(cmodel, cmodel_fix, amodel_fix, batch, optimizerc, gamma=GAMMA):
    """用 TD 误差更新 critic。

        y    = r + γ(1-d)·Q'(s', μ'(s'))
        loss = MSE( Q(s, a), y )

    这里有两处必须 no_grad：
        - 目标网络只是「提供标签」，不该产生梯度
        - a' 由目标 actor 给出，同样不需要梯度
    """
    device = next(cmodel.parameters()).device
    s, a, r, s2, d = batch_to_tensors(batch, device)

    with torch.no_grad():
        a2 = amodel_fix(s2)                                    # 目标 actor 选下一个动作 [B, act_dim]
        q_next = cmodel_fix(torch.cat([s2, a2], dim=1))
        y = r + gamma * q_next * (1.0 - d)

    q = cmodel(torch.cat([s, a], dim=1))                       # a 是真实动作，只当输入，不需要梯度
    loss = F.mse_loss(q, y)

    optimizerc.zero_grad()
    loss.backward()
    optimizerc.step()
    return loss.item()


# ---------------------------------------------------------------------- #
# actor 更新
# ---------------------------------------------------------------------- #
def traina(amodel, cmodel, batch, optimizera):
    """用 critic 的评分反过来更新 actor。

        loss = -Q(s, μ(s)).mean()

    两个关键点：
        1. μ(s) 必须**当场重新前向**，不能拿 rollout 时存下来的动作。
           只有重新前向，计算图里才有「actor 参数 → 动作 → Q 值」这条链，
           梯度才能从 Q 一路回传到 actor。
        2. 这一趟只想更新 actor，所以先把 critic 的参数冻住，
           否则 backward 会顺带算出 critic 的梯度（虽然不 step，但白算一遍还容易出错）。
    """
    device = next(amodel.parameters()).device
    s = batch_to_tensors(batch, device)[0]

    for p in cmodel.parameters():
        p.requires_grad_(False)
    try:
        a_pred = amodel(s)                                     # 重新前向，梯度链从这里开始
        q = cmodel(torch.cat([s, a_pred], dim=1))
        loss = -q.mean()                                       # 最大化 Q 等价于最小化 -Q
        optimizera.zero_grad()
        loss.backward()
        optimizera.step()
    finally:
        for p in cmodel.parameters():
            p.requires_grad_(True)
    return loss.item()



def run_episode(env, agent: Agent, cmodel, cmodel_fix, amodel, amodel_fix,
                buffer: ReplayBuffer, optimizerc, optimizera):
    total_reward = 0.0
    obs, _ = env.reset()
    step = 0
    while True:
        step += 1
        agent.env_step += 1

        # 暖启动阶段动作全随机：只往池子里存，不训练。
        # 此时 actor 还是随机初始化的，光靠它加噪声覆盖不了动作空间。
        if agent.env_step <= WARMUP_SIZE:
            action = env.action_space.sample()
        else:
            action = agent.sample(obs, amodel)

        next_obs, reward, terminated, truncated, _ = env.step(action)   # 只调一次，返回值全部用上


        buffer.push((obs, action, reward, next_obs, terminated))

        obs = next_obs
        total_reward += reward

        if len(buffer) >= max(WARMUP_SIZE, BATCH_SIZE) and step % FREQ == 0:
            batch = buffer.sample_transposed(BATCH_SIZE)        # 只采一次，critic 和 actor 共用
            trainq(cmodel, cmodel_fix, amodel_fix, batch, optimizerc)
            traina(amodel, cmodel, batch, optimizera)
            soft_update(cmodel, cmodel_fix)
            soft_update(amodel, amodel_fix)
            agent.learn_step += 1

        if terminated or truncated:
            break
    return total_reward



def evaluate(env, agent, amodel, n=3):
    rewards = []
    for _ in range(n):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            action = agent.predict(obs, amodel)                 # 评测用确定性动作，不加噪声
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    avg_score = sum(rewards) / len(rewards)
    print("eval 平均回报: %.1f" % avg_score)
    return avg_score


def save_model(amodel, cmodel, episode, score):
    torch.save({
        "actor": {k: v.detach().cpu() for k, v in amodel.state_dict().items()},
        "critic": {k: v.detach().cpu() for k, v in cmodel.state_dict().items()},
        # 一定要转成 python 原生类型：
        # score 来自 sum(rewards)/len(rewards)，是 np.float64；
        # numpy 标量会被 pickle 进存档，之后 torch.load 默认 weights_only=True 就解不开了
        "episode": int(episode),
        "score": float(score),
    }, MODEL_PATH)
    print(">>> 平均回报 %.1f，模型已保存到 %s" % (score, MODEL_PATH))


def load_best_score():
    if not os.path.exists(MODEL_PATH):
        return -float("inf")
    ckpt = load_checkpoint(MODEL_PATH, map_location="cpu")
    score = float(ckpt.get("score", -float("inf")))
    print("发现已有模型文件，历史最好平均回报 %.1f" % score)
    return score


def main():
    env = gym.make("Pendulum-v1")
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    buffer = ReplayBuffer(capacity=MEMORY_SIZE, seed=42)

    amodel = ActorNetwork(obs_dim, action_dim, max_action=max_action).to(DEVICE)
    amodel_fix = ActorNetwork(obs_dim, action_dim, max_action=max_action).to(DEVICE)
    cmodel = CriticNetwork(obs_dim, action_dim).to(DEVICE)
    cmodel_fix = CriticNetwork(obs_dim, action_dim).to(DEVICE)

    # 目标网络必须从在线网络的拷贝出发，否则开局 TD 目标全是随机数
    amodel_fix.load_state_dict(amodel.state_dict())
    cmodel_fix.load_state_dict(cmodel.state_dict())

    optimizera = torch.optim.Adam(amodel.parameters(), lr=LR_ACTOR)
    optimizerc = torch.optim.Adam(cmodel.parameters(), lr=LR_CRITIC)

    agent = Agent(action_low=env.action_space.low,
                  action_high=env.action_space.high,
                  noise_std=0.2,
                  noise_std_min=0.05,
                  noise_decay=0.995)

    print("使用设备: %s | obs_dim=%d act_dim=%d max_action=%.1f"
          % (DEVICE, obs_dim, action_dim, max_action))

    best_score = load_best_score()
    for episode in range(MAX_EPISODE):
        total_reward = run_episode(env, agent, cmodel, cmodel_fix, amodel, amodel_fix,
                                   buffer, optimizerc, optimizera)

        if agent.env_step > WARMUP_SIZE:
            agent.noise_decay()        # 暖启动期间动作是随机的，衰减噪声没意义

        if episode % EVAL_EVERY == 0:
            print("episode %d | train reward %.1f | 噪声 %.3f | 梯度步数 %d | 环境步数 %d"
                  % (episode, total_reward, agent.noise_std, agent.learn_step, agent.env_step))
            score = evaluate(env, agent, amodel)
            if score >= SAVE_THRESHOLD and score > best_score:
                best_score = score
                save_model(amodel, cmodel, episode, score)

    env.close()


if __name__ == "__main__":
    main()
