import os
from model import DQN_model
from agent import Agent
import torch
import numpy
import gymnasium as gym          # 新版 gym：gymnasium
from replay_buffer import ReplayBuffer

MEMORY_SIZE = 10000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE =32
MAX_EPISODE = 2000
FREQ = 4

RENDER_ON_EVAL = False           # 只在评测时渲染画面；训练过程始终不渲染

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 有 CUDA 就用 CUDA，否则用 CPU
EVAL_EVERY = 50                 # 每 50 个 episode 评测一次
SAVE_THRESHOLD = 200.0          # 平均回报达到多少算“优秀”（CartPole-v1 满分 500）

# 探索率衰减：每个 episode 结束乘一次 decay，最低降到 min
E_GREED_START = 0.1
E_GREED_MIN   = 0.01
E_GREED_DECAY = 0.995

# ---------------- 目标网络：软更新 ----------------
# 不再“每 N 步把在线网络的参数整个拷过去”，而是每做一次梯度下降，
# 就让目标网络的参数朝在线网络挪一小步：θ_target ← (1-τ)·θ_target + τ·θ_online
# τ 越大跟得越紧。0.005 意味着目标网络大致是最近 200 次更新的滑动平均。
TAU = 0.005

# ---------------- 暖启动 ----------------
# 前 WARMUP_ENV_STEPS 步用纯随机动作把回放池填起来，这期间不做任何梯度下降。
# 目的是让池子里有足够多样的数据，而不是一开始就拿最近十几步高度相关的样本去学。
# 注意：软更新下目标网络每步都在动，不会出现“停在随机初始化上”的问题，
#      所以以前那套“热身期高频同步”的机制已经不需要了。
WARMUP_ENV_STEPS = 2000

# 模型存到脚本所在目录，避免从别的目录运行时跑到别处去
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_dqn.pt")

def in_warmup(agent):
    return agent.env_step <= WARMUP_ENV_STEPS

def soft_update(model, model_fix, tau=TAU):
    # θ_target ← (1-τ)·θ_target + τ·θ_online
    # 用 in-place 计算，不产生新的计算图；目标网络本来就不需要梯度
    with torch.no_grad():
        for target_param, param in zip(model_fix.parameters(), model.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

def try_episode(env,agent,model_fix,model,buffer):
    total_reward = 0
    obs,_ = env.reset()                 # 新版 API：reset 返回 (obs, info)
    step = 0
    while True:
        step+=1
        agent.env_step += 1
        # 暖启动阶段：动作全随机，只往池子里存，不训练
        if in_warmup(agent):
            action = numpy.random.randint(agent.act_n)
        else:
            action = agent.sample(obs,model)
        next_obs,reward,terminated,truncated,_ = env.step(action)   # 新版 API：返回 5 元组
        done = terminated or truncated
        buffer.push((obs,action,reward,next_obs,done))

        if (len(buffer) > MEMORY_WARMUP_SIZE) and (not in_warmup(agent)) and (step % FREQ == 0):
            batch = buffer.sample_transposed(BATCH_SIZE)
            loss = agent.train(batch,model,model_fix)
            soft_update(model,model_fix)     # 每次梯度下降后，目标网络挪一小步
        total_reward += reward
        obs =next_obs
        if done:
            break
    return total_reward

def evaluate(env,agent,model,n=5):
    rewards = []
    for i in range(n):
        obs,_ = env.reset()
        total_reward=0
        while True:
            action = agent.predict(obs,model)          # 评测用贪心策略
            next_obs,reward,terminated,truncated,_ = env.step(action)
            if RENDER_ON_EVAL:                         # 只有这里渲染，训练时不渲染
                env.render()
            total_reward += reward
            obs = next_obs                             # 必须更新 obs，否则一直在用同一个状态决策
            if terminated or truncated:
                break
        rewards.append(total_reward)
        print(total_reward)
    print("eval 平均回报: %.1f" % (sum(rewards)/len(rewards)))
    return rewards

def save_model(model,episode,score):
    # 存成 CPU 张量：这样换到没有 GPU 的机器也能直接 load，不用加 map_location
    state = {k: v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({
        "model": state,
        "episode": episode,
        "score": score,
    }, MODEL_PATH)
    print(">>> 平均回报 %.1f 达到要求，模型已保存到 %s" % (score,MODEL_PATH))

def load_best_score():
    # 重启训练时读回历史最好成绩，避免把之前更好的模型覆盖掉
    if not os.path.exists(MODEL_PATH):
        return -1.0
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    score = float(ckpt.get("score", -1.0))
    print("发现已有模型文件，历史最好平均回报 %.1f" % score)
    return score

def main():
    env = gym.make('CartPole-v1', render_mode='human' if RENDER_ON_EVAL else None)
    action_dim= env.action_space.n
    obs_shape = env.observation_space.shape

    buffer = ReplayBuffer(capacity=MEMORY_SIZE, seed=42)

    model,model_fix = DQN_model(),DQN_model()
    model.to(DEVICE)                            # 主网络放到设备上
    model_fix.to(DEVICE)                        # 目标网络也要放到同一个设备上
    print("使用设备: %s | 目标网络软更新 tau=%.4f" % (DEVICE,TAU))

    agent = Agent(obs_shape,action_dim,model,gamma=0.995,
                  e_greed=E_GREED_START,
                  e_greed_min=E_GREED_MIN,
                  e_greed_decay=E_GREED_DECAY)
    episode=0
    best_score = load_best_score()               # 历史最好成绩，没有模型文件时是 -1
    model_fix.load_state_dict(model.state_dict())    # 开训前先对齐一次，之后全靠软更新
    warmup_reported = False
    while episode < MAX_EPISODE:
        total_reward = try_episode(env,agent,model_fix,model,buffer)

        if not in_warmup(agent):
            if not warmup_reported:
                warmup_reported = True
                print(">>> 暖启动结束：已交互 %d 步，开始训练（目标网络 tau=%.4f 软更新）"
                      % (agent.env_step,TAU))
            agent.decay_e_greed()                # 暖启动期间动作是随机的，衰减探索率没意义
            if episode % EVAL_EVERY == 0:
                print("episode %d | train reward %.0f | e_greed %.3f | 梯度步数 %d | 环境步数 %d"
                      % (episode,total_reward,agent.e_greed,agent.learn_step,agent.env_step))
                rewards = evaluate(env,agent,model)
                avg_score = sum(rewards)/len(rewards)
                # 成绩优秀、而且比之前保存的更好，才覆盖保存
                if avg_score >= SAVE_THRESHOLD and avg_score > best_score:
                    best_score = avg_score
                    save_model(model,episode,avg_score)
        episode+=1
    env.close()

if "__main__" == __name__ :
    main()
