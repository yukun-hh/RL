"""
demo.py —— 加载训练好的模型跑演示

用法：
    python demo.py

它会读取同目录下的 cartpole_dqn.pt（train.py 保存出来的），
用贪心策略跑几局 CartPole，并把过程画出来。

想关掉画面只看分数，把 RENDER 改成 False。
"""
import os
import time
import torch
import gymnasium as gym
from  model import DQN_model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartpole_dqn.pt")
DEMO_EPISODE = 5        # 演示几局
RENDER = True           # 是否显示画面（无图形界面的服务器上请改成 False）
SLEEP = 0.02            # 每步停一下，否则太快看不清，设 0 就是全速

def load_model(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "没找到模型文件：%s\n请先运行 train.py 训练出一个模型。" % path
        )
    ckpt = torch.load(path, map_location=device)     # 存的时候已经转成 CPU 张量，这里再统一搬到 device
    model = DQN_model()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()                                     # 演示只推理，不训练
    print("已加载模型：%s" % path)
    print("  训练时 episode = %s，评测平均回报 = %.1f" % (ckpt.get("episode","?"), ckpt.get("score",-1)))
    return model

def choose_action(model, obs, device):
    # 和 Agent.predict 一样，只是这里直接用 argmax，演示要的是稳定表现
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(obs_t)
    return int(q.argmax(dim=1).item())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备: %s" % device)

    model = load_model(MODEL_PATH, device)

    env = gym.make('CartPole-v1', render_mode='human' if RENDER else None)
    rewards = []
    for i in range(DEMO_EPISODE):
        obs,_ = env.reset()
        total_reward = 0
        while True:
            action = choose_action(model, obs, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            if RENDER:
                env.render()
                if SLEEP > 0:
                    time.sleep(SLEEP)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        print("第 %d 局：坚持了 %.0f 步" % (i+1, total_reward))
    env.close()

    print("----")
    print("平均 %.1f 步（满分 500）" % (sum(rewards)/len(rewards)))

if "__main__" == __name__:
    main()
