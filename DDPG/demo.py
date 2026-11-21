"""
demo.py —— 加载训练好的 DDPG 模型跑演示（Pendulum-v1）

用法：
    python demo.py                  # 跑 3 局，存动图 + 曲线图
    python demo.py --episodes 5     # 跑 5 局
    python demo.py --mode human     # 开窗口实时看（需要图形界面）
    python demo.py --mode none      # 不录画面，只画曲线

输出（都写在脚本所在目录）：
    pendulum_demo.gif   游戏过程动图
    pendulum_demo.png   角度 / 角速度 / 力矩 / 每步奖励 的曲线

依赖：
    matplotlib  画曲线，必装
    Pillow      存 GIF，没有就跳过动图
    pygame      gymnasium 渲染画面用，没有就只能画曲线

注意：DDPG 的动作是连续的，演示时直接用 actor 输出的确定性动作，
      不加探索噪声，也不存在 DQN 那种 argmax 选动作。
"""

import os
import time
import argparse

import numpy as np
import torch
import gymnasium as gym

from model import ActorNetwork, load_checkpoint

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pendulum_ddpg.pt")
GIF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pendulum_demo.gif")
PNG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pendulum_demo.png")

DEMO_EPISODE = 3        # 演示几局
FPS = 30                # 存 GIF 时的帧率
SLEEP = 0.02            # human 模式下每步停一下，否则太快看不清

# 中文字体候选路径，找不到就退回英文标签（否则图上全是方块）
CJK_FONTS = [
    "/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Light.otf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


# ---------------------------------------------------------------------- #
# 加载模型
# ---------------------------------------------------------------------- #
def load_actor(path, device, obs_dim, action_dim, max_action):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "没找到模型文件：%s\n请先运行 train.py 训练出一个模型。" % path
        )
    ckpt = load_checkpoint(path, map_location=device)
    amodel = ActorNetwork(obs_dim, action_dim, max_action=max_action)
    amodel.load_state_dict(ckpt["actor"])
    amodel.to(device)
    amodel.eval()                                   # 演示只推理，不训练
    print("已加载模型：%s" % path)
    print("  训练时 episode = %s，评测平均回报 = %.1f"
          % (ckpt.get("episode", "?"), ckpt.get("score", float("nan"))))
    return amodel


def choose_action(amodel, obs, device):
    # 确定性策略：直接取 actor 的输出，不加噪声
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action = amodel(obs_t)
    return action.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------- #
# 跑一局
# ---------------------------------------------------------------------- #
def run_episode(env, amodel, device, want_frames=False, sleep=0.0):
    obs, _ = env.reset()
    obs_list = [obs]            # 长度 T+1（含初始状态）
    action_list = []            # 长度 T
    reward_list = []            # 长度 T
    frames = []
    total_reward = 0.0

    while True:
        action = choose_action(amodel, obs, device)
        obs, reward, terminated, truncated, _ = env.step(action)

        obs_list.append(obs)
        action_list.append(float(action[0]))
        reward_list.append(float(reward))
        total_reward += reward

        if want_frames:
            frames.append(env.render())             # rgb_array，返回 H×W×3 的 ndarray
        elif sleep > 0:
            time.sleep(sleep)                       # human 模式下放慢一点

        if terminated or truncated:
            break

    return (total_reward,
            np.asarray(obs_list),
            np.asarray(action_list),
            np.asarray(reward_list),
            frames)


# ---------------------------------------------------------------------- #
# 存动图
# ---------------------------------------------------------------------- #
def save_gif(frames, path, fps=FPS):
    if not frames:
        return False
    try:
        from PIL import Image
    except ImportError:
        print("没有装 Pillow，跳过动图。想存 GIF 请先 pip install pillow")
        return False
    images = [Image.fromarray(np.asarray(f, dtype=np.uint8)) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0)
    print("动图已保存：%s（%d 帧）" % (path, len(images)))
    return True


# ---------------------------------------------------------------------- #
# 画曲线
# ---------------------------------------------------------------------- #
def setup_chinese_font():
    from matplotlib import font_manager, rcParams
    for p in CJK_FONTS:
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
            rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=p).get_name()]
            rcParams["axes.unicode_minus"] = False       # 负号别显示成方块
            return True
    return False


def plot_process(obs_arr, action_arr, reward_arr, path, chinese=True):
    import matplotlib
    matplotlib.use("Agg")                            # 没有图形界面也能存图
    import matplotlib.pyplot as plt

    cos_t = obs_arr[:, 0]
    sin_t = obs_arr[:, 1]
    theta_dot = obs_arr[:, 2]
    # 由 cos/sin 还原出角度（弧度），画起来比两条曲线直观
    theta = np.arctan2(sin_t, cos_t)

    steps_obs = np.arange(len(obs_arr))
    steps_act = np.arange(1, len(action_arr) + 1)

    if chinese:
        L = dict(theta="摆角 θ（弧度）", omega="角速度 dθ/dt", act="力矩（动作）",
                 rew="每步奖励", x="步数", t="Pendulum 演示：状态与动作",
                 cum="累计回报 %.1f")
    else:
        L = dict(theta="angle theta (rad)", omega="angular velocity",
                 act="torque (action)", rew="step reward", x="step",
                 t="Pendulum demo: state and action", cum="return %.1f")

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(steps_obs, theta, color="#D85A30", linewidth=1.2)
    axes[0].axhline(0, color="#888780", linewidth=0.6, linestyle="--")   # 竖直向上
    axes[0].set_ylabel(L["theta"])

    axes[1].plot(steps_obs, theta_dot, color="#378ADD", linewidth=1.2)
    axes[1].axhline(0, color="#888780", linewidth=0.6, linestyle="--")
    axes[1].set_ylabel(L["omega"])

    axes[2].plot(steps_act, action_arr, color="#7F77DD", linewidth=1.2)
    axes[2].axhline(0, color="#888780", linewidth=0.6, linestyle="--")
    axes[2].set_ylim(-2.2, 2.2)                      # Pendulum 力矩上限 ±2
    axes[2].set_ylabel(L["act"])

    axes[3].plot(steps_act, reward_arr, color="#1D9E75", linewidth=1.2)
    axes[3].axhline(0, color="#888780", linewidth=0.6, linestyle="--")
    axes[3].set_ylabel(L["rew"])
    axes[3].set_xlabel(L["x"])

    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5)

    axes[0].set_title("%s   （%s）" % (L["t"], L["cum"] % float(reward_arr.sum())))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("曲线图已保存：%s" % path)


# ---------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="DDPG Pendulum 演示")
    parser.add_argument("--episodes", type=int, default=DEMO_EPISODE, help="演示几局")
    parser.add_argument("--mode", choices=["gif", "human", "none"], default="gif",
                        help="gif=存动图（默认），human=开窗口，none=只画曲线")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备: %s" % device)

    # ---- 建环境 ----
    want_frames = False
    if args.mode == "human":
        try:
            env = gym.make("Pendulum-v1", render_mode="human")
        except Exception as e:
            print("开不了窗口（%s），改成只画曲线" % e)
            env = gym.make("Pendulum-v1")
    elif args.mode == "gif":
        try:
            env = gym.make("Pendulum-v1", render_mode="rgb_array")
            env.reset()
            env.render()                             # 探一下能不能渲染，缺 pygame 会在这里抛
            want_frames = True
        except Exception as e:
            print("渲染画面不可用（%s），改成只画曲线" % e)
            env = gym.make("Pendulum-v1")
    else:
        env = gym.make("Pendulum-v1")

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    max_action = float(env.action_space.high[0])

    amodel = load_actor(MODEL_PATH, device, obs_dim, action_dim, max_action)

    # ---- 跑几局 ----
    returns = []
    best = None
    for i in range(args.episodes):
        total_reward, obs_arr, action_arr, reward_arr, frames = run_episode(
            env, amodel, device,
            want_frames=want_frames,
            sleep=SLEEP if args.mode == "human" else 0.0)

        returns.append(total_reward)
        print("第 %d 局：回报 %.1f（%d 步）" % (i + 1, total_reward, len(action_arr)))

        # 只画回报最高的那一局，免得几局叠在一起
        if best is None or total_reward > best[0]:
            best = (total_reward, obs_arr, action_arr, reward_arr, frames)

    env.close()

    # ---- 输出 ----
    print("----")
    print("平均回报 %.1f（随机策略约 -1200，训练好的能到 -200 以内）"
          % (sum(returns) / len(returns)))

    chinese = setup_chinese_font()
    if not chinese:
        print("没找到中文字体，图上的标签改用英文")

    total_reward, obs_arr, action_arr, reward_arr, frames = best
    print("用最好的一局画图：回报 %.1f" % total_reward)
    plot_process(obs_arr, action_arr, reward_arr, PNG_PATH, chinese=chinese)

    if want_frames:
        save_gif(frames, GIF_PATH)


if __name__ == "__main__":
    main()
