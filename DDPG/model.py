import torch
import torch.nn as nn
import torch.nn.functional as F

"""
DDPG 的两个网络（以 Pendulum-v1 为例：状态 3 维、动作 1 维、动作范围 ±2）

Actor：输入状态，输出动作的确定性值 μ(s)。
       DDPG 的动作分布是「固定方差的高斯」，所以网络只需要给出均值，
       方差由探索噪声在采样时加进去，网络本身不用管。
       注意最后一层必须用 tanh 把输出压到 [-max_action, max_action]，
       否则网络可以输出任意大的动作，和环境动作空间对不上。

Critic：输入 (状态, 动作)，输出 Q(s,a)。
       动作是连续的，没法像 DQN 那样对每个动作算 Q 再取 max，
       所以直接把动作拼进输入，当成一个普通回归问题来学。
"""


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim=3, action_dim=1, hidden=64, max_action=2.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
        self.max_action = max_action
        # 最后一层用小初始化：开局 tanh 的输入接近 0，动作就从 0 附近开始，不会一上来顶着边界
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, X):
        h = F.relu(self.fc1(X))
        h = F.relu(self.fc2(h))
        return self.max_action * torch.tanh(self.fc3(h))


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim=3, action_dim=1, hidden=64):
        super().__init__()
        # 输入维度 = 状态维度 + 动作维度，在 train.py 里用 torch.cat 拼好再喂进来
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        # Q 值量级大概在 [-2000, 0]，输出层初始化太大开局 TD 误差会直接爆
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, X):
        h = F.relu(self.fc1(X))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def load_checkpoint(path, map_location="cpu"):
    """读 checkpoint，兼容老版本存进去的 numpy 标量。

    torch 2.6 起 `torch.load` 默认 `weights_only=True`，只允许反序列化张量和
    基本类型。老存档里的 score 是 `sum(rewards)/len(rewards)` 的结果，类型是
    np.float64，于是会报 `Unsupported global: numpy._core.multiarray.scalar`。

    这里先按安全模式读，失败了再退回完整反序列化。
    **只对本地自己训练出来的存档这么干**，来路不明的 .pt 不要用第二个分支。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)
