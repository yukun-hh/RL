from model import ActorNetwork
import numpy as np
import torch


class Agent():
    """只负责「和环境交互」这一半：产动作、加探索噪声、衰减噪声。

    学习那一半（trainq / traina）放在 train.py 里，
    因为一次更新要同时用到 actor、critic、两个目标网络和两个优化器，
    塞进 Agent 反而看不清数据流。

    参数解释：
        action_low / action_high  动作空间上下界，用来把加完噪声的动作 clip 回去
        noise_std                 高斯噪声的标准差，也就是探索强度
                                  注意：这**不是** DQN 里的 ε-greedy，
                                  名字一样但含义完全不同，别混用
        noise_std_min             噪声下限，不会衰减到 0
        noise_decay               每个 episode 乘一次的衰减系数
        env_step                  已经和环境交互了多少步
        learn_step                已经做了多少次梯度更新
    """

    def __init__(self, action_low=None, action_high=None,
                 noise_std=0.2, noise_std_min=0.05, noise_decay=0.995):
        self.env_step = 0
        self.learn_step = 0
        self.noise_std = noise_std
        self.noise_std_min = noise_std_min
        self.noise_decay_rate = noise_decay
        self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32)
        self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32)

    # 噪声衰减：乘性衰减，和 DQN 里的 ε 衰减一个套路，但不低于下限
    def noise_decay(self):
        self.noise_std = max(self.noise_std_min, self.noise_std * self.noise_decay_rate)
        return self.noise_std

    # 确定性动作，不加噪声，评测时用
    def predict(self, obs, amodel: ActorNetwork):
        device = next(amodel.parameters()).device   # parameter 返回的是迭代器，用 next 取一个
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)   # [1, obs_dim]
        with torch.no_grad():
            action = amodel(obs_t)
        return action.squeeze(0).cpu().numpy()      # [action_dim]

    # 训练时用的动作：确定性动作 + 高斯噪声，再 clip 回动作空间
    def sample(self, obs, amodel: ActorNetwork):
        action = self.predict(obs, amodel)
        noise = np.random.normal(0.0, self.noise_std, size=action.shape).astype(np.float32)
        action = action + noise
        if self.action_low is not None:
            np.clip(action, self.action_low, self.action_high, out=action)
        return action
