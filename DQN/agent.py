from model import DQN_model
import numpy as np
import torch
class Agent():
    def __init__(self,obs_n,act_n,model,lr=0.001,gamma=0.9,e_greed=0.1,
                 e_greed_min=0.01,e_greed_decay=0.995):
        self.act_n = act_n
        self.lr = lr
        self.gamma = gamma
        self.e_greed = e_greed
        self.e_greed_min = e_greed_min        # 探索率的下限，不会衰减到 0
        self.e_greed_decay = e_greed_decay    # 每次衰减的系数
        self.learn_step = 0                   # 已经做了多少次梯度下降（用来决定何时同步目标网络）
        self.env_step = 0                     # 已经和环境交互了多少步（用来判断暖启动是否结束）
        self.optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        self.loss_fn = torch.nn.MSELoss()

    #探索率衰减：每调用一次就乘一个衰减系数，但不低于下限
    def decay_e_greed(self):
        self.e_greed = max(self.e_greed_min, self.e_greed*self.e_greed_decay)
        return self.e_greed

    #采样部分
    def sample(self,obs,model):
        if np.random.uniform(0,1)<(1.0-self.e_greed) :
            action =self.predict(obs,model)
        else:
            action = np.random.randint(self.act_n)
        return action
    def predict(self,obs,model):
        model.eval()                                    # 推理模式
        device = next(model.parameters()).device        # 跟着模型走：有 CUDA 就在 CUDA 上算
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            Q = model(obs_t)
        Q = Q.squeeze(0).cpu().numpy()                  # 转 numpy 前必须先回到 CPU
        maxQ = np.max(Q)
        action_list = np.where(Q == maxQ)[0]
        return int(np.random.choice(action_list))
    def train(self,batch,model,model_fix):
        obs_b,act_b,rew_b,next_obs_b,done_b = batch
        device     = next(model.parameters()).device    # 张量要和模型在同一个设备上
        obs_b      = torch.as_tensor(np.asarray(obs_b),      dtype=torch.float32, device=device)
        act_b      = torch.as_tensor(np.asarray(act_b),      dtype=torch.long,    device=device).unsqueeze(1)   # [B,1]
        rew_b      = torch.as_tensor(np.asarray(rew_b),      dtype=torch.float32, device=device).unsqueeze(1)  # [B,1]
        next_obs_b = torch.as_tensor(np.asarray(next_obs_b), dtype=torch.float32, device=device)
        done_b     = torch.as_tensor(np.asarray(done_b),     dtype=torch.float32, device=device).unsqueeze(1)  # [B,1]
        model.train()
        q_value = model(obs_b)
        q_sa = q_value.gather(1,act_b) # 选择每一个在obs状态小选择的路线
        # 接下来要用固定模型去计算next_obs 原来的Q值 来更新新的obs的q值
        # q值应该接近于 gammma*max_next_q+r
        with torch.no_grad():
            next_q = model_fix(next_obs_b)
            max_nex_q = next_q.max(dim=1,keepdim=True)[0] # max语法 第一个是值 第二个是位置 上次就已经错过y一次了
            target_q = rew_b+self.gamma*(max_nex_q)*(1.0-done_b)
        loss = self.loss_fn(q_sa,target_q)
        self.optimizer.zero_grad()                      # 清空上一步的梯度，否则梯度会一直累加
        loss.backward()
        self.optimizer.step()
        self.learn_step += 1                            # 一次 train 就是一次梯度下降

        return loss.item()
