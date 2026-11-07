import torch.nn as nn
import torch
import torch.nn.functional as F
#状态是四维的
#这其实就是我们所谓的Q表格 只是用神经网络来进行计算了
class DQN_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(4,40)
        self.fc2=nn.Linear(40,40)
        self.fc3=nn.Linear(40,2)
    def forward(self,X):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(X)))))
