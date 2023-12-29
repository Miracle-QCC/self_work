import math

import torch
from torch import nn
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else "cpu"

# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0
#
#     def push(self, state, action, reward, next_state, done):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = (state, action, reward, next_state, done)
#         self.position = (self.position + 1) % self.capacity
#
#
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = zip(*batch)
#         state = np.array(state, dtype=np.float32)
#         action = np.array(action, dtype=np.float32)
#         reward = np.array(reward, dtype=np.float32)
#         next_state = np.array(next_state, dtype=np.float32)
#         done = np.array(done, dtype=np.float32)
#
#         return state, action, reward, next_state, done
#
#     def __len__(self):
#
#         return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, state_dim, act_dim):
        super(DQN, self).__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,act_dim)
        )

    def forward(self,obs):
        q = self.q(obs)
        return q


class DDQN_Agent:
    # def __init__(self, obs_dim, act_dim, buff_size=int(1e5), gamma=0.99, eps = 0.9, lr=1e-3, tau=0.02):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        ## 创建Q网络和target网络
        self.q = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ.load_state_dict(self.q.state_dict())
        for parm in self.q_targ.parameters():
            parm.requires_grad = False
        self.buffer = []
        self.criterion = nn.MSELoss()
        self.optim = torch.optim.Adam(self.q.parameters(), lr=self.lr)
        self.tau = 0.02
        self.steps = 0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def act(self, obs):

        ### 随机选一个动作
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(obs, dtype=torch.float).view(1, -1).to(device)
            a0 = torch.argmax(self.q(s0)).item()
        return a0

    def update(self, curious_net=None):

        ### 数据太少了，不训练
        if len(self.buffer) < self.batch_size:
            return
        samples = random.sample(self.buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*samples)

        # state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.from_numpy(np.array(state)).to(device).float()
        action = torch.tensor(action).to(device).view(self.batch_size, -1).long()
        reward = torch.tensor(reward).to(device).view(self.batch_size, -1).float()
        next_state = torch.from_numpy(np.array(next_state)).to(device).float()
        done = torch.tensor(done).to(device).float()

        with torch.no_grad():
            if curious_net:
                curious_value = curious_net.value(next_state)
                q_targ = reward + (
                            ((1 - done) * self.gamma) * (torch.max(self.q_targ(next_state).detach(), dim=1)[0] + curious_value)).reshape(-1, 1)

            else:
                q_targ = reward + (((1-done) * self.gamma) * torch.max(self.q_targ(next_state).detach(), dim=1)[0]).reshape(-1, 1)

        q = self.q(state)
        q_cur = q.gather(1, action)

        ### 进行更新
        loss = self.criterion(q_cur, q_targ)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        ### 软更新taget
        for p,p_targ in zip(self.q.parameters(), self.q_targ.parameters()):
            p_targ.data = (1 - self.tau) * p_targ.data + self.tau * p.data













