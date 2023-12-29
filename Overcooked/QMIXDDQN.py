import math
from QMIX import QMixNet
import torch
from torch import nn
import random
import numpy as np
from DQN import DQN
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
device = 'cuda' if torch.cuda.is_available() else "cpu"



class Qmix_DDQN_Agent:
    # def __init__(self, obs_dim, act_dim, buff_size=int(1e5), gamma=0.99, eps = 0.9, lr=1e-3, tau=0.02):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        ## 创建Q网络和target网络
        self.q1 = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ1 = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ1.load_state_dict(self.q1.state_dict())
        for parm in self.q_targ1.parameters():
            parm.requires_grad = False

        self.q2 = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ2 = DQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ2.load_state_dict(self.q2.state_dict())
        for parm in self.q_targ2.parameters():
            parm.requires_grad = False

        self.qmix = QMixNet().to(device)
        self.buffer = []
        self.criterion = nn.MSELoss()
        self.optim1 = torch.optim.Adam(self.q1.parameters(), lr=self.lr)
        self.optim2 = torch.optim.Adam(self.q2.parameters(), lr=self.lr)
        self.optim_total = torch.optim.Adam(self.qmix.parameters(), lr=self.lr)

        self.tau = 0.005
        self.steps = 0


    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def act(self, obs1, obs2):

        ### 随机选一个动作
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if random.random() < epsi:
            a0 = random.randrange(self.action_space_dim)
        else:
            s0 = torch.tensor(obs1, dtype=torch.float).view(1, -1).to(device)
            a0 = torch.argmax(self.q1(s0)).item()

        if random.random() < epsi:
            a1 = random.randrange(self.action_space_dim)
        else:
            s1 = torch.tensor(obs2, dtype=torch.float).view(1, -1).to(device)
            a1 = torch.argmax(self.q2(s1)).item()
        return a0, a1

    def logp(self, q_cur, q):
        p = torch.exp(q)/ torch.sum(torch.exp(q),dim=-1).reshape(-1,1)
        entropy = - p * torch.log2(p)
        return entropy.mean()


    def update(self, curious_net=None):

        ### 数据太少了，不训练
        if len(self.buffer) < self.batch_size:
            return
        samples = random.sample(self.buffer, self.batch_size)
        state1,state2, action1,action2, reward1,reward2, next_state1,next_state2, done = zip(*samples)

        # state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.from_numpy(np.concatenate((state1,state2))).to(device).float()
        state1 = torch.from_numpy(np.array(state1)).to(device).float()
        state2 = torch.from_numpy(np.array(state2)).to(device).float()
        action1 = torch.tensor(action1).to(device).view(self.batch_size, -1).long()
        action2 = torch.tensor(action2).to(device).view(self.batch_size, -1).long()

        reward1 = torch.tensor(reward1).to(device).view(self.batch_size, -1).float()
        reward2 = torch.tensor(reward2).to(device).view(self.batch_size, -1).float()

        next_state = torch.from_numpy(np.concatenate((next_state1, next_state2), axis=-1)).to(device).float()
        next_state1 = torch.from_numpy(np.array(next_state1)).to(device).float()
        next_state2 = torch.from_numpy(np.array(next_state2)).to(device).float()

        done = torch.tensor(done).to(device).float().view(-1,1)

        with torch.no_grad():

            if curious_net:
                curious_value = curious_net.value(next_state).view(-1,1)
                q_targ1 = reward1 + (
                            ((1 - done) * self.gamma) * (
                            torch.max(self.q_targ1(next_state1).detach(), dim=1)[0]).reshape(-1, 1))
                q_targ2 = reward2 + (
                        ((1 - done) * self.gamma) * (
                            torch.max(self.q_targ2(next_state2).detach(), dim=1)[0]).reshape(-1, 1))

                q_total_target = self.qmix(torch.cat([q_targ1,q_targ2],dim=-1), next_state) + curious_value * 0.1
            else:
                q_targ1 = reward1 + (((1-done) * self.gamma) * torch.max(self.q_targ1(next_state1).detach(), dim=1)[0]).reshape(-1, 1)
                q_targ2 = reward2 + (((1-done) * self.gamma) * torch.max(self.q_targ2(next_state2).detach(), dim=1)[0]).reshape(-1, 1)
                q_total_target = self.qmix(torch.cat([q_targ1, q_targ2], dim=-1),
                                           next_state).view(-1,1)

        q1 = self.q1(state1)
        q2 = self.q2(state2)

        q1_cur = q1.gather(1, action1)
        q2_cur = q2.gather(1, action2)

        entry1 = self.logp(q1_cur, q1)
        entry2 = self.logp(q2_cur, q2)
        q_total = self.qmix(torch.cat([q1_cur,q2_cur],dim=-1), state)

        ### 进行更新
        loss = self.criterion(q_total, q_total_target) - 0.1 * entry1 - 0.1 * entry2
        self.optim1.zero_grad()
        self.optim2.zero_grad()
        self.optim_total.zero_grad()

        loss.backward()

        self.optim1.step()
        self.optim2.step()
        self.optim_total.step()

        ### 软更新taget
        for p,p_targ in zip(self.q1.parameters(), self.q_targ1.parameters()):
            p_targ.data = (1 - self.tau) * p_targ.data + self.tau * p.data

        ### 软更新taget
        for p, p_targ in zip(self.q2.parameters(), self.q_targ2.parameters()):
            p_targ.data = (1 - self.tau) * p_targ.data + self.tau * p.data












