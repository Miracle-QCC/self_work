import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else "cpu"

class RND(nn.Module):
    def __init__(self, obs_dim):
        super(RND, self).__init__()
        self.net_work1 = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
        )

        self.net_work2 = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        self.optimi = torch.optim.Adam(self.net_work1.parameters(), lr = 1e-3)
        self.batch_size = 64
        for p in self.net_work2.parameters():
            p.requires_grad = False


    def forward(self, obs):
        # obs = torch.from_numpy(obs).float().to(device)
        f1 = self.net_work1(obs)
        f2 = self.net_work2(obs)

        return f1,f2

    def value(self,obs):
        f1, f2 = self.forward(obs)
        loss = (f1 - f2) ** 2

        return loss.mean(dim=-1)

    def update(self, agent=None, states=None):
        if agent:
            if len(agent.buffer) < self.batch_size:
                return
            samples = random.sample(agent.buffer, self.batch_size)
            state, action, reward, next_state, done = zip(*samples)
            state = torch.from_numpy(np.array(state)).to(device).float()
            f1,f2 = self.forward(state)
        if states:
            if len(states) < self.batch_size:
                return
            samples = random.sample(states, self.batch_size)
            state = torch.from_numpy(np.stack(samples)).to(device).float()
            f1, f2 = self.forward(state)
        loss = F.l1_loss(f1,f2)
        self.optimi.zero_grad()
        loss.backward()
        self.optimi.step()
