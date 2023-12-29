import numpy as np
from algo import ValueFunctionWithApproximation

# import tensorflow as
import torch
from torch import nn
class ValueFunctionWithNN(ValueFunctionWithApproximation,nn.Module):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        super(ValueFunctionWithNN, self).__init__()
        self.v = nn.Sequential(
            nn.Linear(state_dims, 32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.loss_fn = nn.MSELoss()
        self.optim = None
    def __call__(self,s):
        # TODO: implement this method
        self.v.eval()
        s = torch.from_numpy(s).float()
        return self.v(s).item()

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        self.v.train()
        if isinstance(G, float):
            G = torch.Tensor([G]).float()
        elif not isinstance(G, torch.Tensor):
            G = torch.from_numpy(G).float()
        self.optim = torch.optim.Adam(self.v.parameters(), lr = alpha, betas=[0.9,0.999])
        s_tau = torch.from_numpy(s_tau).float()

        for i in range(20):
            loss = self.loss_fn(G, self.v(s_tau)) * 0.5
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

