import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixNet(nn.Module):
    def __init__(self):
        super(QMixNet, self).__init__()
        self.n_agents = 2
        self.n_state = 96 * 2
        self.hidden = 128
        self.hyper_w1 = nn.Linear(96 * 2, 2 * 128)
        self.hyper_w2 = nn.Linear(96 * 2, 128)


        self.hyper_b1 = nn.Linear(96 * 2, 128)
        self.hyper_b2 = nn.Sequential(nn.Linear(96 * 2, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 1)
                                     )

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)
        batch_size = q_values.size(0)
        q_values = q_values.view(-1, 1, self.n_agents)  # n * 1 * 2
        states = states.reshape(-1, self.n_state)  # n * (2 * 96)

        w1 = torch.abs(self.hyper_w1(states))  # (n, 128 * 2)
        b1 = self.hyper_b1(states)  # (n, 128)

        w1 = w1.view(-1, self.n_agents, 128)  # (n, 2, 128)
        b1 = b1.view(-1, 1, 128)  # (n, 1, 128)

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (n, 1, 128)

        w2 = torch.abs(self.hyper_w2(states))  # (n, 128)
        b2 = self.hyper_b2(states)  # (n, 1)

        w2 = w2.view(-1, self.hidden, 1)  # ((n, 128, 1)
        b2 = b2.view(-1, 1, 1)  # (n, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2  # (n, 1, 1)
        q_total = q_total.squeeze(-1)  # (n, 1)
        return q_total

if __name__ == '__main__':
    states = torch.rand((32,96*2))
    q_values = torch.rand((32 , 2))
    net = QMixNet()
    net(q_values,states)