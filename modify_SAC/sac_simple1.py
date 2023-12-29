import gym
import numpy as np
import scipy.signal
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import random
from torch.optim import Adam
from copy import deepcopy
import matplotlib.pyplot as plt



def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.fc = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = F.relu(self.fc(obs))
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


class Critic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, obs, act):
        # q = self.q(torch.cat([obs, act], dim=-1))
        x = torch.cat([obs, act], dim=-1)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        q = self.out(out)
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256,256),
                 activation=nn.ELU, act_limit = 2.0):
        super().__init__()

        # build policy and value functions
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = Critic(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.detach().cpu().numpy()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SAC:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=MLPActorCritic, seed=0,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.3):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.capacity = replay_size
        self.time = 0
        act_bound = torch.FloatTensor(act_bound).to(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.ac = actor_critic(obs_dim, act_dim, act_limit=act_bound).to(device)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = self.ac.q.parameters()

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Experience buffer
        self.replay_buffer = []

    def store(self,*sample):
        if len(self.replay_buffer) == self.capacity:
            self.replay_buffer.remove(self.replay_buffer[0])
        self.replay_buffer.append(sample)

    # def sample_batch(self,batch_size):
    #     batch = random.sample(self.replay_buffer,batch_size)
    #     return batch

    # Set up function for computing SAC_baseline Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = zip(*data)
        with torch.no_grad():
            o = torch.from_numpy(np.array(o)).to(device)
            a = torch.from_numpy(np.array(a)).to(device)
            r = torch.from_numpy(np.array(r)).to(device)
            o2 = torch.from_numpy(np.array(o2)).to(device)
            d = torch.from_numpy(np.array(d)).to(device).float()

        q = self.ac.q(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac_targ.pi(o2)

            # Target Q-values
            q_pi_targ = self.ac_targ.q(o2, a2)
            #####   My modified location
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()
        #writer.add_scalar('loss_q',loss_q,global_step=self.time)
        # Useful info for logging
        #q_info = dict(Q1Vals=q1.detach().numpy(),
                      #Q2Vals=q2.detach().numpy())

        return loss_q
    # Set up function for computing SAC_baseline pi loss
    def compute_loss_pi(self, data):
        o,_,_,_,_ = zip(*data)
        o = torch.from_numpy(np.array(o)).to(device)
        pi, logp_pi = self.ac.pi(o)
        q_pi = self.ac.q(o, pi)

        # loss_pi = (self.alpha * logp_pi - q_pi).mean()
        data = random.sample(self.replay_buffer, batch_size)
        self.q_optimizer.zero_grad()
        loss_pi = -q_pi.mean()
        return loss_pi
    def update(self, batch_size):
        self.time += 1
        # First run one gradient descent step for Q1 and Q2
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, deterministic=False):
        o = torch.FloatTensor(o).to(device)
        return self.ac.act(o,deterministic)

def eval_actor():
    o = env.reset()
    ep_reward = 0
    with torch.no_grad():
        for j in range(MAX_STEP):
            if episode > 10:
                a = sac.get_action(o, deterministic=True)
            else:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            sac.store(o, a, r, o2, d)
            o = o2
            ep_reward += r
            if d:
                break
    return ep_reward
def soft_data(data):
    n = len(data)
    mom = data[0]
    for i in range(1,n):
        data[i] = 0.6 * mom + data[i] * 0.4
        mom = mom * 0.6 + 0.4 * data[i]

    return data
if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_bound = (env.action_space.high - env.action_space.low) / 2.0


    sac = SAC(obs_dim, act_dim, act_bound)
    trainTimes = 0
    MAX_EPISODE = int(800)
    MAX_TotalStep = int(1000)
    MAX_STEP = 200
    MaxTrain = 10
    batch_size = 256
    rewardList = []
    ep_rewards = []
    for episode in range(MAX_EPISODE):
        o = env.reset()
        ep_reward = 0
        for j in range(MAX_STEP):
            if episode > 10:
                a = sac.get_action(o)
            else:
                a = env.action_space.sample()
            o2, r, d, _ = env.step(a)
            sac.store(o, a, r, o2, d)
            o = o2
            # ep_reward += r
            if d:
                break
        ep_reward = eval_actor()
        ep_rewards.append(ep_reward)
        print(f"{episode}/{MAX_EPISODE} :{ep_reward}")

        if episode >= 10:
            for i in range(MaxTrain):  # Training 50 times per round
                trainTimes += 1
                sac.update(batch_size)
    ep_rewards = soft_data(ep_rewards)

    plt.plot(ep_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    # plt.show()
    plt.savefig("score_figure1.jpg")

