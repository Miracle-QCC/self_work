import numpy as np
import torch
from torch import nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

### define a Actor-Critic model
class ACModel(nn.Module):
    def __init__(self, env, action_dim, action_std_init):
        super().__init__()
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        self.actor = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
    ## get the var of the action
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)

    def forward(self, obs):
        action_mean = self.actor(obs)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        ## sample a action
        return MultivariateNormal(action_mean, cov_mat), self.critic(obs).reshape(-1)


class PPOTrainer:
    def __init__(self, env):
        self.env = env
        self.rewards = []
        self.gamma = 0.99
        self.lamda = 0.95
        self.worker_steps = 2000
        self.n_mini_batch = 10
        self.learn_epochs = 30 # Old sample learning times
        self.batch_size = self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        self.policy = ACModel(env, env.action_space.shape[0], 0.5).to(device)
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 0.0001}, # optim of actor
            {'params': self.policy.critic.parameters(), 'lr': 0.001} # optim of critic
        ])
        self.policy_old = ACModel(env, env.action_space.shape[0], 0.5).to(device)  # old model
        self.policy_old.load_state_dict(self.policy.state_dict())  # copy weight
        self.all_episode_rewards = []
        self.all_mean_rewards = []
        self.episode = 0
        self.obs = None
        self.a_loss = []
        self.c_loss = []

    def save_checkpoint(self):
        """
        save checkpoint
        :return:
        """
        a_filename = 'actor_checkpoint_last.pth'
        c_filename = 'citic_checkpoint_last.pth'

        torch.save(self.policy_old.actor.state_dict(), f=a_filename)
        torch.save(self.policy_old.critic.state_dict(), f=c_filename)

        print('Checkpoint saved ')

    def load_checkpoint(self, a_filename, c_filename):
        self.policy_old.actor.load_state_dict(torch.load(a_filename))
        self.policy_old.critic.load_state_dict(torch.load(c_filename))
        print('Resuming training from checkpoint {}   {}'.format(a_filename, c_filename))

    def get_action(self, state):
        with torch.no_grad():
            s = torch.from_numpy(state).to(device)
            a = self.policy_old.actor(s)
        return a.cpu().numpy()


    ### Calculate Adv based on GAE
    def calculate_advantages(self, done, rewards, values):
        _, last_value = self.policy_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
        last_value = last_value.cpu().data.numpy()
        values = np.append(values, last_value)
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - done[i]
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.lamda * mask * gae
            returns.insert(0, gae + values[i])
        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-6)

    def train(self, samples, clip_range):
        indexes = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indexes = indexes[start: end]
            mini_batch = {}
            for k, v in samples.items():
                mini_batch[k] = v[mini_batch_indexes]
            for _ in range(self.learn_epochs):
                loss = self.calculate_loss(clip_range=clip_range, samples=mini_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.policy_old.load_state_dict(self.policy.state_dict())

    ## ppo loss
    def calculate_loss(self, samples, clip_range):
        sampled_returns = samples['returns']
        sampled_advantages = samples['advantages']
        pi, value = self.policy(samples['obs'])
        ratio = torch.exp(pi.log_prob(samples['actions']) - samples['log_pis'])
        clipped_ratio = ratio.clamp(min=1.0 - clip_range, max=1.0 + clip_range) # clip KL
        policy_reward = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)
        entropy_bonus = pi.entropy() # action entropy
        vf_loss = self.mse_loss(value, sampled_returns) # value loss
        loss = -policy_reward + 0.5 * vf_loss - 0.01 * entropy_bonus  # total loss

        self.a_loss.append(policy_reward.mean().item())
        self.c_loss.append(vf_loss.mean().item())
        return loss.mean()


