import torch
from torch.optim import Adam
from copy import deepcopy
import itertools
import core as core
import numpy as np
import random
# from tensorboardX import SummaryWriter

# writer = SummaryWriter('runs/loss')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
alpha = torch.tensor(0.2,dtype=torch.float32).to(device) #CQL-parameters
gamma = torch.tensor(1, dtype=torch.float32).to(device)
class SAC:
    def __init__(self, obs_dim, act_dim, act_bound, actor_critic=core.MLPActorCritic, seed=0,
                 replay_size=int(1e7), gamma=0.99, polyak=0.995, lr=1e-3, alpha=0.3):
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
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Experience buffer
        self.replay_buffer = []

    def store(self,*sample):
        if len(self.replay_buffer) == self.capacity:
            self.replay_buffer.remove(self.replay_buffer[0])
        self.replay_buffer.append(sample)

    def store_batch(self,*samples):
        o, a, r, o2 = samples
        for i in range(len(o)):
            if len(self.replay_buffer) == self.capacity:
                self.replay_buffer.remove(self.replay_buffer[0])

            self.replay_buffer.append([o[i], a[i],r[i],o2[i], False])
    # def sample_batch(self,batch_size):
    #     batch = random.sample(self.replay_buffer,batch_size)
    #     return batch

    # Set up function for computing SAC_baseline Q-losses
    def compute_loss_q(self, data):
        o, a, e, o2, d = zip(*data)
        with torch.no_grad():
            o = torch.from_numpy(np.array(o)).float().to(device)
            a = torch.from_numpy(np.array(a)).float().to(device)
            e = torch.from_numpy(np.array(e)).float().to(device)
            o2 = torch.from_numpy(np.array(o2)).float().to(device)
            # d = torch.from_numpy(d).to(device)
        q1 = self.ac.q1(o,a)
        q2 = self.ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)
            #
            # # Target Q-values
            # q1_pi_targ = self.ac_targ.q1(o2, a2)
            # q2_pi_targ = self.ac_targ.q2(o2, a2)
            # q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # #####   My modified location
            # backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)
            Gt_1 = torch.max(self.ac_targ.q1(o2,a2), self.ac_targ.q2(o2,a2))

            backup = torch.max(Gt_1, e)

        ###Calculating CQL Loss1
        q1_pred_actions = self.ac.q1(o, a)
        q1_curr_actions = self.ac.q1(o, self.ac.pi(o)[0])
        q1_next_curr_actions = self.ac.q1(o, self.ac.pi(o2)[0])
        cat_q1 = torch.cat(
            [q1_pred_actions, q1_next_curr_actions, q1_curr_actions], dim=-1
        )
        min_qf1_loss = torch.logsumexp(cat_q1, dim=-1).mean() * alpha  # logsumexp()
        min_qf1_loss = min_qf1_loss - (q1_pred_actions.mean() * alpha)

        ###Calculating CQL Loss2
        q2_pred_actions = self.ac.q2(o, a)
        q2_curr_actions = self.ac.q2(o, self.ac.pi(o)[0])
        q2_next_curr_actions = self.ac.q2(o, self.ac.pi(o2)[0])
        cat_q2 = torch.cat(
            [q2_pred_actions, q2_next_curr_actions, q2_curr_actions], dim=-1
        )
        min_qf2_loss = torch.logsumexp(cat_q2, dim=-1).mean() * alpha  # logsumexp()
        min_qf2_loss = min_qf2_loss - (q2_pred_actions.mean() * alpha)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2 + min_qf1_loss + min_qf2_loss
        #writer.add_scalar('loss_q',loss_q,global_step=self.time)
        # Useful info for logging
        #q_info = dict(Q1Vals=q1.detach().numpy(),
                      #Q2Vals=q2.detach().numpy())

        return loss_q
    # Set up function for computing SAC_baseline pi loss
    def compute_loss_pi(self, data):
        o,_,_,_,_ = zip(*data)
        o = torch.from_numpy(np.array(o)).float().to(device)
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        #####   My modified location
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        #writer.add_scalar('-loss_pi', -loss_pi, global_step=self.time)
        # Useful info for logging
        #pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi
    def update(self, batch_size):
        self.time += 1
        # First run one gradient descent step for Q1 and Q2
        data = random.sample(self.replay_buffer, batch_size)
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        # for p in self.q_params:
        #     p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        # 
        # # Unfreeze Q-networks so you can optimize it at next DDPG step.
        # for p in self.q_params:
        #     p.requires_grad = True

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