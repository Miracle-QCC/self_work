"""Proximal Policy Optimization (clip objective)."""
from copy import deepcopy
import itertools

import torch
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import kl_divergence

from torch.nn.utils.rnn import pad_sequence

import time

import numpy as np
import os, sys

import ray

from rl.envs import WrapEnv

from rl.policies.actor import Gaussian_FF_Actor, Gaussian_LSTM_Actor
from rl.policies.critic import FF_V, LSTM_V
from rl.envs.normalize import get_normalization_params, PreNormalizer

import pickle
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha1 = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, max_size, batch_size):
        # max_size--经验池大小
        self.tree = SumTree(max_size)
        # batch_size--一次采样的经验数量
        self.batch_size = batch_size
        # mem_cnt--经验池中的经验数量计数器
        self.mem_cnt = 0

    def __len__(self):
        return self.mem_cnt

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s.flatten(), a, r, s_.flatten(), done))
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p
        self.mem_cnt += 1

    def sample_buffer(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / self.batch_size  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        min_prob += 1e-4
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)

            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha1)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class SAC:
    def __init__(self, args, save_path):
        self.env_name = args['env_name']
        self.gamma = args['gamma']
        self.lr = args['lr']
        self.eps = args['eps']
        self.entropy_coeff = args['entropy_coeff']
        self.clip = args['clip']
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.num_steps = args['num_steps']
        self.max_traj_len = args['max_traj_len']
        self.n_proc = args['num_procs']
        self.recurrent = args['recurrent']
        self.buffer = Memory(max_size=1e6, batch_size=args['batch_size'])
        self.total_steps = 0
        self.highest_reward = -1
        self.limit_cores = 0

        self.save_path = save_path

    def save(self, policy, critic1,critic2):

        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt"  # pytorch model
        torch.save(policy, os.path.join(self.save_path, "actor" + filetype))
        torch.save(critic1, os.path.join(self.save_path, "critic1" + filetype))
        torch.save(critic2, os.path.join(self.save_path, "critic2" + filetype))


    @ray.remote
    @torch.no_grad()
    def sample(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0, term_thresh=0):
        """
        Sample at least min_steps number of total timesteps, truncating
        trajectories only if they exceed max_traj_len number of timesteps
        """
        torch.set_num_threads(1)  # By default, PyTorch will use multiple cores to speed up operations.
        # This can cause issues when Ray also uses multiple cores, especially on machines
        # with a lot of CPUs. I observed a significant speedup when limiting PyTorch
        # to a single core - I think it basically stopped ray workers from stepping on each
        # other's toes.

        env = WrapEnv(env_fn)  # TODO

        memory = []
        num_steps = 0
        while num_steps < min_steps:
            state = torch.Tensor(env.reset())

            done = False
            traj_len = 0
            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()
            if hasattr(critic, 'init_hidden_state'):
                critic.init_hidden_state()
            while not done and traj_len < max_traj_len:
                action = policy(state, deterministic=deterministic)
                # value = critic(state)
                next_state, reward, done, _ = env.step(action.numpy(), term_thresh=term_thresh)
                memory.append(state.numpy(), action.numpy(), reward, next_state, done)
                state = torch.Tensor(next_state)
                traj_len += 1
                num_steps += 1
        return memory

    def sample_parallel(self, env_fn, policy, critic, min_steps, max_traj_len, deterministic=False, anneal=1.0,
                        term_thresh=0):
        worker = self.sample
        args = (
            self, env_fn, policy, critic, min_steps // self.n_proc, max_traj_len, deterministic, anneal, term_thresh)

        # Create pool of workers, each getting data for min_steps
        workers = [worker.remote(*args) for _ in range(self.n_proc)]

        result = []
        total_steps = 0
        rewards = []

        self.ep_returns = []  # for logging
        self.ep_lens = []
        while total_steps < min_steps:
            # get result from a worker
            ready_ids, _ = ray.wait(workers, num_returns=1)

            # update result
            result.append(ray.get(ready_ids[0]))

            # remove ready_ids from workers (O(n)) but n isn't that big
            workers.remove(ready_ids[0])

            # update total steps
            total_steps += len(result[-1])

            # start a new worker
            workers.append(worker.remote(*args))

        # O(n)
        def merge(buffers):
            # merged = PPOBuffer(self.gamma, self.lam)
            for buf in buffers:
                state, action, reward, next_state, done = zip(*buf)
                self.buffer.store_transition(state, action, reward, next_state, done)
                rewards += reward

        # 将收集的数据存入sac的buff中
        merge(result)
        return rewards
        # Set up function for computing SAC_baseline Q-losses

    def compute_loss_q(self, data, q1, q2, q1_targ, q2_targ, pi):
        o, a, r, o2, d = zip(*data)
        with torch.no_grad():
            o = torch.FloatTensor(o)
            a = torch.FloatTensor(a)
            r = torch.FloatTensor(r)
            o2 = torch.FloatTensor(o2)
            d = torch.FloatTensor(d)
        q1 = q1(o, a)
        q2 = q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = pi(o2)

            # Target Q-values
            q1_pi_targ = q1_targ(o2, a2)
            q2_pi_targ = q2_targ(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            #####   My modified location
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        # writer.add_scalar('loss_q',loss_q,global_step=self.time)
        # Useful info for logging
        # q_info = dict(Q1Vals=q1.detach().numpy(),
        # Q2Vals=q2.detach().numpy())

        return loss_q

    def compute_loss_pi(self, data, q1, q2, pi):
        o, _, _, _, _ = zip(*data)
        o = torch.FloatTensor(o)
        pi, logp_pi = pi(o)
        q1_pi = q1(o, pi)
        q2_pi = q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        #####   My modified location
        loss_pi = (self.alpha * logp_pi - q_pi).mean()
        # writer.add_scalar('-loss_pi', -loss_pi, global_step=self.time)
        # Useful info for logging
        # pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi

    def train(self,
              env_fn,
              policy,
              policy_targ,
              q1,
              q1_targ,
              q2,
              q2_targ,
              n_itr,
              logger=None, anneal_rate=1.0):
        self.pi = policy
        self.pi_targ = policy_targ
        self.q1 = q1
        self.q1_targ = q1_targ
        self.q2 = q2
        self.q2_targ = q2_targ

        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_targ_params = itertools.chain(self.q1_targ.parameters(), self.q2_targ.parameters())
        self.actor_optimizer = optim.Adam(self.pi.parameters(), lr=self.lr, eps=self.eps)
        self.critic_optimizer = optim.Adam(self.q_params, lr=self.lr, eps=self.eps)
        start_time = time.time()

        env = env_fn()
        obs_mirr, act_mirr = None, None
        if hasattr(env, 'mirror_observation'):
            if env.clock_based:
                obs_mirr = env.mirror_clock_observation
            else:
                obs_mirr = env.mirror_observation

        if hasattr(env, 'mirror_action'):
            act_mirr = env.mirror_action

        curr_anneal = 1.0
        curr_thresh = 0
        start_itr = 0
        ep_counter = 0
        do_term = False
        for itr in range(n_itr):
            print("********** Iteration {} ************".format(itr))

            if self.highest_reward > (2 / 3) * self.max_traj_len and curr_anneal > 0.5:
                curr_anneal *= anneal_rate
            if do_term and curr_thresh < 0.35:
                curr_thresh = .1 * 1.0006 ** (itr - start_itr)
            batch = self.sample_parallel(env_fn, self.policy, self.critic, self.num_steps, self.max_traj_len,
                                         anneal=curr_anneal, term_thresh=curr_thresh)

            print("time elapsed: {:.2f} s".format(time.time() - start_time))

            optimizer_start = time.time()
            for epoch in range(self.epochs):
                losses = []
                tree_idx, minibatch, ISWeights = self.buffer.sample_buffer(self.batch_size)
                state = minibatch[:, 0:self.state_space_dim]
                action = minibatch[:, self.state_space_dim:self.state_space_dim + 1]
                reward = minibatch[:, self.state_space_dim + 1:self.state_space_dim + 2]
                next_state = minibatch[:, self.state_space_dim + 2:-1]
                done = minibatch[:, -1]

                data = zip(state, action, reward, next_state, done)
                loss_q = self.compute_loss_q(data, self.q1, self.q2, self.q1_targ, self.q2_targ, self.pi)

                self.critic_optimizer.zero_grad()
                loss_q.backward()
                self.critic_optimizer.step()

                loss_pi = self.compute_loss_pi(data, self.q1, self.q2, self.pi)
                self.actor_optimizer.zero_grad()
                loss_pi.backward()
                self.actor_optimizer.step()
                losses.append([loss_q.item(), loss_pi.item()])

                # TODO: add verbosity arguments to suppress this
                print(' '.join(["%g" % x for x in np.mean(losses, axis=0)]))

            opt_time = time.time() - optimizer_start
            print("optimizer time elapsed: {:.2f} s".format(opt_time))

            if np.mean(batch.ep_lens) >= self.max_traj_len * 0.75:
                ep_counter += 1
            if do_term == False and ep_counter > 50:
                do_term = True
                start_itr = itr

            if logger is not None:
                evaluate_start = time.time()
                test = self.sample_parallel(env_fn, self.pi, self.critic, self.num_steps // 2, self.max_traj_len,
                                            deterministic=True)
                eval_time = time.time() - evaluate_start
                print("evaluate time elapsed: {:.2f} s".format(eval_time))

                avg_eval_reward = np.mean(test[])
                avg_batch_reward = np.mean(batch)
                # avg_ep_len = np.mean(batch.ep_lens)
                mean_losses = np.mean(losses, axis=0)
                # print("avg eval reward: {:.2f}".format(avg_eval_reward))

                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (test)', avg_eval_reward) + "\n")
                sys.stdout.write("| %15s | %15s |" % ('Return (batch)', avg_batch_reward) + "\n")
                # sys.stdout.write("| %15s | %15s |" % ('Mean Eplen', avg_ep_len) + "\n")
                # sys.stdout.write("| %15s | %15s |" % ('Mean KL Div', "%8.3g" % kl) + "\n")
                # sys.stdout.write("| %15s | %15s |" % ('Mean Entropy', "%8.3g" % entropy) + "\n")
                sys.stdout.write("-" * 37 + "\n")
                sys.stdout.flush()

                entropy = np.mean(entropies)
                kl = np.mean(kls)

                logger.add_scalar("Test/Return", avg_eval_reward, itr)
                logger.add_scalar("Train/Return", avg_batch_reward, itr)
                # logger.add_scalar("Train/Mean Eplen", avg_ep_len, itr)
                logger.add_scalar("Train/Mean KL Div", kl, itr)
                logger.add_scalar("Train/Mean Entropy", entropy, itr)

                logger.add_scalar("Misc/Critic Loss", mean_losses[2], itr)
                logger.add_scalar("Misc/Actor Loss", mean_losses[0], itr)
                logger.add_scalar("Misc/Mirror Loss", mean_losses[5], itr)
                logger.add_scalar("Misc/Timesteps", self.total_steps, itr)

                # logger.add_scalar("Misc/Sample Times", samp_time, itr)
                logger.add_scalar("Misc/Optimize Times", opt_time, itr)
                logger.add_scalar("Misc/Evaluation Times", eval_time, itr)
                logger.add_scalar("Misc/Termination Threshold", curr_thresh, itr)

            # TODO: add option for how often to save model
            if self.highest_reward < avg_eval_reward:
                self.highest_reward = avg_eval_reward
                self.save(self.pi, self.q1, self.q2)


def run_experiment(args):
    from util.env import env_factory
    from util.log import create_logger

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, simrate=args.simrate, command_profile=args.command_profile,
                         input_profile=args.input_profile, learn_gains=args.learn_gains,
                         dynamics_randomization=args.dyn_random, reward=args.reward, history=args.history,
                         mirror=args.mirror, ik_baseline=args.ik_baseline, no_delta=args.no_delta, traj=args.traj)
    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]

    # Set up Parallelism
    os.environ['OMP_NUM_THREADS'] = '1'
    if not ray.is_initialized():
        if args.redis_address is not None:
            ray.init(num_cpus=args.num_procs, redis_address=args.redis_address)
        else:
            ray.init(num_cpus=args.num_procs)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.previous is not None:
        policy = torch.load(os.path.join(args.previous, "actor.pt"))
        critic = torch.load(os.path.join(args.previous, "critic.pt"))
        # TODO: add ability to load previous hyperparameters, if this is something that we event want
        # with open(args.previous + "experiment.pkl", 'rb') as file:
        #     args = pickle.loads(file.read())
        print("loaded model from {}".format(args.previous))
    else:
        if args.recurrent:
            policy = Gaussian_LSTM_Actor(obs_dim, action_dim, fixed_std=np.exp(-2), env_name=args.env_name)
            critic = LSTM_V(obs_dim)
        else:
            if args.learn_stddev:
                policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=None, env_name=args.env_name,
                                           bounded=args.bounded)
            else:
                policy = Gaussian_FF_Actor(obs_dim, action_dim, fixed_std=np.exp(args.std_dev), env_name=args.env_name,
                                           bounded=args.bounded)
            critic = FF_V(obs_dim)

        with torch.no_grad():
            policy.obs_mean, policy.obs_std = map(torch.Tensor,
                                                  get_normalization_params(iter=args.input_norm_steps, noise_std=1,
                                                                           policy=policy, env_fn=env_fn,
                                                                           procs=args.num_procs))
        critic.obs_mean = policy.obs_mean
        critic.obs_std = policy.obs_std

    policy.train()
    critic.train()

    print("obs_dim: {}, action_dim: {}".format(obs_dim, action_dim))

    # create a tensorboard logging object
    logger = create_logger(args)

    algo = PPO(args=vars(args), save_path=logger.dir)

    print()
    print("Synchronous Distributed Proximal Policy Optimization:")
    print(" ├ recurrent:      {}".format(args.recurrent))
    print(" ├ run name:       {}".format(args.run_name))
    print(" ├ max traj len:   {}".format(args.max_traj_len))
    print(" ├ seed:           {}".format(args.seed))
    print(" ├ num procs:      {}".format(args.num_procs))
    print(" ├ lr:             {}".format(args.lr))
    print(" ├ eps:            {}".format(args.eps))
    print(" ├ lam:            {}".format(args.lam))
    print(" ├ gamma:          {}".format(args.gamma))
    print(" ├ learn stddev:  {}".format(args.learn_stddev))
    print(" ├ std_dev:        {}".format(args.std_dev))
    print(" ├ entropy coeff:  {}".format(args.entropy_coeff))
    print(" ├ clip:           {}".format(args.clip))
    print(" ├ minibatch size: {}".format(args.minibatch_size))
    print(" ├ epochs:         {}".format(args.epochs))
    print(" ├ num steps:      {}".format(args.num_steps))
    print(" ├ use gae:        {}".format(args.use_gae))
    print(" ├ max grad norm:  {}".format(args.max_grad_norm))
    print(" └ max traj len:   {}".format(args.max_traj_len))
    print()

    algo.train(env_fn, policy, critic, args.n_itr, logger=logger, anneal_rate=args.anneal)
