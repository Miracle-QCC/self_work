import numpy as np
import torch
import sys

sys.path.append('../../')
import rl.policies.core_cuda as core

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
        self.tree[tree_idx] = np.sqrt(abs(p))
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += abs(change) * 0.5

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
        self.max_size = max_size

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
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        # print("total:", self.tree.total_p)
        if min_prob == 0:
            min_prob = 0.001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            try:
                v = np.random.uniform(a, b)
            except:
                print(self.tree.total_p, a,b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(abs(prob), self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        self.tree.tree = np.zeros(2 * self.max_size - 1)
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha1)
        for ti, p in zip(tree_idx, ps):
            # print(p)
            self.tree.update(ti, p)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.

    with Running Mean and Var from hill-a/stable-baselines
    """

    def __init__(self, obs_dim, act_dim, size, clip_limit, norm_update_every=1000):
        """
        :param obs_dim: observation dimension
        :param act_dim: action dimension
        :param size: buffer sizes
        :param clip_limit: limit for clip value
        :param norm_update_every: update freq
        """
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        # Running z-score normalization parameters
        self.clip_limit = clip_limit
        self.norm_update_every = norm_update_every
        self.norm_update_batch = np.zeros(core.combined_shape(norm_update_every, obs_dim), dtype=np.float32)
        self.norm_update_count = 0
        self.norm_total_count = np.finfo(np.float32).eps.item()
        self.mean, self.var = np.zeros(obs_dim, dtype=np.float32), np.ones(obs_dim, dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        """
        Insert entry into memory
        :param obs: observation
        :param act: action
        :param rew: reward
        :param next_obs: observation after action
        :param done: if true then episode done
        """
        self.obs_buf[self.ptr] = obs.squeeze()
        self.obs2_buf[self.ptr] = next_obs.squeeze()
        self.act_buf[self.ptr] = act.squeeze()
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # Update Mean and Variance
        # Have to at least update mean and variance once before training starts
        self.norm_update_batch[self.norm_update_count] = obs
        self.norm_update_count += 1
        if self.norm_update_count == self.norm_update_every:
            self.norm_update_count = 0
            batch_mean, batch_var = self.norm_update_batch.mean(axis=0), self.norm_update_batch.var(axis=0)
            tmp_total_count = self.norm_total_count + self.norm_update_every
            delta_mean = batch_mean - self.mean
            self.mean += delta_mean * (self.norm_update_every / tmp_total_count)
            m_a = self.var * self.norm_total_count
            m_b = batch_var * self.norm_update_every
            m_2 = m_a + m_b + np.square(delta_mean) * self.norm_total_count * self.norm_update_every / tmp_total_count
            self.var = m_2 / tmp_total_count
            self.norm_total_count = tmp_total_count

    def sample_batch(self, device, batch_size=32):
        """
        Sample batch from memory
        :param device: pytorch device
        :param batch_size: batch size
        :return: batch
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.normalize_obs(self.obs_buf[idxs]),
                     obs2=self.normalize_obs(self.obs2_buf[idxs]),
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}

    def normalize_obs(self, obs):
        """
        Do z-score normalization on observation
        :param obs: observation
        :return: norm_obs
        """
        eps = np.finfo(np.float32).eps.item()
        norm_obs = np.clip((obs - self.mean) / np.sqrt(self.var + eps),
                           -self.clip_limit, self.clip_limit)
        return norm_obs
