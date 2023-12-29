import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class Memory(object):
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        clipped_errors = np.minimum(abs_errors.data, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_feature, n_hidden)
        self.values = nn.Linear(n_hidden, 1)
        self.advantages = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        value = self.values(x)
        advantages = self.advantages(x)
        out = value + (advantages-torch.mean(advantages, dim=1, keepdim=True))
        return out

class PrioritizedReplayD3QN():
    def __init__(self, n_actions, n_features, n_hidden=20, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
                 replace_target_iter=200, memory_size=500, batch_size=32, e_greedy_increment=None,
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.memory = Memory(capacity=memory_size)
        self.loss_func = nn.MSELoss()
        self.cost_his = []
        self._build_net()
    def _build_net(self):
        self.q_eval = Net(self.n_features, self.n_hidden, self.n_actions)
        self.q_target = Net(self.n_features, self.n_hidden, self.n_actions)
        self.optimizer = torch.optim.RMSprop(self.q_eval.parameters(), lr=self.lr)

    def store_transition(self, s, a, r, s_):
        r = float(r)
        transition = np.hstack((s, [a, r], s_))
        self.memory.store(transition)

    def choose_action(self, observation,e_step,sfc_placement_row,node_path_remove):
        observation = torch.Tensor(observation[np.newaxis, :])
        actions_value = self.q_eval(observation)
        action_value_noisy = actions_value.data.numpy() + np.random.randn(1, self.n_actions) * (1. / (e_step + 1))
        # action_value_noisy = actions_value.data.numpy() + np.random.randn(1, self.n_actions)
        for k in range(14):
            if k in sfc_placement_row or k in node_path_remove:
                action_value_noisy[0][k] = -float('inf')    # 负无穷
        action=np.argmax(action_value_noisy)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        q_next, q_eval4next = self.q_target(torch.Tensor(batch_memory[:, -self.n_features:])), self.q_eval(
            torch.Tensor(batch_memory[:, -self.n_features:]))
        q_eval = self.q_eval(torch.Tensor(batch_memory[:, :self.n_features]))
        q_target = torch.Tensor(q_eval.data.numpy().copy())
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = torch.Tensor(batch_memory[:, self.n_features + 1])
        max_act4next = torch.max(q_eval4next, dim=1)[1]
        selected_q_next = q_next[batch_index, max_act4next]
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        self.abs_errors = torch.sum(torch.abs(q_target - q_eval), dim=1)
        loss=torch.mean(torch.mean(torch.Tensor(ISWeights) * (q_target - q_eval) ** 2, dim=1))
        self.memory.batch_update(tree_idx, self.abs_errors)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.cost_his.append(loss)
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()