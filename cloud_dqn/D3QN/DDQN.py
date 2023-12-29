import math
import matplotlib.pyplot as plt

import torch
from torch import nn
import random
import numpy as np
from buffer1 import Memory
seed_value = 42
mask_num_per = 10  # 在这儿改数量

np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
device = 'cuda' if torch.cuda.is_available() else "cpu"
import matplotlib
matplotlib.use('TkAgg')
# 用于更新sumtree
def sumtree_loss(q,q_targ,entroy):
    return (torch.abs(q-q_targ) + 0.01*entroy.unsqueeze(-1)) * 0.5

def weight_loss(q,q_target,weight):
    return torch.mean((q-q_target)**2)

def get_entropy(q):
    # 使用 softmax 转换为概率分布
    action_probs = torch.softmax(q, dim=1)
    # 计算每个样本的熵
    entropies = -torch.sum(action_probs * torch.log(action_probs+1e-4), dim=1)
    return entropies

## 构建DQN网络
## get_entropy
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, act_dim,fc1_dim=256):
        super(DuelingDQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc1_dim)
        self.fc3 = nn.Linear(fc1_dim, fc1_dim)
        self.act = nn.ReLU()
        self.Q = nn.Linear(fc1_dim, act_dim)

    def forward(self,obs):
        # q = self.q(obs)
        x = self.act(self.fc1(obs))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))

        Q = self.Q(x)
        return Q

class DDQN_Agent:
    # def __init__(self, obs_dim, act_dim, buff_size=int(1e5), gamma=0.99, eps = 0.9, lr=1e-3, tau=0.02):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        ## 创建Q网络和target网络
        self.q = DuelingDQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ = DuelingDQN(self.state_space_dim, self.action_space_dim).to(device)
        self.q_targ.load_state_dict(self.q.state_dict())
        for parm in self.q_targ.parameters():
            parm.requires_grad = False
        # 创建优先级回放buffer
        if self.prioritized:
            self.buffer = Memory(max_size=self.capacity, batch_size=self.batch_size)
        else:
            self.buffer = []
        # MSE损失函数
        self.criterion = nn.MSELoss()
        # Adam优化器
        self.optim = torch.optim.Adam(self.q.parameters(), lr=self.lr, weight_decay=1e-5)
        self.tau = 0.005 # 软更新参数
        self.steps = 0

    # 存放训练样本
    def put(self, *transition):
        if self.prioritized:
            self.buffer.store_transition(*transition)
        else:
            if len(self.buffer) == self.capacity:
                self.buffer.pop(0)
            self.buffer.append(transition)

    def act(self, obs, action_mask=None):
        self.steps += 1
        epsi = self.epsi_low + (self.epsi_high - self.epsi_low) * (math.exp(-1.0 * self.steps / self.decay))
        if not action_mask is None: # 如果有动作MASK则开启
            ### 随机选一个动作
            if random.random() < epsi:
                p = np.zeros(self.action_space_dim) + 1/(self.action_space_dim - mask_num_per)
                p[action_mask] = 0
                a0 = np.random.choice(range(self.action_space_dim), p=p)
            else:
                s0 = torch.tensor(obs, dtype=torch.float).view(1, -1).to(device)
                q_val = self.q(s0).flatten()
                q_val[action_mask] = -100
                a0 = torch.argmax(q_val).item()
        else:
            ### 随机选一个动作
            if random.random() < epsi:
                a0 = random.randrange(self.action_space_dim)
            else:
                s0 = torch.tensor(obs, dtype=torch.float).view(1, -1).to(device)
                a0 = torch.argmax(self.q(s0)).item()
        return a0

    def update(self):

        ### 数据太少了，不训练
        if len(self.buffer) < self.batch_size:
            return
        if self.prioritized:
            # 获取训练样本
            tree_idx, minibatch, ISWeights = self.buffer.sample_buffer(self.batch_size)
            state = minibatch[:,  0:self.state_space_dim]
            action = minibatch[:, self.state_space_dim:self.state_space_dim + 1]
            reward = minibatch[:, self.state_space_dim + 1:self.state_space_dim + 2]
            next_state = minibatch[:, self.state_space_dim + 2:-1]
            done = minibatch[:, -1]
            ISWeights_tensor = torch.from_numpy(ISWeights).to(device).float()
        else:
            samples = random.sample(self.buffer, self.batch_size)
            state, action, reward, next_state, done = zip(*samples)

        # state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.from_numpy(np.array(state)).to(device).float()
        action = torch.tensor(action).to(device).view(self.batch_size, -1).long()
        reward = torch.tensor(reward).to(device).view(self.batch_size, -1).float()
        next_state = torch.from_numpy(np.array(next_state)).to(device).float()
        done = torch.tensor(done).to(device).float()


        with torch.no_grad():
            # 计算目标Q值
            q_targ = reward + (((1-done) * self.gamma) * torch.max(self.q_targ(next_state).detach(), dim=1)[0]).reshape(-1, 1)

        q = self.q(state)
        q_cur = q.gather(1, action) # 获取对应动作Q值

        ### 进行更新
        if self.prioritized:
            abs_errors = sumtree_loss(q_cur, q_targ, get_entropy(q))  # for updating Sumtree
            loss = weight_loss(q_cur, q_targ, ISWeights_tensor.flatten()) # 计算带权TD误差
            self.buffer.batch_update(tree_idx, abs_errors.detach().cpu().numpy())  # update priority
        else:
            loss = self.criterion(q_cur, q_targ)

        # 进行更新
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        ### 软更新taget
        for p,p_targ in zip(self.q.parameters(), self.q_targ.parameters()):
            p_targ.data = (1 - self.tau) * p_targ.data + self.tau * p.data




class CloudManufacturingEnvironment:
    def __init__(self, num_tasks=30, num_resources_per_task=5):
        self.num_tasks = num_tasks  # 子任务数量
        self.num_resources_per_task = num_resources_per_task  # 每个子任务的资源数量
        self.current_task = 0  # 当前子任务索引
        self.qos_values = np.random.rand(num_tasks, num_resources_per_task) # 随机生成的QoS值
        self.qos_values = (self.qos_values - np.mean(self.qos_values,axis=1)) / (np.std(self.qos_values,axis=1) + 1e-4) # 对奖励进行标准化
        # self.qos_values *= 0.2
        self.state = np.zeros(num_tasks)  # 初始化状态，表示每个子任务的资源匹配状态

    def step(self, action):
        done = False
        self.state = np.zeros(self.num_tasks)
        self.state[self.current_task] = 1  # 标记当前子任务的资源已匹配
        reward = self.qos_values[self.current_task, action]  # 奖励为该资源的QoS值
        self.current_task += 1

        if self.current_task == self.num_tasks:
            done = True  # 所有子任务都匹配完毕
        return self.state, reward, done

    def reset(self):
        self.state = np.zeros(self.num_tasks)  # 重置状态
        self.state[0] = 1
        self.current_task = 1
        return self.state
def soft(data):
    for i in range(1,len(data)):
        data[i] = data[i-1] * 0.8 + 0.2 * data[i]
    return data
def run():
    num_tasks = 30
    num_resources_per_task = 30
    input_size = num_tasks
    output_size = num_resources_per_task
    ### 超参数
    params = {
        'gamma': 0.9,
        'epsi_high': 1.0,
        'epsi_low': 0.001,
        'decay': 1000,
        'lr': 0.001,
        'capacity': 1000000,
        'batch_size': 128,
        'state_space_dim': input_size,
        'action_space_dim': output_size,
        'prioritized': True
    }

    agent = DDQN_Agent(**params)
    env = CloudManufacturingEnvironment(num_tasks=num_tasks, num_resources_per_task=num_resources_per_task)
    totalreward = []

    num_episodes = 300
    rewards = []
    # 动态调整mask
    actions_masks = [None for _ in range(num_tasks)]
    for i in range(num_tasks):
        random_number = np.random.choice(np.arange(num_tasks), size=mask_num_per, replace=False)
        actions_masks[i] = random_number
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state, action_mask=actions_masks[env.current_task]) # 获取动作，带有mask
            next_state, reward, done = env.step(action)
            agent.put(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.update()

        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        totalreward.append(total_reward)
    agent.q.eval()
    state = env.reset()
    acts = []
    while True:
        action = agent.act(state)
        acts.append(action)
        print("state: ", state)
        print("选择动作： ", action)
        state, reward, done = env.step(action)
        if done:
            break
    totalreward = soft(totalreward)
    print('最优动作集：',acts)
    plt.plot(totalreward)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()


if __name__ == "__main__":
    run()







