import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
class DQNAgent:
    def __init__(self, input_size, output_size,update_target_every=10, gamma=0.99, epsilon_start=1.0, epsilon_end=0.00, epsilon_decay=0.999, learning_rate=0.0001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        # self.target_net.load_state_dict(self.policy_net.state_dict())
        policy_state_dict = OrderedDict(self.policy_net.state_dict())
        self.target_net.load_state_dict(policy_state_dict)
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.output_size = output_size  # 使用 output_size 替代 num_actions
        self.update_target_every = update_target_every
        self.update_count = 0

    def select_action(self, state):
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                # print("policy_net(state)",self.policy_net(state))
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.output_size)  # 使用 output_size 作为动作数量

    def train(self, batch_size, replay_buffer):
        if len(replay_buffer) < batch_size:
            return
        if self.update_count % self.update_target_every == 0:
            self.update_target_network()
        self.update_count += 1
        transitions = replay_buffer.sample(batch_size)
        batch = list(zip(*transitions))

        state_batch = torch.tensor(batch[0], dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.long).to(self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(batch[3], dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.uint8).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        # print("current_q_values",current_q_values)
        next_q_values = torch.zeros(batch_size, device=self.device)
        next_q_values[~done_batch] = self.target_net(next_state_batch).max(1)[0].detach()
        # print("next_q_values", next_q_values)
        expected_q_values = (next_q_values * self.gamma) + reward_batch

        loss = nn.functional.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
        # print("loss",loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        policy_state_dict = OrderedDict(self.policy_net.state_dict())
        self.target_net.load_state_dict(policy_state_dict)
        # self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

class CloudManufacturingEnvironment:
    def __init__(self, num_tasks=30, num_resources_per_task=5):
        self.num_tasks = num_tasks  # 子任务数量
        self.num_resources_per_task = num_resources_per_task  # 每个子任务的资源数量
        self.current_task = 0  # 当前子任务索引
        self.qos_values = np.random.rand(num_tasks, num_resources_per_task)  # 随机生成的QoS值
        self.qos_values = (self.qos_values - np.mean(self.qos_values, axis=1)) / (
                    np.std(self.qos_values, axis=1) + 1e-4)
        self.state = np.zeros(num_tasks)  # 初始化状态，表示每个子任务的资源匹配状态

    def step(self, action):
        reward = 0
        done = False

        if self.state[self.current_task] == 0:
            self.state[self.current_task] = 1  # 标记当前子任务的资源已匹配
            reward = self.qos_values[self.current_task, action]  # 奖励为该资源的QoS值
            self.current_task += 1  # 转到下一个子任务

        if self.current_task >= self.num_tasks:
            done = True  # 所有子任务都匹配完毕

        return self.state, reward, done

    def reset(self):
        self.state = np.zeros(self.num_tasks)  # 重置状态
        self.current_task = 0  # 重置当前子任务索引
        return self.state

def run():
    num_tasks = 30
    num_resources_per_task = 30
    input_size = num_tasks
    output_size = num_resources_per_task
    agent = DQNAgent(input_size, output_size)
    replay_buffer = ReplayBuffer(capacity=1000000)
    env = CloudManufacturingEnvironment(num_tasks=num_tasks, num_resources_per_task=num_resources_per_task)
    totalreward = []

    num_episodes = 1500
    batch_size = 64
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            # print("action:",action)
            next_state, reward, done = env.step(action)
            # print("next_state:",next_state)
            replay_buffer.push(state, action, reward / 10, next_state, done)
            state = next_state
            total_reward += reward
            if episode % 2 == 0:
                for _ in range(10):
                    agent.train(batch_size, replay_buffer)
            agent.decay_epsilon()

        agent.update_target_network()
        rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}  epsilon:{agent.epsilon}")
        totalreward.append(total_reward)

    plt.plot(totalreward)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

if __name__ == "__main__":
    run()
