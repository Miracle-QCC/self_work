import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections
# hyper-parameters
EPISODES = 1000  # 训练/测试幕数
BATCH_SIZE = 64
LR = 0.00025
GAMMA = 0.98
SAVING_IETRATION = 500  # 保存Checkpoint的间隔
MEMORY_CAPACITY = 10000  # Memory的容量
MIN_CAPACITY = 100  # 开始学习的下限
Q_NETWORK_ITERATION = 10  # 同步target network的间隔
EPSILON = 1.0  # epsilon-greedy
SEED = 0
MODEL_PATH = ''
SAVE_PATH_PREFIX = './log/dqn/'
TEST = True

env = gym.make('MountainCar-v0', render_mode="human" if TEST else None,max_episode_steps=500)
# env = gym.make('CartPole-v1')
# env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
# env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)


random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

NUM_ACTIONS = env.action_space.n  # 2
NUM_STATES = env.observation_space.shape[0]  # 4
ENV_A_SHAPE = 0 if np.issubdtype(type(env.action_space.sample()),
                                 np.integer) else env.action_space.sample().shape  # 0, to confirm the shape


class Model(nn.Module):
    def __init__(self, num_inputs=4):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(NUM_STATES, 256)
        self.linear2 = nn.Linear(256, 256)
        self.V = nn.Linear(256, 1)
        self.A = nn.Linear(256, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        A = self.A(x)
        V = self.V(x)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)
        return Q


class Data:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        # TODO
        self.buffer.append(data)

    def get(self, batch_size):
        # TODO
        sample = random.sample(self.buffer, batch_size)
        return sample


class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Model().to(device), Model().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = Memory(capacity=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPSILON = 1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        else:
            # random policy
            action = np.random.randint(0,NUM_ACTIONS)  # int random number
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1

    def learn(self):
        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

        self.learn_step_counter += 1

        # TODO
        data = self.memory.get(BATCH_SIZE)
        states = np.array([s.state for s in data])
        actions = np.array([s.action for s in data])
        rewards = np.array([s.reward for s in data])
        next_states = np.array([s.next_state for s in data])
        dones = np.array([s.done for s in data])

        # get tensor
        states = torch.from_numpy(np.array(states)).to(device).float()
        actions = torch.tensor(actions).to(device).view(BATCH_SIZE, -1).long()
        rewards = torch.tensor(rewards).to(device).view(BATCH_SIZE, -1).float()
        next_states = torch.from_numpy(np.array(next_states)).to(device).float()
        dones = torch.tensor(dones).to(device).float()

        with torch.no_grad():
            q_targ = rewards + (((1-dones) * GAMMA) * torch.max(self.target_net(next_states).detach(), dim=1)[0]).reshape(-1, 1)

        q = self.eval_net(states)
        q_cur = q.gather(1, actions)

        ### 进行更新
        loss = self.loss_func(q_cur, q_targ)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))

def main():
    global EPSILON
    dqn = DQN()
    dqn.load_net('log/double_Dueling_dqn/ckpt/20000.pth')
    # writer = SummaryWriter(f'{SAVE_PATH_PREFIX}')

    # if TEST:
    #     dqn.load_net(MODEL_PATH)
    for i in range(EPISODES):
        print("EPISODE: ", i)
        # state, info = env.reset(seed=SEED)
        state = env.reset()

        if isinstance(state, tuple):
            state = state[0]
        ep_reward = 0
        while True:
            # print(state)
            EPSILON = max(0.01, EPSILON)
            action = dqn.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)  # choose best action
            next_state, reward, done, truncated, info = env.step(action)  # observe next state and reward
            # next_state, reward, done, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            dqn.store_transition(Data(state, action, reward, next_state, done))
            ep_reward += reward
            if TEST:
                env.render()
            if dqn.memory_counter >= MIN_CAPACITY and not TEST:
                dqn.learn()
                if done:
                    EPSILON *= 0.9
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                if TEST:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
                break
            state = next_state
        # writer.add_scalar('reward', ep_reward, global_step=i)


if __name__ == '__main__':
    main()