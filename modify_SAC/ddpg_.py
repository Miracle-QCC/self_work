import os
import random
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import argparse
import datetime
import time
import torch.optim as optim
import torch.nn.functional as F
import gym
from torch import nn
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_rewards(rewards):
    sns.set()
    plt.figure()
    plt.title(u"{}".format('Pendulum-v1'))
    plt.xlabel(u'epochs')
    plt.plot(rewards)
    plt.plot(smooth(rewards))
    plt.legend(('rewards'), loc="best")

    plt.savefig(f"ing_curve_cn.png")


def smooth(data, weight=0.9):

    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)
        return action

    def reverse_action(self, action):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action


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
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):

        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action.reshape(-1,1)], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DDPG:
    def __init__(self, n_states, n_actions, arg_dict):
        self.device = torch.device(arg_dict['device'])
        self.critic = Critic(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)
        self.actor = Actor(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)
        self.target_critic = Critic(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)
        self.target_actor = Actor(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=arg_dict['critic_lr'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=arg_dict['actor_lr'])
        self.memory = ReplayBuffer(arg_dict['memory_capacity'])
        self.batch_size = arg_dict['batch_size']
        self.soft_tau = arg_dict['soft_tau']  # 软更新参数
        self.gamma = arg_dict['gamma']

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0, 0]

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach()).detach()
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    # def save_model(self, path):
    #     Path(path).mkdir(parents=True, exist_ok=True)
    #     torch.save(self.actor.state_dict(), path + 'checkpoint.pt')
    #
    # def load_model(self, path):
    #     self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))



def train(arg_dict, env, agent):
    startTime = time.time()
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    print("开始训练智能体......")
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(arg_dict['train_eps']):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            if arg_dict['train_render']:
                env.render()
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        if (i_ep + 1) % 10 == 0:
            print(f'Env:{i_ep + 1}/{arg_dict["train_eps"]}, Reward:{ep_reward:.2f}')
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('训练结束 , 用时: ' + str(time.time() - startTime) + " s")
    # 关闭环境
    env.close()
    return {'episodes': range(len(rewards)), 'rewards': rewards, 'ma_rewards': ma_rewards}


# 测试函数
def test(arg_dict, env, agent):
    startTime = time.time()
    print("开始测试智能体......")
    print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(arg_dict['test_eps']):
        state = env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        while not done:
            if arg_dict['test_render']:
                env.render()
            i_step += 1
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        print(f"Epside:{i_ep + 1}/{arg_dict['test_eps']}, Reward:{ep_reward:.1f}")
    print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
    env.close()
    return {'episodes': range(len(rewards)), 'rewards': rewards}


# 创建环境和智能体
def create_env_agent(arg_dict):
    env = NormalizedActions(gym.make(arg_dict['env_name']))
    env.seed(arg_dict['seed'])
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    agent = DDPG(n_states, n_actions, arg_dict)
    return env, agent


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # 相关参数设置
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='Pendulum-v1', type=str, help="name of environment")
    parser.add_argument('--train_eps', default=100, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=8000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=520, type=int, help="seed")
    parser.add_argument('--show_fig', default=False, type=bool, help="if show figure or not")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    parser.add_argument('--train_render', default=False, type=bool,
                        help="Whether to render the environment during training")
    parser.add_argument('--test_render', default=True, type=bool,
                        help="Whether to render the environment during testing")
    args = parser.parse_args()
    default_args = {'result_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/results/",
                    'model_path': f"{curr_path}/outputs/{args.env_name}/{curr_time}/models/",
                    }
    # 将参数转化为字典 type(dict)
    arg_dict = {**vars(args), **default_args}
    print("算法参数字典:", arg_dict)

    # 创建环境和智能体
    env, agent = create_env_agent(arg_dict)
    # 传入算法参数、环境、智能体，然后开始训练
    res_dic = train(arg_dict, env, agent)
    print("算法返回结果字典:", res_dic)
    # 保存相关信息
    # agent.save_model(path=arg_dict['model_path'])
    plot_rewards(res_dic['rewards'])
# rds(res_dic['rewards'])
