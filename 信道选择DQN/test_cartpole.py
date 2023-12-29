import math
import random
import torch
import gym
from DQN import DDQN_Agent
import matplotlib.pyplot as plt
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    params = {
        'gamma': 0.9,
        'epsi_high': 1.0,
        'epsi_low': 0.01,
        'decay': 10000,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n,
    }
    # agent = DDQN_Agent(state_dim,act_dim)

    agent = DDQN_Agent(**params)
    score = []
    mean = []
    for episode in range(300):
        s0 = env.reset()
        total_reward = 0
        while True:
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            # if s1[0] <= -0.5:
            #     r1 += abs(s1[1])
            # elif -0.5 < s1[0] < 0.5:
            #     r1 = math.pow(2, (s1[0] + 1)) + abs(s1[1]) ** 2
            # else:
            #     r1 = 10
            # if done:
            #     r1 = 1
            # Modifying reward. If a cart reaches 0.5 distance or higher - it gets additional reward
            reward = 100 * ((np.sin(3 * s1[0]) * 0.0025 + 0.5 * s1[1] * s1[1]) - (
                        np.sin(3 * s0[0]) * 0.0025 + 0.5 * s0[1] * s0[1]))
            if s1[0] >= 0.5:
                reward += 1
            # r1 = modified_reward(r1,s1)
            agent.put(s0, a0, reward, s1, done)
            total_reward += r1

            if done:
                break

            s0 = s1
            agent.update()

        score.append(total_reward)
        mean.append(sum(score[-10:]) / 10)
        print(f'{episode} reward :',total_reward)
    # plot(score, mean)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(score[10:])
    plt.plot(mean[10:])
    # plt.text(len(score) - 1, score[-1], str(score[-1]))
    # plt.text(len(mean) - 1, mean[-1], str(mean[-1]))
    plt.savefig("rewards_episode.png")



