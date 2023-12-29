import time

import gym
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ppo_agent import PPOTrainer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make("BipedalWalker-v3")
# env.seed(42)
env.action_space.seed(42)
torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    trainer = PPOTrainer(env)
    ep_rs = []
    episode = 0
    start_time = time.time()
    while True:
        episode += 1
        # collect data
        rewards = np.zeros(trainer.worker_steps, dtype=np.float32)
        actions = np.zeros((trainer.worker_steps, env.action_space.shape[0]), dtype=np.float32)
        done = np.zeros(trainer.worker_steps, dtype=bool)
        obs = np.zeros((trainer.worker_steps, env.observation_space.shape[0]), dtype=np.float32)
        log_pis = np.zeros(trainer.worker_steps, dtype=np.float32)
        values = np.zeros(trainer.worker_steps, dtype=np.float32)
        trainer.obs = env.reset()
        if isinstance(trainer.obs, tuple):
            trainer.obs = trainer.obs[0]
        ep_r = 0
        for t in range(trainer.worker_steps):
            with torch.no_grad():
                obs[t] = trainer.obs
                pi, v = trainer.policy_old(torch.tensor(trainer.obs, dtype=torch.float32, device=device).unsqueeze(0))
                values[t] = v.cpu().numpy()
                a = pi.sample()
                actions[t] = a.cpu().numpy()
                log_pis[t] = pi.log_prob(a).cpu().numpy()
            trainer.obs, rewards[t], done[t], _ = env.step(actions[t])
            if isinstance(trainer.obs, tuple):
                trainer.obs = trainer.obs[0]
            # env.render()
            ep_r += rewards[t]

            trainer.rewards.append(rewards[t])
            if done[t]:
                trainer.episode += 1
                trainer.all_episode_rewards.append(np.sum(trainer.rewards))
                trainer.rewards = []
                env.reset()
                # break
                ep_r = 0
        print('Episode: {}, average reward: {}'.format(episode, ep_r))
        ## get returns ,adv
        returns, advantages = trainer.calculate_advantages(done, rewards, values)
        samples = {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_pis': torch.tensor(log_pis, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'returns': torch.tensor(returns, device=device, dtype=torch.float32)
        }
        trainer.train(samples, 0.2)

        ep_rs.append(ep_r)
        if ep_r >= 300:
            break
    end_time = time.time()
    training_time = end_time - start_time
    np.savez("train_data.npz",scores = np.array(ep_rs), actor_loss = np.array(trainer.a_loss), critic_loss=np.array(trainer.c_loss))
    print(f"Training time: {training_time} seconds")

    trainer.save_checkpoint()
    plt.figure(1)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.plot(ep_rs)
    # plt.savefig("episode_reward.png")
    # plt.clf()
    plt.figure(2)
    plt.xlabel('episode')
    plt.ylabel('actor loss')
    plt.plot(trainer.a_loss)
    # plt.savefig("actor_loss.png")
    # plt.clf()

    plt.figure(3)
    plt.xlabel('episode')
    plt.ylabel('critic loos')
    plt.plot(trainer.c_loss)
    # plt.savefig("critic_loss.png")
    # plt.clf()

    plt.show()



