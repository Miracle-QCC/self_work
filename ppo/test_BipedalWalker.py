import gym
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from ppo_agent import PPOTrainer

env = gym.make("BipedalWalker-v3")
env.seed(42)
env.action_space.seed(42)
torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    trainer = PPOTrainer(env)
    ## load checkpoint
    trainer.load_checkpoint('actor_checkpoint_last.pth','critic_checkpoint_last.pth')
    trainer.policy.eval()

    for i in range(10):
        done = False
        ep_r = 0
        s = env.reset()


        while not done:
            a = trainer.get_action(s)
            res = env.step(a)
            # print(a)
            if len(res) == 4:
                s,r,done,_ = res
            else:
                s,r,done,_,_ = res
            ep_r += r
            env.render()
        print("get rewards : ",ep_r)
