""" Deep RL Algorithms for OpenAI Gym environments
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from args import Args
from dqn import DQN
from Environment import Environment

action_dim=3
state_dim=(3,)
args=Args()
def main():
    # Pick algorithm to train
    env = Environment(r"D:\PycharmProjects\pythonProject2\DDQN\sumo\example.sumocfg")
    summary_writer = tf.summary.FileWriter("ddqn"+ "/tensorboard_" + "sumo")
    algo = DQN(action_dim, state_dim, args)
    #     # Train
    stats = algo.train(env, args, summary_writer)
if __name__ == "__main__":
    main()
