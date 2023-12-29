from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
import gym
import numpy as np
import torch
from PIL import Image
import os
from IPython.display import display, Image as IPImage
from DQN import DDQN_Agent
import numpy as np
layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
# layout = "counter_circuit_o_1order"

# Reward shaping is disabled by default.  This data structure may be used for
# reward shaping.  You can, of course, do your own reward shaping in lieu of, or
# in addition to, using this structure.
from stable_baselines3 import PPO
reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    # "DISH_DISP_DISTANCE_REW": -1,
    # "POT_DISTANCE_REW": -1,
    # "SOUP_DISTANCE_REW": -1,
}

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400

# Build the environment.  Do not modify!
mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
env = gym.make("Overcooked-v0", base_env=base_env,
               featurize_fn=base_env.featurize_state_mdp)

# 使用 Proximal Policy Optimization (PPO) 算法
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 保存模型
model.save("overcooked_model")

# 加载模型
loaded_model = PPO.load("overcooked_model")

# 测试模型
for e in range(10):
    obs = env.reset()
    rs = 0
    for _ in range(1000):
        action, _states = loaded_model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    # env.render()

# 关闭环境
env.close()