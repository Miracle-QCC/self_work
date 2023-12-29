from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from matplotlib.pyplot import plot as plt
# There already exists an environment generator
# that will make and wrap atari environments correctly.
vec_env = make_atari_env("PongDeterministic-v4", n_envs=1, seed=0, monitor_dir="path_to_log_dir")
# Frame-stacking with 4 frames
vec_env = VecFrameStack(vec_env, n_stack=4)
# vec_env = Monitor(vec_env, "logs/")

model = PPO("CnnPolicy", vec_env, verbose=1, )
model.learn(total_timesteps=200000)
model.save("ppo_pong_model")
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")

