import random

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo import MAPPO_Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.agent import NNPolicy, AgentFromPolicy, AgentPair
import gym
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
device = 'cuda' if torch.cuda.is_available() else "cpu"

# layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
layout = "counter_circuit_o_1order"
reward_shaping = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 1,
    "POT_DISTANCE_REW": 1,
    "SOUP_DISTANCE_REW": 1,
}

# Length of Episodes.  Do not modify for your submission!
# Modification will result in a grading penalty!
horizon = 400
num_episodes = 2000
# Build the environment.  Do not modify!
mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
env = gym.make("Overcooked-v0", base_env=base_env,
               featurize_fn=base_env.featurize_state_mdp)

class Runner_MAPPO_OverCooked:
    def __init__(self, args, env_name='OverCooked'):
        self.args = args
        self.env_name = env_name
        # self.number = number
        # self.seed = seed
        # # Set random seed
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # Create env
        self.env = env
        # self.env_info = self.env.get_env_info()
        self.args.N = 2  # The number of agents
        self.args.obs_dim = 96   # The dimensions of an agent's observation space
        self.args.state_dim = 96 * 2  # The dimensions of global state space
        self.args.action_dim = 6  # The dimensions of an agent's action space
        self.args.episode_limit = horizon  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        # Create N agents
        self.agent_n = MAPPO_Overcooked(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(log_dir='runs/MAPPO/MAPPO_env_{}'.format(self.env_name))

        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_overcooked(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        soups_ = 0
        for _ in range(self.args.evaluate_times):
            soups, episode_reward, _ = self.run_episode_overcooked(evaluate=True)
            # if win_tag:
            #     win_times += 1
            evaluate_reward += episode_reward
            soups_ += soups

        # win_rate = win_times / self.args.evaluate_times
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        soups_ = soups_ / self.args.evaluate_times

        # self.win_rates.append(win_rate)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, soups_, evaluate_reward))
        # self.writer.add_scalar('win_rate_{}'.format(self.env_name), soups_, global_step=self.total_steps)
        # Save the win rates
        # np.save('./data_train/MAPPO_env_{}.npy'.format(self.env_name), np.array(self.evaluate_reward))
        # self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_overcooked(self, evaluate=False):
        episode_reward = 0
        soups = 0
        obs = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            # obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            # s = self.env.get_state()  # s.shape=(state_dim,)

            obs_n = np.array(obs["both_agent_obs"]) # obs_n.shape=(N,obs_dim)
            s = np.concatenate(obs["both_agent_obs"], axis=-1)  # s.shape=(state_dim,)
            # Select random actions from the set {North, South, East, West, Stay, Interact}
            # for each agent.
            # avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            avail_a_n = np.zeros((2,6)) + 1
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, avail_a_n, evaluate=evaluate)  # Get actions and the corresponding log probabilities of N agents
            v_n = self.agent_n.get_value(s, obs_n)  # Get the state values (V(s)) of N agents
            obs_next, r, done, info = self.env.step(a_n)  # Take a step
            shape_rewards = np.array(info['shaped_r_by_agent'])
            # r_total = sum(shape_rewards)
            print(r_total)
            episode_reward += r_total
            soups += r / 20

            if not evaluate:
                if self.args.use_reward_norm:
                    r_total = self.reward_norm(r_total)
                elif args.use_reward_scaling:
                    r_total = self.reward_scaling(r_total)
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r_total, dw)

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            # obs_n = self.env.get_obs()
            # s = self.env.get_state()
            obs_n = np.array(obs["both_agent_obs"])  # obs_n.shape=(N,obs_dim)
            s = np.concatenate(obs["both_agent_obs"], axis=-1)  # s.shape=(state_dim,)
            v_n = self.agent_n.get_value(s, obs_n)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return soups, episode_reward, episode_step + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of MLP")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific", type=float, default=False, help="Whether to use agent specific global state.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    env_names = ['3m', '8m', '2s3z']
    env_index = 0
    runner = Runner_MAPPO_OverCooked(args)
    runner.run()