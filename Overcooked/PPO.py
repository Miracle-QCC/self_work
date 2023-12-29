import os
from datetime import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import namedtuple

gamma = 0.99
render = False
seed = 1
log_interval = 10

Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self,num_state, num_action, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self,num_state, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 32

    def __init__(self,state_dim,num_act):
        super(PPO, self).__init__()
        self.actor_net = Actor(state_dim,num_act)
        self.critic_net = Critic(state_dim)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')


    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net' + str(time.time())[:10], +'.pkl')
        torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net' + str(time.time())[:10], +'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        #reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        #next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience




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
import numpy as np
layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
# layout = "counter_circuit_o_1order"
def get_event(current_timestamp):
    agent0_event_lst = []
    agent1_event_lst = []
    for key, value in base_env.game_stats.items():
        if key not in ['cumulative_sparse_rewards_by_agent', 'cumulative_shaped_rewards_by_agent']:
            agent0_timestamps = list(value[0])
            agent1_timestamps = list(value[1])
            if current_timestamp in agent0_timestamps:
                agent0_event_lst.append(key)
            if current_timestamp in agent1_timestamps:
                agent1_event_lst.append(key)
    return [agent0_event_lst, agent1_event_lst]


def calculate_rewards(events):
    # Define the rewards/penalties for different actions
    ONION_PICKUP_REWARD = 0
    USEFUL_ONION_PICKUP_REWARD = 2  # 1
    USEFUL_ONION_DROP_REWARD = 2
    VIABLE_ONION_POTTING_REWARD = 3
    OPTIMAL_ONION_POTTING_REWARD = 1
    DISH_PICKUP_REWARD = 1
    USEFUL_DISH_PICKUP_REWARD = 3
    SOUP_COOKING_REWARD = 4  # ???
    SOUP_PICKUP_REWARD = 5
    SOUP_DELIVERY_REWARD = 20

    ONION_DROP_PENALTY = -3
    UNPRODUCTIVE_POTTING_PENALTY = -2
    useful_dishdrop = 4
    CATASTROPHIC_POTTING_PENALTY = -6
    USELESS_ONION_POTTING_PENALTY = -3
    SOUP_DROP_PENALTY = -15
    USELESS_ACTION_PENALTY = -1
    DISH_DROP_PENALTY = -4
    USEFUL_DISH_DROP_PENALTY = -3

    # Initialize rewards for each agent
    rewards = [0, 0]

    # Analyze the actions and assign rewards/penalties
    for agent_id in range(2):
        if len(events[agent_id]) != 0:  # if event happened
            if 'onion_pickup' in events[agent_id]:
                rewards[agent_id] += ONION_PICKUP_REWARD
            if 'useful_onion_pickup' in events[agent_id]:
                rewards[agent_id] += USEFUL_ONION_PICKUP_REWARD
            if 'useful_onion_drop' in events[agent_id]:
                rewards[agent_id] += USEFUL_ONION_DROP_REWARD
            if 'optimal_onion_potting' in events[agent_id]:
                rewards[agent_id] += OPTIMAL_ONION_POTTING_REWARD
            if 'viable_onion_potting' in events[agent_id]:
                rewards[agent_id] += VIABLE_ONION_POTTING_REWARD
            if 'useful_onion_drop' in events[agent_id]:
                rewards[agent_id] += USEFUL_ONION_DROP_REWARD
            if 'soup_cooking' in events[agent_id]:
                rewards[agent_id] += SOUP_COOKING_REWARD
            if 'useful_dish_pickup' in events[agent_id]:
                rewards[agent_id] += USEFUL_DISH_PICKUP_REWARD
            if 'soup_pickup' in events[agent_id]:
                rewards[agent_id] += SOUP_PICKUP_REWARD
            if 'soup_delivery' in events[agent_id]:
                rewards[agent_id] += SOUP_DELIVERY_REWARD

            if 'onion_drop' in events[agent_id] and 'useful_onion_drop' not in events[agent_id]:
                rewards[agent_id] += ONION_DROP_PENALTY
            if 'useless_onion_potting' in events[agent_id]:
                rewards[agent_id] += USELESS_ONION_POTTING_PENALTY
            if 'catastrophic_onion_potting' in events[agent_id]:
                rewards[agent_id] += CATASTROPHIC_POTTING_PENALTY
            if 'dish_drop' in events[agent_id] and 'useful_dish_drop' not in events[agent_id]:
                rewards[agent_id] += DISH_DROP_PENALTY
    return rewards
# Reward shaping is disabled by default.  This data structure may be used for
# reward shaping.  You can, of course, do your own reward shaping in lieu of, or
# in addition to, using this structure.
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

# Build the environment.  Do not modify!
mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
env = gym.make("Overcooked-v0", base_env=base_env,
               featurize_fn=base_env.featurize_state_mdp)

num_episodes = 1000
params = {
    'gamma': 0.8,
    'epsi_high': 0.9,
    'epsi_low': 0.05,
    'decay': 200,
    'lr': 0.001,
    'capacity': 100000,
    'batch_size': 256,
    'state_space_dim': 96*2,
    'action_space_dim': 6*6,
}
actions_list = []
for i in range(6):
    for j in range(6):
        actions_list.append((i,j))

agent0 = PPO(96,6)
agent1 = PPO(96,6)

for e in range(num_episodes):
    # Episode termination flag
    done = False

    # The number of soups the agent pair made during the episode
    num_soups_made = 0

    # Reset the environment at the start of each episode
    obs = env.reset()
    current_ts = -1
    e_rewards = 0
    while not done:
        current_ts = current_ts + 1

        # Obtain observations for each agent
        obs0 = obs["both_agent_obs"][0]
        obs1 = obs["both_agent_obs"][1]

        # Select random actions from the set {North, South, East, West, Stay, Interact}
        # for each agent.
        a0, logp_a0 = agent0.select_action(obs0)
        a1, logp_a1 = agent1.select_action(obs1)
        # Take the selected actions and receive feedback from the environment
        # The returned reward "R" only reflects completed soups.  You can find
        # the separate shaping rewards in the "info" variables
        # info["shaped_r_by_agent"][0] and info["shaped_r_by_agent"][1].  Note that
        # this shaping reward does *not* include the +20 reward for completed
        # soups returned in "R".
        obs_next, R, done, info = env.step([a0, a1])
        events = get_event(current_ts)

        # calculate shape rewards for both agents
        if (len(events[0]) != 0) or (len(events[1]) != 0):
            # print('current_ts: {}, events: {}'.format(current_ts, events))
            shape_rewards = calculate_rewards(events)
        else:
            shape_rewards = [0, 0]
        # onion_drop_info = info.get('POT_DISTANCE_REW')
        # if onion_drop_info:
        #     print("dsadasdasdasdasd")
        obs0_next = obs_next["both_agent_obs"][0]
        obs1_next = obs_next["both_agent_obs"][1]
        # Accumulate the number of soups made
        num_soups_made += int(R / 20) # Each served soup generates 20 reward
        e_rewards += sum(shape_rewards)
        # state, action, reward, next_state, done
        # agent.put(obs_total, a_total, R, obs_total_next, done)
        trans0 = Transition(obs0, a0, logp_a0, shape_rewards[0] / 10, obs0_next)
        trans1 = Transition(obs1, a1, logp_a1, shape_rewards[1] / 10, obs0_next)

        agent0.store_transition(trans0)
        agent1.store_transition(trans1)

        if done:
            agent0.update(e)
            agent1.update(e)

    # Display status
    print("Ep {0}".format(e + 1), end=" ")
    print("number of soups made: {0}".format(num_soups_made), end="  rewards:")
    print(e_rewards)

# The info flag returned by the environemnt contains special status info
# specifically when done == True.  This information may be useful in
# developing, debugging, and analyzing your results.  It may also be a good
# way for you to find a metric that you can use in evaluating collaboration
# between your agents.
print("\nExample info dump:\n\n", info)