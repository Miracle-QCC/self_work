import random

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
from RND import RND
import numpy as np
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
device = 'cuda' if torch.cuda.is_available() else "cpu"
from QMIXDDQN__ import Qmix_DDQN_Agent


# layout = "cramped_room"
# layout = "asymmetric_advantages"
# layout = "coordination_ring"
# layout = "forced_coordination"
layout = "counter_circuit_o_1order"
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

def judge_str_in(events, s):
    for e in events:
        if s in e:
            return True
    return False

class detail_feature:
    def __init__(self,obs):

        self.p_i_orientation = obs[:4] ### orientation
        self.p_i_obj = obs[4:8] ## onion, soup, dish, tomato

        self.p_i_closest_onion = obs[8:10]
        # self.p_i_closest_tomato = obs[10:12]
        self.p_i_closest_dish = obs[12:14]
        self.p_i_closest_soup = obs[14:16]

        self.p_i_closest_soup_n_ingredients = obs[16:18]
        self.p_i_closest_serving_area = obs[18:20]
        self.p_i_closest_empty_counter = obs[20:22]

        ## pot
        self.p_i_closest_potj_exists = obs[22]
        self.p_i_closest_potj_status = obs[23:27] # is empty| is full| is cooking| is ready
        self.p_i_closest_potj_num = obs[27:29] # union_num | tomato_num
        self.p_i_closest_potj_cook_time = obs[29] # Remaining cooking time
        self.p_i_closest_potj = obs[30:32] # The relative position of the nearest pot

        # p_i_closest_potj_exists = obs[32]
        # p_i_closest_potj_status = obs[33:37]  # is empty| is full| is cooking| is ready
        # p_i_closest_potj_num = obs[37:39]  # union_num | tomato_num
        # p_i_closest_potj_cook_time = obs[39]  # Remaining cooking time
        # p_i_closest_potj = obs[40:42]  # The relative position of the nearest pot

        self.p_i_wall = obs[42:]  # boolean value of whether player i has a wall immediately in direction j



def calculate_rewards(events, obs, obs_pre):

    # Initialize rewards for each agent
    rewards = [0, 0]
    stay_time_union = 0
    stay_time_pot = 0
    for agent_id in range(2):
        df = detail_feature(obs[agent_id][:46])
        df_pre = detail_feature(obs_pre[agent_id][:46])
        loc = obs[agent_id][-2:]
        pre_loc = obs_pre[agent_id][-2:]
        if events[agent_id]:
            ## take union
            if df.p_i_closest_potj_num[0] < 3 and judge_str_in(events[agent_id], 'useful_onion_pickup'):
                rewards[agent_id] += 2
            ## put union
            elif df.p_i_closest_potj_num[0] < 3 and judge_str_in(events[agent_id], 'viable_onion_potting'):
                rewards[agent_id] += 5
            # take dish
            elif df.p_i_closest_potj_num[0] == 3 and judge_str_in(events[agent_id], 'dish_pickup'):
                rewards[agent_id] += 10
            # take soup
            elif df.p_i_closest_potj_status[3] and judge_str_in(events[agent_id], 'soup_pickup'):
                rewards[agent_id] += 10
            #  soup delivery
            elif judge_str_in(events[agent_id], 'soup_delivery'):
                rewards[agent_id] += 20
            # others
            else:
                rewards[agent_id] -= 2
        else:
            if pre_loc[0] == loc[0] and pre_loc[1] == loc[1]:
                rewards[agent_id] -= 1
            # with soup
            elif df.p_i_obj[1]:
                soup_dis_r = np.sqrt(df_pre.p_i_closest_serving_area[0] ** 2 + df_pre.p_i_closest_serving_area[1] ** 2) - \
                           np.sqrt(df.p_i_closest_serving_area[0] ** 2 + df.p_i_closest_serving_area[1] ** 2)
                rewards[agent_id] += soup_dis_r
            elif df.p_i_closest_potj_num[0] < 3:
                # hold union
                if df.p_i_obj[0]:
                    pot_dis_r = np.sqrt(df_pre.p_i_closest_potj[0] ** 2 + df_pre.p_i_closest_potj[1] ** 2) - \
                                np.sqrt(df.p_i_closest_potj[0] ** 2 + df.p_i_closest_potj[1] ** 2)
                    rewards[agent_id] += pot_dis_r
                # without
                else:
                    onion_dis_r = np.sqrt(df_pre.p_i_closest_onion[0] ** 2 + df_pre.p_i_closest_onion[1] ** 2) - \
                                  np.sqrt(df.p_i_closest_onion[0] ** 2 + df.p_i_closest_onion[1] ** 2)
                    if onion_dis_r == 0:
                        rewards[agent_id] -= 1
                    rewards[agent_id] += onion_dis_r
                    # ### stay
                    # if np.sqrt(df.p_i_closest_onion[0] ** 2 + df.p_i_closest_onion[1] ** 2) == np.sqrt(df_pre.p_i_closest_onion[0] ** 2 + df_pre.p_i_closest_onion[1] ** 2):
                    #     stay_time_union += 1
                    #     rewards[agent_id] -= 1

            elif df.p_i_closest_potj_num[0] == 3:
                # with dish
                if df.p_i_obj[2]:
                    dish_2_pot_dis_r = np.sqrt(df_pre.p_i_closest_potj[0] ** 2 + df_pre.p_i_closest_potj[1] ** 2) - \
                                       np.sqrt(df.p_i_closest_potj[0] ** 2 + df.p_i_closest_potj[1] ** 2)
                    rewards[agent_id] += dish_2_pot_dis_r
                else:
                    dish_dis_r = np.sqrt(df_pre.p_i_closest_dish[0] ** 2 + df_pre.p_i_closest_dish[1] ** 2) - \
                                 np.sqrt(df.p_i_closest_dish[0] ** 2 + df.p_i_closest_dish[1] ** 2)

                    if dish_dis_r == 0:
                        rewards[agent_id] -= 1
                    else:
                        rewards[agent_id] += dish_dis_r

                    ## stay
                    if np.sqrt(df.p_i_closest_potj[0] ** 2 + df.p_i_closest_potj[1] ** 2) == np.sqrt(
                            df_pre.p_i_closest_potj[0] ** 2 + df_pre.p_i_closest_potj[1] ** 2):
                        # stay_time_pot += 1
                        rewards[agent_id] -= 1

            # elif
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
num_episodes = 2500
# Build the environment.  Do not modify!
mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=reward_shaping)
base_env = OvercookedEnv.from_mdp(mdp, horizon=horizon, info_level=0)
env = gym.make("Overcooked-v0", base_env=base_env,
               featurize_fn=base_env.featurize_state_mdp)

params = {
    'gamma': 0.9,
    'epsi_high': 0.9,
    'epsi_low': 0.01,
    'decay': 100000,
    'lr': 0.0001,
    'capacity': 1000000,
    'batch_size': 256,
    'state_space_dim': 96,
    'action_space_dim': 6,
}

rnd_total = RND(96*2).to(device)
rnd_buffer = []
qmix_agent = Qmix_DDQN_Agent(**params)
e = 0
while e < num_episodes:
    # Episode termination flag
    done = False

    # The number of soups the agent pair made during the episode
    num_soups_made = 0

    # Reset the environment at the start of each episode
    obs = env.reset()
    current_ts = -1
    e_rewards = 0
    skip = False
    while not done:
        current_ts = current_ts + 1

        # Obtain observations for each agent
        obs0 = obs["both_agent_obs"][0]
        obs1 = obs["both_agent_obs"][1]

        # Select random actions from the set {North, South, East, West, Stay, Interact}
        # for each agent.
        a0, a1 = qmix_agent.act(obs0, obs1)

        obs_next, R, done, info = env.step([a0, a1])
        if info['policy_agent_idx'] == 1:
            skip = True
            break
        events = get_event(current_ts)
        obs0_next = obs_next["both_agent_obs"][0]
        obs1_next = obs_next["both_agent_obs"][1]
        # calculate shape rewards for both agents
        # if (len(events[0]) != 0) or (len(events[1]) != 0):
        #     # print('current_ts: {}, events: {}'.format(current_ts, events))
        #     shape_rewards = calculate_rewards(events)
        # else:
        #     shape_rewards = [0, 0]
        # onion_drop_info = info.get('POT_DISTANCE_REW')
        # if onion_drop_info:
        #     print("dsadasdasdasdasd")
        # shape_rewards = np.array(calculate_rewards(events, [obs0_next, obs1_next], [obs0, obs1])) + np.array(info['shaped_r_by_agent'])
        shape_rewards = np.array(info['shaped_r_by_agent'])
        if info['policy_agent_idx'] == 1:
            shape_rewards = shape_rewards[::-1]
        # Accumulate the number of soups made
        num_soups_made += int(R / 20)  # Each served soup generates 20 reward
        # shape_rewards = info['sparse_r_by_agent']
        e_rewards += sum(shape_rewards) + R
        # state, action, reward, next_state, done
        # agent.put(obs_total, a_total, R, obs_total_next, done)
        # r_curious0 = rnd0.value(torch.from_numpy(obs0_next).float().to(device))
        # r_curious1 = rnd1.value(torch.from_numpy(obs1_next).float().to(device))
        qmix_agent.put(obs0, obs1, a0, a1, shape_rewards[0] / 100, shape_rewards[1] / 100, obs0_next, obs1_next, done)
        if len(rnd_buffer) < 1e6:
            rnd_buffer.append(np.concatenate([obs0, obs1], axis=-1))
        else:
            rnd_buffer.pop()
            rnd_buffer.append(np.concatenate(obs0, obs1))

        qmix_agent.update(rnd_total)
        rnd_total.update(states=rnd_buffer)
        obs = obs_next
    # Display status
    if not skip:
        print("Ep {0} / {1}".format(e + 1, num_episodes), end=" ")
        print("number of soups made: {0}".format(num_soups_made), end="  rewards:")
        print(e_rewards)
        e += 1

# The info flag returned by the environemnt contains special status info
# specifically when done == True.  This information may be useful in
# developing, debugging, and analyzing your results.  It may also be a good
# way for you to find a metric that you can use in evaluating collaboration
# between your agents.
print("\nExample info dump:\n\n", info)

acts = ['up','down','left','right','stay','interact']
class StudentPolicy(NNPolicy):
    """ Generate policy """
    def __init__(self, agent):
        super(StudentPolicy, self).__init__()
        self.agent = agent

    def state_policy(self, state, agent_index):
        """
        This method should be used to generate the poiicy vector corresponding to
        the state and agent_index provided as input.  If you're using a neural
        network-based solution, the specifics depend on the algorithm you are using.
        Below are two commented examples, the first for a policy gradient algorithm
        and the second for a value-based algorithm.  In policy gradient algorithms,
        the neural networks output a policy directly.  In value-based algorithms,
        the policy must be derived from the Q value outputs of the networks.  The
        uncommented code below is a placeholder that generates a random policy.
        """
        featurized_state = base_env.featurize_state_mdp(state)
        input_state = torch.FloatTensor(featurized_state[agent_index]).unsqueeze(0)
        act = self.agent.act(input_state)

        # Random deterministic policy
        action_probs = np.zeros(env.action_space.n)
        action_probs[act] = 1
        print("agent idx:",agent_index, f"  act: {acts[act]}" )
        return action_probs

    def multi_state_policy(self, states, agent_indices):
        """ Generate a policy for a list of states and agent indices """
        return [self.state_policy(state, agent_index) for state, agent_index in zip(states, agent_indices)]


class StudentAgent(AgentFromPolicy):
    """Create an agent using the policy created by the class above"""
    def __init__(self, policy):
        super(StudentAgent, self).__init__(policy)



torch.save(qmix_agent.q1.state_dict(),'q1_5_o_with_1.pt')
torch.save(qmix_agent.q2.state_dict(),'q2_5_o_with_1.pt')
torch.save(qmix_agent.qmix.state_dict(),'qmix_5_o_with_1.pt')


# Instantiate the policies for both agents
# policy0 = StudentPolicy(agent0)
# policy1 = StudentPolicy(agent1)
#
# # Instantiate both agents
# agent0 = StudentAgent(policy0)
# agent1 = StudentAgent(policy1)
# agent_pair = AgentPair(agent0, agent1)
# torch.save(agent0.state_dict(), "agent0.pt")
# torch.save(agent1.state_dict(), "agent1.pt")

# Generate an episode
ae = AgentEvaluator.from_layout_name({"layout_name": layout}, {"horizon": horizon})
trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)
print("\nlen(trajs):", len(trajs))


# Modify as appropriate
img_dir = "images/"
ipython_display = True
gif_path = "test4.gif"

# Do not modify -- uncomment for GIF generation
StateVisualizer().display_rendered_trajectory(trajs, img_directory_path=img_dir, ipython_display=ipython_display)

img_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
img_list.sort(key=lambda x: os.path.getmtime(os.path.join(img_dir, x)))
images = [Image.open(img_dir + img).convert('RGBA') for img in img_list]
images[0].save(gif_path, save_all=True, append_images=images[1:], optimize=False, duration=250, loop=0)
with open(gif_path,'rb') as f: display(IPImage(data=f.read(), format='png'))