from Experiment_DQN_PyTorch_23z.network_env import Network_Env
from Experiment_DQN_PyTorch_23z.DQN import PrioritizedReplayD3QN
from Experiment_DQN_PyTorch_23z.network_env import sfc_sum
import numpy as np
import matplotlib.pyplot as plt
import random


def run_maze():
	step = 0
	ave_td_show=[]

	ave_dif1_show=[]
	ave_dif2_show=[]
	reward_return=[]
	sum_reward=0
	random_list = []
	sfc_count_episode = []
	sum_sfc_count = 0

	for episode in range(500):
		print("................第", episode, "轮开始................")
		for i in range(14):
			ran = random.random()
			random_list.append(ran)
		observation = random_list
		observation=np.array(observation)
		observation=np.hstack((observation.reshape(1, 14).squeeze()))
		env.reset_env()
		e_step=0
		while True:
			if env.flag_SFC:
				env.reset_net()
			action = RL.choose_action(observation,e_step,env.sfc_placement_row,env.node_path_remove)
			observation_, reward, done = env.step(action)
			# print(reward)
			sum_reward+=reward
			RL.store_transition(observation, action, reward, observation_)
			if (step>200) and (step%5==0):
				RL.learn()
			observation = observation_
			e_step+=1
			if done:
				break
			step += 1
		print("................第",episode,"轮结束................")
		# print("total_link_td:",env.total_link_td)
		# print("ave_td:",env.ave_td)
		# print("sum_dif1:",env.sum_dif1)
		# print("ave_dif1",env.ave_dif1)
		# print("sum_dif2:",env.sum_dif2)
		# print("ave_dif2",env.ave_dif2)
		# print("sum_reward:",sum_reward)
		ave_td_show.append(env.ave_td)
		ave_dif1_show.append(env.ave_dif1)
		ave_dif2_show.append(env.ave_dif2)
		reward_return.append(sum_reward)
		sum_reward=0
		random_list.clear()

		sfc_count_episode.append(env.sfc_count)
		# print("env.sfc_count",env.sfc_count)
	print('game over')

	env.sum_ave_td /= 490
	env.sum_ave_dif1 /= 490
	env.sum_ave_dif2 /= 490
	print(env.sum_ave_td)
	print(env.sum_ave_dif1)
	print(env.sum_ave_dif2)

	'''绘图'''
	# plt.plot(np.arange(len(ave_td_show)), ave_td_show)
	# plt.ylabel('ave_td')
	# plt.xlabel('training steps')
	# plt.show()
	# plt.plot(np.arange(len(ave_dif1_show)), ave_dif1_show)
	# plt.ylabel('ave_dif1')
	# plt.xlabel('training steps')
	# plt.show()
	# plt.plot(np.arange(len(ave_dif2_show)), ave_dif2_show)
	# plt.ylabel('ave_dif2')
	# plt.xlabel('training steps')
	# plt.show()
	# plt.plot(np.arange(len(reward_return)), reward_return)
	# plt.ylabel('reward_return')
	# plt.xlabel('training steps')
	# plt.show()

	for i in range(0,len(sfc_count_episode)):
		sum_sfc_count+=sfc_count_episode[i]
	sum_sfc_count/=500
	print(sum_sfc_count)

if __name__ == '__main__':
	env = Network_Env()
	RL = PrioritizedReplayD3QN(env.n_actions, 14,	### 199需要修改
					learning_rate=0.01,
					reward_decay=0.9,
					e_greedy=0.9,
					replace_target_iter=100,
					memory_size=2000
					)
	run_maze()
	# RL.plot_cost()