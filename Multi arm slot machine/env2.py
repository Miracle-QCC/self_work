import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from slotenv import softmax
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Env:
    def __init__(self):
        ## 每个状态有两个动作
        self.action_space = [0,1]
        ## 总共6个状态
        self.state_space = np.arange(6)

        ## 定义状态转移率
        ###  动作a1的
        self.p_a1 = np.zeros((6,6))
        self.p_a1[0,1] = 0.3
        self.p_a1[0,2] = 0.7

        self.p_a1[1,1] = 0.3
        self.p_a1[1,2] = 0.7

        self.p_a1[2,0] = 0.4
        self.p_a1[2,3] = 0.6

        self.p_a1[3,4] = 0.6
        self.p_a1[3,5] = 0.4

        ###  定义动作a2的
        self.p_a2 = np.zeros((6,6))
        self.p_a2[0,0] = 0.3
        self.p_a2[0,3] = 0.7

        self.p_a2[1,0] = 0.3
        self.p_a2[1,3] = 0.7

        self.p_a2[2,1] = 0.4
        self.p_a2[2,2] = 0.6

        ###定义状态奖励
        self.status_rewards = np.zeros(6) + 1
        # self.status_rewards[0] = 1
        # self.status_rewards[1] = 1
        # self.status_rewards[2] = 1
        # self.status_rewards[3] = 1
        self.status_rewards[4] = 0
        self.status_rewards[5] = 0

        ### 保存环境当前状态，初始为0
        self.st = 0

        ### 定义策略，表示在状态s下选择各个动作的概率,初始为均匀分布
        self.policy = np.zeros((len(self.status_rewards), len(self.action_space)))
        self.policy[:,1] += 1
        self.policy = softmax(self.policy)

        ### 还要考虑状态5和6终止了，不能再运行
        self.policy[4] = np.array([0,0])
        self.policy[5] = np.array([0,0])

        self.iter_totals = 0

        self.discount_factor = 0.9 #折扣因子是0.9
        self.value_function = np.zeros(len(self.state_space))
    def step(self, act):
        st = self.st
        if st in [4,5]:
            return st, 0, True

        if act == 0: # 选择动作a1
            prob = self.p_a1[st].squeeze() # 状态转移率
            if prob.sum() != 1:
                return st,0,True
            st_ = np.random.choice(a=self.state_space, size=1, replace=True, p=prob) # 根据状态转移率跳转下一状态
            r = self.status_rewards[st]
        else:
            prob = self.p_a2[st].squeeze()   # 状态转移率
            if prob.sum() != 1:
                return st,0,True
            st_ = np.random.choice(a=self.state_space, size=1, replace=True, p=prob)  # 根据状态转移率跳转下一状态
            r = self.status_rewards[st]
        # 状态转换
        self.st = st_
        # 返回下一状态和奖励


        return st_, r, False

    # 初始化环境
    def reset(self):
        self.st = 0
        self.iter_totals = 0
        return self.st

    def policy_evaluation(self, policy, theta=1e-2):
        # 策略评估
        # V = np.zeros(len(self.state_space))
        # 为了逐步体现效果，每次只迭代一次
        while True:
            delta = 0
            for s in self.state_space:
                v = self.value_function[s]
                # 在状态s分别选择两个动作，同时考虑未来收益，将下一状态的收益也纳入
                self.value_function[s] = sum(policy[s, a] * sum(
                    self.p_a1[s, s_prime] * (self.status_rewards[s] + self.discount_factor * self.value_function[s_prime])
                    for s_prime in self.state_space) for a in self.action_space)
                delta = max(delta, abs(v - self.value_function[s]))
            self.iter_totals += 1
            if delta < theta:
                break

        return self.value_function

    def policy_improvement(self, V):
        # 策略改进
        policy_stable = True
        for s in self.state_space:
            old_action = np.argmax(self.policy[s])

            action_values = [sum(self.p_a1[s, s_prime] * (self.status_rewards[s] + self.discount_factor * V[s_prime])
                                 for s_prime in self.state_space) for a in self.action_space]

            best_action = np.argmax(action_values)

            if old_action != best_action:
                policy_stable = False

            self.policy[s] = np.eye(len(self.action_space))[best_action]

        return policy_stable

    def policy_iteration(self, discount_factor=1.0, iter_num=20):
        for i in range(iter_num):
            V = self.policy_evaluation(self.policy)  # 获取稳定的状态价值函数
            policy_stable = self.policy_improvement(V) # 进行策略优化
            self.iter_totals += 1

            if self.iter_totals % 20 == 0:
                break
        return self.policy

    ## 价值迭代
    def value_iteration(self, V, theta=1e-6):
        while True:
            delta = 0
            for s in self.env.state_space:
                v = self.value_function[s]
                action_values = [sum(self.p_a1[s, s_prime] * (
                            self.env.status_rewards[s] + self.discount_factor * self.value_function[s_prime]) for
                                     s_prime in self.state_space) for a in self.action_space]
                best_action_value = max(action_values)
                self.value_function[s] = best_action_value
                delta = max(delta, abs(v - self.value_function[s]))

            if delta < theta:
                break

    def improve_policy(self, iter_num=20):
        policy = np.zeros((len(self.state_space), len(self.action_space)))

        for s in self.state_space:
            action_values = [sum(self.p_a1[s_prime, s] * (
                        self.status_rewards[s] + self.discount_factor * self.value_function[s_prime]) for s_prime in
                                 self.state_space) for a in self.action_space]
            # 将最优动作的概率置为1
            best_action = np.argmax(action_values)
            policy[s][best_action] = 1.0
            self.iter_totals += 1

            if self.iter_totals % 20 == 0:
                break
        return policy


