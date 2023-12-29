import numpy as np
from env2 import Env
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Env()

    ### TODO 任务:求解状态动作值,表示在状态st下选择动作at的价值
    V = np.zeros(6)
    Q = np.zeros((6, 2))
    #根据Vt = r + gamma * Vt+1进行迭代,理论上应该继续迭代，本次只展示一次迭代
    for s in range(6):
        V[s] = env.status_rewards[s] + env.discount_factor * ((V * env.p_a1[s]).sum() * env.policy[s,0] + (V * env.p_a2[s]).sum() * env.policy[s,1])
    #根据Qt = r + gamma * Vt+1进行迭代,理论上应该继续迭代，本次只展示一次迭代
    for s in range(6):
        for a in range(2):
            Q[s,a] = env.status_rewards[s] + env.discount_factor * ((V * env.p_a1[s]).sum() * env.policy[s,0] + (V * env.p_a2[s]).sum() * env.policy[s,1])
    print("所有的状态动作值")
    print(Q)
