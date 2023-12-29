import numpy as np
from env2 import Env
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Env()

    V = np.zeros(6)
    Q = np.zeros((6, 2))
    # 根据Vt = r + gamma * Vt+1进行迭代,理论上应该继续迭代，本次只展示一次迭代
    for s in range(6):
        V[s] = env.status_rewards[s] + env.discount_factor * (
                    (V * env.p_a1[s]).sum() * env.policy[s, 0] + (V * env.p_a2[s]).sum() * env.policy[s, 1])
    # 根据Qt = r + gamma * Vt+1进行迭代,理论上应该继续迭代，本次只展示一次迭代
    for s in range(6):
        for a in range(2):
            Q[s, a] = env.status_rewards[s] + env.discount_factor * (
                        (V * env.p_a1[s]).sum() * env.policy[s, 0] + (V * env.p_a2[s]).sum() * env.policy[s, 1])
    print("所有的状态动作值")
    print(Q)

    ### TODO 任务：修改策略，绘图展示
    # 调整状态s下选择动作a的概率
    s = 0
    a = 0
    rs_s0 = []  # 表示状态价值

    # 状态价值 = 该状态的状态动作值 * 对应的动作概率之和
    # 假设第一个状态选择动作a的概率从0~1,间隔为0.1
    probs_a0 = np.arange(0, 1.1, 0.1)  # 选第一个动作的概率
    for p in probs_a0:
            V[0] = env.status_rewards[s] + env.discount_factor * (
                    (V * env.p_a1[s]).sum() * p  + (V * env.p_a2[s]).sum() * (1 - p))
            rs_s0.append(V[0])

    # 绘制散点图
    plt.scatter(probs_a0, rs_s0)
    print(rs_s0)
    # 添加标签和标题
    plt.xlabel('prob')
    plt.ylabel('value')
    plt.title('Probability change diagram of state value')
    # # 显示图形
    # plt.show()
    plt.savefig("状态价值概率变化图.jpg")