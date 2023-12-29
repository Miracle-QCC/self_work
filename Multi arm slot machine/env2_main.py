import numpy as np
from env2 import Env
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = Env()

    ### TODO 任务:求解状态动作值,表示在状态st下选择动作at的价值
    state_act_val = np.zeros((6, 2))
    for s in range(6):
        for a in range(2):
            # 等于当前状态下，选择动作a1后，所有转移状态的奖励与概率的乘积和
            state_act_val[s, a] = (env.status_rewards * env.p_a1[s]).sum()

    print("所有的状态动作值")
    print(state_act_val)

    ### TODO 任务：修改策略，绘图展示
    # 调整状态s下选择动作a的概率
    s = 0
    a = 0
    rs_s0 = []  # 表示状态价值

    # 状态价值 = 该状态的状态动作值 * 对应的动作概率之和
    # 假设第一个状态选择动作a的概率从0~1,间隔为0.1
    probs_a0 = np.arange(0, 1.1, 0.1)  # 选第一个动作的概率
    for p in probs_a0:
        r = p * state_act_val[s, 0] + (1 - p) * state_act_val[s, 1]
        rs_s0.append(r)

    # 绘制散点图
    plt.scatter(probs_a0, rs_s0)

    # 添加标签和标题
    plt.xlabel('概率')
    plt.ylabel('状态价值')
    plt.title('状态价值概率变化图')
    # # 显示图形
    # plt.show()
    plt.savefig("状态价值概率变化图.jpg")

    ### TODO 任务.策略迭代
    X = range(10)
    R_list = []
    done = False
    st = env.reset()
    iters = 0
    for e in range(10):
        R = 0
        st = env.reset()
        for i in range(100):
            pi = env.policy[st].squeeze()
            if pi.sum() != 1:
                continue
            act = np.random.choice(a=env.action_space, size=1, replace=True, p=pi)  # 根据策略选择动作
            st_, r, done = env.step(act)

            st = st_
            R += r
            iters += 1
            if done:
                break
        R_list.append(R)
        #  进行策略迭代
        env.policy_iteration(iter_num=20)
    plt.plot(X, R_list, label='policy iter')
    # 添加标签和标题
    plt.xlabel('iter num/20')
    plt.ylabel('the Accumulated rewards')
    # # 显示图形
    # plt.show()
    # plt.savefig("1.jpg")
    # print(R_list)

    ### TODO 任务.价值迭代
    R_list = []
    done = False
    st = env.reset()
    iters = 0
    for e in range(10):
        R = 0
        st = env.reset()
        for i in range(100):
            pi = env.policy[st].squeeze()
            if pi.sum() != 1:
                continue
            act = np.random.choice(a=env.action_space, size=1, replace=True, p=pi)  # 根据策略选择动作
            st_, r, done = env.step(act)

            st = st_
            R += r
            iters += 1
            if done:
                break
        R_list.append(R)
        #  进行策略迭代
        env.improve_policy(iter_num=20)

    plt.plot(X, R_list, label='value iter')
    # 添加标签和标题
    plt.xlabel('iter num/20')
    plt.ylabel('the Accumulated rewards')
    # # 显示图形
    plt.show()
    plt.savefig("累计奖励变化图.jpg")