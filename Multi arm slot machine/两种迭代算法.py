import numpy as np
from env2 import Env
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = Env()

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
    # plt.show()
    plt.savefig("累计奖励变化图.jpg")