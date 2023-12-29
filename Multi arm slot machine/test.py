import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from slotenv import SlotEnv

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

if __name__ == '__main__':

    epoch = 100 # 总的epoch数
    steps = 100 # 每个epoch最多执行的步数
    n = 100
    done = False
    ###### 上下文无关奖励测试分析
    reward_type = 0 ## 0表示上下文无关奖励，1表示上下文相关
    data1 = [] # 收集数据绘制分布图
    for e in tqdm(range(epoch)):
        env = SlotEnv(n)
        R = 0
        env.reset() # 初始化，重置状态
        step = 0
        for i in range(steps):
            pi = env.pi_uniform[env.st].squeeze() # 在状态st的策略
            # print((pi.shape))
            act = np.random.choice(a=env.actions, size=1, replace=True, p=pi)  # 根据策略选择动作
            st_,r, done = env.step(act, reward_type)
            R += r
            # 如果到了终止态，则直接跳出本次epoch
            step += 1
            if done:
                break
        R_mean = R / step

        data1.append(R_mean)
    # 绘制一维散点图
    plt.subplot(211)  # 2行1列，第1个子图
    plt.scatter(data1, np.zeros_like(data1), alpha=0.5)
    plt.xlim([-3, 3])  # 设置你想要的范围
    plt.xlabel('Value')
    plt.title('Policy 1')
    # 第二种策略
    data2 = []  # 收集数据绘制分布图
    for e in tqdm(range(epoch)):
        env = SlotEnv(n)
        R = 0
        env.reset()  # 初始化，重置状态
        step = 0
        for i in range(steps):
            pi = env.pi_random[env.st].squeeze()  # 在状态st的策略
            # print((pi.shape))
            act = np.random.choice(a=env.actions, size=1, replace=True, p=pi)  # 根据策略选择动作
            st_, r, done = env.step(act, reward_type)
            R += r
            # 如果到了终止态，则直接跳出本次epoch
            step += 1
            if done:
                break
        R_mean = R / step

        data2.append(R_mean)

    # 绘制一维散点图
    plt.subplot(212)  # 2行1列，第2个子图
    plt.scatter(data2, np.zeros_like(data2), alpha=0.5)
    plt.xlim([-3, 3])  # 设置你想要的范围
    plt.xlabel('Value')
    plt.title('Policy 2')
    plt.tight_layout()  # 自动调整子图参数，防止重叠
    # plt.show()
    plt.savefig("不相关奖励点图.png")
    ###### 上下文无关奖励测试分析 ########################


    ###### 上下文相关奖励测试分析
    reward_type = 1 ## 0表示上下文无关奖励，1表示上下文相关
    data1 = []  # 收集数据绘制分布图
    for e in tqdm(range(epoch)):
        env = SlotEnv(n)
        R = 0
        env.reset()  # 初始化，重置状态
        step = 0
        for i in range(steps):
            pi = env.pi_uniform[env.st].squeeze()  # 在状态st的策略
            # print((pi.shape))
            act = np.random.choice(a=env.actions, size=1, replace=True, p=pi)  # 根据策略选择动作
            st_, r, done = env.step(act, reward_type)
            R += r
            # 如果到了终止态，则直接跳出本次epoch
            step += 1
            if done:
                break
        R_mean = R / step

        data1.append(R_mean)
    # 绘制一维散点图
    plt.subplot(211)  # 2行1列，第1个子图
    plt.scatter(data1, np.zeros_like(data1), alpha=0.5)
    plt.xlim([-3, 3])  # 设置你想要的范围
    plt.xlabel('Value')
    plt.title('Policy 1')
    # 第二种策略
    data2 = []  # 收集数据绘制分布图
    for e in tqdm(range(epoch)):
        env = SlotEnv(n)
        R = 0
        env.reset()  # 初始化，重置状态
        step = 0
        for i in range(steps):
            pi = env.pi_random[env.st].squeeze()  # 在状态st的策略
            # print((pi.shape))
            act = np.random.choice(a=env.actions, size=1, replace=True, p=pi)  # 根据策略选择动作
            st_, r, done = env.step(act, reward_type)
            R += r
            # 如果到了终止态，则直接跳出本次epoch
            step += 1
            if done:
                break
        R_mean = R / step

        data2.append(R_mean)

    # 绘制一维散点图
    plt.subplot(212)  # 2行1列，第2个子图
    plt.scatter(data2, np.zeros_like(data2), alpha=0.5)
    plt.xlim([-3, 3])  # 设置你想要的范围
    plt.xlabel('Value')
    plt.title('Policy 2')
    plt.tight_layout()  # 自动调整子图参数，防止重叠
    # plt.show()
    plt.savefig("相关奖励点图.png")
    ###### 上下文相关奖励测试分析

    """
         通过实验发现，上下文无关奖励的环境下，获取到的奖励与策略的关系不大;
         但上下文相关的奖励环境，与所采用策略相关
    """