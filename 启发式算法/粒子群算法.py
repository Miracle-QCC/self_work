import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

city_num = 100  # 城市数量
np.random.seed(42)  # 固定随机数种子


df = pd.DataFrame()
X = [208,66,98,243,301,17,616,16,289,611,538,572,547,47,470,789,731,680,108,644,519,792,104,29,466,406,521,436,461,87,792,133,161,525,528,214,465,106,778,206,724,537,120,81,116,742,781,534,744,386,362,786,177,530,769,168,363,601,265,721,309,590,630,792,336,755,756,592,510,273,391,649,248,428,604,725,791,164,730,257,26,318,294,413,700,508,689,396,230,542,111,282,557,222,793,212,715,499,712,531]
Y = [11,754,245,583,369,500,663,498,459,695,65,272,247,494,372,6,267,633,343,371,632,106,408,386,0,385,770,706,114,230,786,172,515,750,28,606,81,245,253,676,573,642,45,565,220,114,193,758,722,624,39,332,330,119,365,278,35,465,158,724,798,62,215,568,153,179,344,139,627,96,89,374,128,552,346,429,515,684,502,752,74,367,632,587,328,394,282,745,678,16,475,377,466,703,145,397,159,617,577,247]


def clac_distance(X, Y):
    """
    计算两个城市之间的欧式距离，二范数
    :param X: 城市X的坐标.np.array数组
    :param Y: 城市Y的坐标.np.array数组
    :return:
    """
    distance_matrix = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(city_num):
            if i == j:
                continue

            distance = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
            distance_matrix[i][j] = distance

    return distance_matrix

def fitness_func(distance_matrix, x_i):
    """
    适应度函数
    :param distance_matrix: 城市距离矩阵
    :param x_i: PSO的一个解（路径序列）
    :return:
    """
    total_distance = 0
    for i in range(1, city_num):
        start_city = x_i[i - 1]
        end_city = x_i[i]
        total_distance += distance_matrix[start_city][end_city]
    total_distance += distance_matrix[x_i[-1]][x_i[0]]  # 从最后的城市返回出发的城市

    return total_distance

def get_ss(x_best, x_i, r):
    """
    计算交换序列，即x2结果交换序列ss得到x1，对应PSO速度更新公式中的 r1(pbest-xi) 和 r2(gbest-xi)
    :param x_best: pbest or gbest
    :param x_i: 粒子当前的解
    :param r: 随机因子
    :return:
    """
    velocity_ss = []
    for i in range(len(x_i)):
        if x_i[i] != x_best[i]:
            j = np.where(x_i == x_best[i])[0][0]
            so = (i, j, r)  # 得到交换子
            velocity_ss.append(so)
            x_i[i], x_i[j] = x_i[j], x_i[i]  # 执行交换操作

    return velocity_ss


def do_ss(x_i, ss):
    """
    执行交换操作
    :param x_i:
    :param ss: 由交换子组成的交换序列
    :return:
    """
    for i, j, r in ss:
        rand = np.random.random()
        if rand <= r:
            x_i[i], x_i[j] = x_i[j], x_i[i]
    return x_i


def main(X, Y):
    # 参数设置
    size = 50
    # r1 = np.random.rand()
    r1 = 0.7
    # r2 = np.random.rand()
    r2 = 0.8
    iter_max_num = 100
    fitness_value_lst = []

    distance_matrix = clac_distance(X, Y)

    # 初始化种群各个粒子的位置，作为个体的历史最优pbest
    # 每行都是1-10的不重复随机数，表示城市的访问顺序
    pbest_init = np.zeros((size, city_num), dtype=np.int64)
    for i in range(size):
        pbest_init[i] = np.random.choice(list(range(city_num)), size=city_num, replace=False)

    # 计算每个粒子对应的适应度
    pbest = pbest_init
    pbest_fitness = np.zeros((size, 1))
    for i in range(size):
        pbest_fitness[i] = fitness_func(distance_matrix, x_i=pbest_init[i])

    # 计算全局适应度和对应的gbest
    gbest = pbest_init[pbest_fitness.argmin()]
    gbest_fitness = pbest_fitness.min()

    # 记录算法迭代效果
    fitness_value_lst.append(gbest_fitness)

    # 迭代过程
    for i in range(iter_max_num):
        # 控制迭代次数
        for j in range(size):
            # 遍历每个粒子
            pbest_i = pbest[j].copy()  # 此处要用.copy 否则会出现浅拷贝问题
            x_i = pbest_init[j].copy()

            # 计算交换序列，即 v = r1(pbest-xi) + r2(gbest-xi)
            ss1 = get_ss(pbest_i, x_i, r1)
            ss2 = get_ss(gbest, x_i, r2)
            ss = ss1 + ss2
            # print(f'{ss1} + {ss2} = {ss}')
            # 执行交换操作，即 x = x + v
            x_i = do_ss(x_i, ss)

            fitness_new = fitness_func(distance_matrix, x_i)
            fitness_old = pbest_fitness[j]
            if fitness_new < fitness_old:
                pbest_fitness[j] = fitness_new
                pbest[j] = x_i

            gbest_fitness_new = pbest_fitness.min()
            gbest_new = pbest[pbest_fitness.argmin()]
            if gbest_fitness_new < gbest_fitness:
                gbest_fitness = gbest_fitness_new
                gbest = gbest_new

            fitness_value_lst.append(gbest_fitness)
        df.loc[len(df), 'best'] = gbest_fitness
        print(i, gbest_fitness)
    # 输出迭代结果
    print(f'迭代最优路劲为：{gbest}')
    print(f'迭代最优值为：{gbest_fitness}')
    # plot_tsp(gbest)
    # plt.title(f'TSP路径规划结果-{time.localtime()}-Sum3')
    # plt.show()
if __name__ == '__main__':
    main(X,Y)
    df.to_csv('POS_best.csv')