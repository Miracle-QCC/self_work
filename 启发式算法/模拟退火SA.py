
import pandas as pd
import math
import random

df = pd.DataFrame()
X = [208,66,98,243,301,17,616,16,289,611,538,572,547,47,470,789,731,680,108,644,519,792,104,29,466,406,521,436,461,87,792,133,161,525,528,214,465,106,778,206,724,537,120,81,116,742,781,534,744,386,362,786,177,530,769,168,363,601,265,721,309,590,630,792,336,755,756,592,510,273,391,649,248,428,604,725,791,164,730,257,26,318,294,413,700,508,689,396,230,542,111,282,557,222,793,212,715,499,712,531]
Y = [11,754,245,583,369,500,663,498,459,695,65,272,247,494,372,6,267,633,343,371,632,106,408,386,0,385,770,706,114,230,786,172,515,750,28,606,81,245,253,676,573,642,45,565,220,114,193,758,722,624,39,332,330,119,365,278,35,465,158,724,798,62,215,568,153,179,344,139,627,96,89,374,128,552,346,429,515,684,502,752,74,367,632,587,328,394,282,745,678,16,475,377,466,703,145,397,159,617,577,247]

cities = [[x,y] for x,y in zip(X,Y)]

def cost(solution):
    # 计算目标函数值，这里假设目标函数是最小化的
    # 实际应用中需要根据具体问题来定义目标函数
    # 这里简化为路径长度之和
    total_distance = 0.0
    for i in range(1, len(solution)):
        total_distance += distance(cities[solution[i-1]], cities[solution[i]])
    total_distance += distance(cities[solution[-1]], cities[solution[0]])  # 回到起点
    return total_distance

def distance(city1, city2):
    # 计算两个城市之间的距离，这里简化为欧式距离
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def simulated_annealing(initial_solution, initial_temperature, cooling_rate, iterations):
    current_solution = initial_solution
    best_solution = current_solution
    temperature = initial_temperature

    for iteration in range(iterations):
        # 生成新解
        new_solution = current_solution.copy()
        pos1, pos2 = random.sample(range(len(new_solution)), 2)
        new_solution[pos1], new_solution[pos2] = new_solution[pos2], new_solution[pos1]

        # 计算新解和当前解的目标函数值
        current_cost = cost(current_solution)
        new_cost = cost(new_solution)

        # 判断是否接受新解
        if new_cost < current_cost or random.uniform(0, 1) < math.exp(-(new_cost - current_cost) / temperature):
            current_solution = new_solution

        # 更新最优解

        if cost(current_solution) < cost(best_solution):
            best_solution = current_solution

        # 降低温度
        temperature *= 1 - cooling_rate
        print(cost(best_solution))
        df.loc[len(df),'best'] = cost(best_solution)
    return best_solution

# 示例：求解旅行商问题（TSP）


# 设置初始解、初始温度、冷却率和迭代次数
initial_solution = list(range(len(cities)))
initial_temperature = 1000.0
cooling_rate = 0.0015
iterations = 100

# 调用模拟退火算法求解TSP问题
result_solution = simulated_annealing(initial_solution, initial_temperature, cooling_rate, iterations)

# 输出结果
print("最优解路径:", result_solution)
print("最优解总距离:", cost(result_solution))
df.to_csv('SA_best.csv')