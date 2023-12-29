import random
import pandas as pd
# 定义城市坐标
df = pd.DataFrame()

X = [208,66,98,243,301,17,616,16,289,611,538,572,547,47,470,789,731,680,108,644,519,792,104,29,466,406,521,436,461,87,792,133,161,525,528,214,465,106,778,206,724,537,120,81,116,742,781,534,744,386,362,786,177,530,769,168,363,601,265,721,309,590,630,792,336,755,756,592,510,273,391,649,248,428,604,725,791,164,730,257,26,318,294,413,700,508,689,396,230,542,111,282,557,222,793,212,715,499,712,531]
Y = [11,754,245,583,369,500,663,498,459,695,65,272,247,494,372,6,267,633,343,371,632,106,408,386,0,385,770,706,114,230,786,172,515,750,28,606,81,245,253,676,573,642,45,565,220,114,193,758,722,624,39,332,330,119,365,278,35,465,158,724,798,62,215,568,153,179,344,139,627,96,89,374,128,552,346,429,515,684,502,752,74,367,632,587,328,394,282,745,678,16,475,377,466,703,145,397,159,617,577,247]
city_coords = {}
for i in range(100):
    city_coords[i+1] = (X[i],Y[i])


# 遗传算法参数设置
population_size = 100  # 种群大小
mutation_rate = 0.1  # 变异率
generations = 100  # 迭代次数


# 初始化种群
def init_population(num_cities):
    population = []
    for _ in range(population_size):
        chromosome = list(range(1, num_cities + 1))
        random.shuffle(chromosome)
        population.append(chromosome)
    return population


# 计算染色体适应度（总距离）
def fitness(chromosome):
    total_distance = 0
    num_cities = len(chromosome)
    for i in range(num_cities - 1):
        city_start, city_end = chromosome[i], chromosome[i + 1]
        distance = ((city_coords[city_end][0] - city_coords[city_start][0]) ** 2 +
                    (city_coords[city_end][1] - city_coords[city_start][1]) ** 2) ** 0.5
        total_distance += distance

    # 添加最后一个城市和第一个城市之间的距离，构成闭环路径
    first_city, last_city = chromosome[0], chromosome[-1]
    distance = ((city_coords[last_city][0] - city_coords[first_city][0]) ** 2 +
                (city_coords[last_city][1] - city_coords[first_city][1]) ** 2) ** 0.5
    total_distance += distance

    return total_distance


# 选择父代染色体（轮盘赌选择）
def selection(population):
    population_fitness = [fitness(chromosome) for chromosome in population]
    total_fitness = sum(population_fitness)
    probabilities = [fitness / total_fitness for fitness in population_fitness]

    selected_indices = random.choices(range(len(population)), weights=probabilities, k=2)

    return population[selected_indices[0]], population[selected_indices[1]]


# 单点交叉
def crossover(parent1, parent2):
    num_cities = len(parent1)
    crossover_point = random.randint(1, num_cities - 1)

    child1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
    child2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]

    return child1, child2


# 变异操作
def mutation(chromosome):
    num_cities = len(chromosome)

    # 随机选取两个位置进行交换
    index1, index2 = random.sample(range(num_cities), 2)
    chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]

    return chromosome


# 遗传算法主函数
def genetic_algorithm(num_cities):
    # 初始化种群
    best_distance = float('inf')
    population = init_population(num_cities)
    for _ in range(generations):
        new_population = []
        while len(new_population) < population_size:
            # 选择父代染色体
            parent1, parent2 = selection(population)
            # 交叉
            child1, child2 = crossover(parent1, parent2)
            # 变异
            if random.random() < mutation_rate:
                child1 = mutation(child1)
            if random.random() < mutation_rate:
                child2 = mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
        best_chromosome = min(population, key=fitness)
        best_distance = min(best_distance,fitness(best_chromosome))
        df.loc[len(df), 'best'] = best_distance
        print(_,best_distance)
    best_chromosome = min(population, key=fitness)
    best_distance = fitness(best_chromosome)

    return best_chromosome, best_distance


#
if __name__ == '__main__':
    num_cities_100 = 100
    best_path_100, best_distance_100 = genetic_algorithm(num_cities_100)
    print(f"Best path for {num_cities_100} cities: {best_path_100}")
    print(f"Total distance: {best_distance_100}")
    df.to_csv('GA_best.csv')