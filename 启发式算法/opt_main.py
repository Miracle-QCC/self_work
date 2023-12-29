import matplotlib.pyplot as plt
import itertools
import random
import copy
import time
import sys
import math
import tkinter #//GUI模块
import threading
from functools import reduce
import numpy as np
import pandas as pd

# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''
Start_City = 0  # 该值范围在0~city_num-1，对应坐标为(distance_x[Start_City],distance_y[Start_City])。当设置为None时，随机选取初始起点。
TOP_K_level_1_weight = 3
TOP_K_level_2_weight = 2
TOP_K_level_3_weight = 1
TOP_K_GAMA = 1.5
TOP_K = 20
(ALPHA, BETA, RHO, Q) = (1.0,2.0,0.5,100.0)
# 城市数，蚁群
(city_num, ant_num) = (100,50)
distance_x = [208,66,98,243,301,17,616,16,289,611,538,572,547,47,470,789,731,680,108,644,519,792,104,29,466,406,521,436,461,87,792,133,161,525,528,214,465,106,778,206,724,537,120,81,116,742,781,534,744,386,362,786,177,530,769,168,363,601,265,721,309,590,630,792,336,755,756,592,510,273,391,649,248,428,604,725,791,164,730,257,26,318,294,413,700,508,689,396,230,542,111,282,557,222,793,212,715,499,712,531]
distance_y = [11,754,245,583,369,500,663,498,459,695,65,272,247,494,372,6,267,633,343,371,632,106,408,386,0,385,770,706,114,230,786,172,515,750,28,606,81,245,253,676,573,642,45,565,220,114,193,758,722,624,39,332,330,119,365,278,35,465,158,724,798,62,215,568,153,179,344,139,627,96,89,374,128,552,346,429,515,684,502,752,74,367,632,587,328,394,282,745,678,16,475,377,466,703,145,397,159,617,577,247]
#城市距离和信息素
distance_graph = [ [0.0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph = [ [1.0 for col in range(city_num)] for raw in range(city_num)]



#----------- 蚂蚁 -----------
class Ant(object):

    # 初始化
    def __init__(self,ID):

        self.ID = ID                 # ID
        self.__clean_data()          # 随机初始化出生点

    # 初始数据
    def __clean_data(self):

        self.path = []               # 当前蚂蚁的路径           
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_city = -1       # 当前停留的城市
        self.open_table_city = [True for i in range(city_num)] # 探索城市的状态

        city_index = Start_City if Start_City is not None else random.randint(0,city_num-1)
        city_index = 10
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self):

        next_city = -1
        select_citys_prob = [0.0 for i in range(city_num)]  #存储去下个城市的概率
        total_prob = 0.0

        # 获取去下一个城市的概率
        for i in range(city_num):
            if self.open_table_city[i]:
                try :
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow((1.0/distance_graph[self.current_city][i]), BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print ('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID = self.ID, current = self.current_city, target = i))
                    sys.exit(1)

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break

        # 未从概率产生，顺序选择一个未访问城市
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break

        if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)

        # 返回下一个城市序号
        return next_city

    # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in range(1, city_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += distance_graph[start][end]

        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        if temp_distance == self.best_distance:
            # pos1, pos2 = random.sample(self.path,2)
            # new_path = copy.deepcopy(self.path)
            # new_path[new_path.index(pos1)], new_path[new_path.index(pos2)] = self.path[pos2], self.path[pos1]
            # random_temp_distance = 0.0

            # for i in range(1, city_num):
            #     start, end = new_path[i], new_path[i-1]
            #     random_temp_distance += distance_graph[start][end]
            # # 回路
            # end = new_path[0]
            # random_temp_distance += distance_graph[start][end]
            # # if random_temp_distance < temp_distance:
            # self.path = new_path
            # temp_distance = random_temp_distance
            pos = random.choice(range(city_num-6))
            points = self.path[pos:pos+6]
            combinations = [list(c) for c in itertools.permutations(points, 6)]
            for combo in combinations:
                if combo[0] != points[0] or combo[-1] != points[-1]:
                    continue
                new_path = self.path[:pos] + list(combo) + self.path[pos+6:]
                random_temp_distance = 0.0

                for i in range(1, city_num):
                    start, end = new_path[i], new_path[i-1]
                    random_temp_distance += distance_graph[start][end]
                # 回路
                end = new_path[0]
                random_temp_distance += distance_graph[start][end]
                if random_temp_distance<temp_distance:
                    self.path = new_path
                    temp_distance = random_temp_distance
        self.total_distance = temp_distance


    # 移动操作
    def __move(self, next_city):

        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1

    # 搜索路径
    def search_path(self,best_distance):
        self.best_distance = best_distance
        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        while self.move_count < city_num:
            # 移动到下一个城市
            next_city =  self.__choice_next_city()
            self.__move(next_city)

        # 计算路径总长度
        self.__cal_total_distance()

#----------- TSP问题 -----------

class TSP(object):

    def __init__(self, root, width = 800, height = 600, n = city_num):

        # 创建画布
        self.root = root                               
        self.width = width      
        self.height = height
        # 城市数目初始化为city_num
        self.n = n
        # tkinter.Canvas
        self.canvas = tkinter.Canvas(
                root,
                width = self.width,
                height = self.height,
                bg = "#EBEBEB",             # 背景白色 
                xscrollincrement = 1,
                yscrollincrement = 1
            )
        self.canvas.pack(expand = tkinter.YES, fill = tkinter.BOTH)
        self.title("TSP蚁群算法(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5
        self.__lock = threading.RLock()     # 线程锁
        self.best_result = []
        self.avg_result = []
        self.__bindEvents()
        self.new()

        # 计算城市之间的距离
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] =float(int(temp_distance + 0.5))
        x = 1
    # 按键响应程序
    def __bindEvents(self):

        self.root.bind("q", self.quite)        # 退出程序
        self.root.bind("n", self.new)          # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)         # 停止搜索
        self.root.bind("w", self.write)        # 重置

    # 更改标题
    def title(self, s):

        self.root.title(s)

    # 初始化
    def new(self, evt = None):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()     # 清除信息 
        self.nodes = []  # 节点坐标
        self.nodes2 = [] # 节点对象

        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - self.__r,
                    y - self.__r, x + self.__r, y + self.__r,
                    fill = "#ff0000",      # 填充红色
                    outline = "#000000",   # 轮廓白色
                    tags = "node",
                )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x,y-10,              # 使用create_text方法在坐标（302，77）处绘制文字
                    text = '('+str(x)+','+str(y)+')',    # 所绘制文字的内容
                    fill = 'black'                       # 所绘制文字的颜色为灰色
                )

        # 顺序连接城市
        #self.line(range(city_num))

        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 1.0

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1)                          # 初始最优解
        self.best_ant.total_distance = 1 << 31           # 初始最大距离
        self.iter = 1                                    # 初始化迭代次数 
        self.best_result.append([])
        self.avg_result.append([])
    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")
        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill = "#000000", tags = "line")
            return i2

        # order[-1]为初始值
        reduce(line2, order, order[-1])

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print (u"\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    # 开始搜索
    def search_path(self, evt = None):

        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()
        best_result = []
        avg_result = []
        st = time.time()
        while self.iter<100:
            # 遍历每一只蚂蚁
            tmp_avg_result = []
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path(self.best_ant.total_distance)
                tmp_avg_result.append(ant.total_distance)
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            self.avg_result[-1].append(sum(tmp_avg_result)/len(tmp_avg_result))
            self.best_result[-1].append(self.best_ant.total_distance)    
            if self.iter > 1:
                top = True
            else:
                top = False
            # 更新信息素
            self.__update_pheromone_gragh(top)
            print (u"迭代次数：",self.iter,u"最佳路径总距离：",int(self.best_ant.total_distance))
            # 连线
            self.line(self.best_ant.path)
            # 设置标题
            self.title("TSP蚁群算法(n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d" % self.iter)
            # 更新画布
            self.canvas.update()
            self.iter += 1
        print(f'第{len(self.avg_result)}轮跑数结束，耗时：{time.time()-st}s')

    # 更新信息素
    def __update_pheromone_gragh(self,top):

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        if top:
            distances_list = []
            ant_list = []
            for ant in self.ants:
                distances_list.append(ant.total_distance)
                ant_list.append(ant)
            for i in range(1,TOP_K+1):
                if i == 1:
                    if min(distances_list) < self.best_ant.total_distance:
                        ant = ant_list[distances_list.index(min(distances_list))]
                        for i in range(1,city_num):
                            start, end = ant.path[i-1], ant.path[i]
                            # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                            temp_pheromone[start][end] += Q / ant.total_distance*(i*TOP_K_GAMA*TOP_K_level_1_weight)
                            temp_pheromone[end][start] = temp_pheromone[start][end]
                        distances_list[distances_list.index(min(distances_list))] = np.inf
                    else:
                        ant = self.best_ant
                        for i in range(1,city_num):
                            start, end = ant.path[i-1], ant.path[i]
                            # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                            temp_pheromone[start][end] += Q / ant.total_distance*(i*TOP_K_GAMA*TOP_K_level_2_weight)
                            temp_pheromone[end][start] = temp_pheromone[start][end]
                else:
                    ant = ant_list[distances_list.index(min(distances_list))]
                    for i in range(1,city_num):
                        start, end = ant.path[i-1], ant.path[i]
                        # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                        temp_pheromone[start][end] += Q / ant.total_distance*(i*TOP_K_GAMA*TOP_K_level_3_weight)
                        temp_pheromone[end][start] = temp_pheromone[start][end]
                    distances_list[distances_list.index(min(distances_list))] = np.inf

        # 获取每只蚂蚁在其路径上留下的信息素
        else:
            for ant in self.ants:
                for i in range(1,city_num):
                    start, end = ant.path[i-1], ant.path[i]
                    # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                    temp_pheromone[start][end] += Q / ant.total_distance
                    temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]
    def write(self, evt = None):
        multi_avg_result = []
        multi_best_result = []
        for j in range(len(self.avg_result[0])):
            tmp_avg = 0
            tmp_best = np.inf
            for i in range(len(self.avg_result)):
                tmp_avg += self.avg_result[i][j]
                if tmp_best> self.best_result[i][j]:
                    tmp_best = self.best_result[i][j]
            multi_avg_result.append(tmp_avg/len(self.avg_result))
            multi_best_result.append(tmp_best)
        df = pd.DataFrame()
        df.loc[:,"avg"] = multi_avg_result
        df.loc[:,"best"] = multi_best_result
        df.to_csv(f"./{len(self.avg_result)}_rounds_result.csv", index=False)
        # 绘制适应度迭代图
        plt.figure()
        plt.title(f'{len(self.avg_result)} rounds')
        plt.plot(multi_avg_result)
        plt.plot(multi_best_result)
        plt.grid(True)
        plt.show()
    # 主循环
    def mainloop(self):
        self.root.mainloop()

#----------- 程序的入口处 -----------

if __name__ == '__main__':


    TSP(tkinter.Tk()).mainloop()