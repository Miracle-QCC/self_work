import numpy as np
import networkx as nx
import random
import copy
import math
from Experiment_DQN_PyTorch_23z.Dijkstra import Dijkstra

class Node(object):
    def __init__(self, data):
        self.data = data
vnf1 = Node(3)
vnf2 = Node(4)
vnf3 = Node(5)
vnf4 = Node(6)
vnf5 = Node(7)
vnf6 = Node(8)
vnf7 = Node(9)
vnf8 = Node(10)

list = [vnf1, vnf2, vnf3, vnf4, vnf5, vnf6, vnf7, vnf8]

sfc_sum=50
vnf_sum=2

slice1=[-1 for i1 in range(0,sfc_sum)]
G_sfc=[-1 for i2 in range(0,sfc_sum)]
bw_sfc=[-1 for i3 in range(0,sfc_sum)]
td_sfc=[-1 for i4 in range(0,sfc_sum)]

for j in range(sfc_sum):
    bw_sfc[j] = np.random.uniform(low=5, high=10, size=1)
    td_sfc[j] = np.random.uniform(low=20, high=50, size=1)
    slice1[j]=random.sample(list, vnf_sum)
    G_sfc[j]=nx.DiGraph()
    for k in range(vnf_sum):
        G_sfc[j].add_node(k, res=slice1[j][k].data)
    G_sfc[j].add_edges_from([(0, 1), (1, 2)])   # 注意改了vnf_sum，则此语句也得改，但只影响绘图，不影响代码
G = nx.Graph()

G.add_node(0, flag=0, vnf=-1, res=400)
G.add_node(1, flag=0, vnf=-1, res=400)
G.add_node(2, flag=0, vnf=-1, res=400)
G.add_node(3, flag=0, vnf=-1, res=400)
G.add_node(4, flag=0, vnf=-1, res=400)
G.add_node(5, flag=0, vnf=-1, res=400)
G.add_node(6, flag=0, vnf=-1, res=400)
G.add_node(7, flag=0, vnf=-1, res=400)
G.add_node(8, flag=0, vnf=-1, res=400)
G.add_node(9, flag=0, vnf=-1, res=400)
G.add_node(10, flag=0, vnf=-1, res=400)
G.add_node(11, flag=0, vnf=-1, res=400)
G.add_node(12, flag=0, vnf=-1, res=400)
G.add_node(13, flag=0, vnf=-1, res=400)

# 宽带有10兆、20兆、50兆、100兆、200兆、500兆，家庭用一般为100兆，百度能搜索到400兆带宽
G.add_edge(0, 1, flag_edge=0, id=0, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(0, 2, flag_edge=0, id=1, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(0, 3, flag_edge=0, id=2, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(1, 2, flag_edge=0, id=19, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(1, 7, flag_edge=0, id=20, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(2, 5, flag_edge=0, id=3, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(3, 4, flag_edge=0, id=4, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(3, 9, flag_edge=0, id=5, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(4, 5, flag_edge=0, id=6, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(4, 6, flag_edge=0, id=7, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(5, 8, flag_edge=0, id=8, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(5, 12, flag_edge=0, id=9, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(6, 7, flag_edge=0, id=10, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(7, 10, flag_edge=0, id=11, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(8, 10, flag_edge=0, id=12, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(9, 11, flag_edge=0, id=13, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(9, 13, flag_edge=0, id=14, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(10, 11, flag_edge=0, id=15, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(10, 13, flag_edge=0, id=16, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(11, 12, flag_edge=0, id=17, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))
G.add_edge(12, 13, flag_edge=0, id=18, bw=400, td=np.random.uniform(low=1, high=7, size=1),
           plr=np.random.uniform(low=0.01, high=0.04, size=1))

a = nx.adjacency_matrix(G).todense()

sfc_placement = [[-1 for j1 in range(vnf_sum)] for i1 in range(sfc_sum)]
node_res=[0 for i in range(14)]
link_res=[0 for i in range(42)]
net_res = [0 for i in range(14)]
sfc_placement_row=[-1 for i in range(vnf_sum)]

G_node_res=[1 for i in range(a.shape[0])]
G_node_res=np.array(G_node_res)

G_node_bw=[1 for i in range(G.size())]
G_node_bw=np.array(G_node_bw)
G_remove = nx.Graph()

class Network_Env():
    def __init__(self):
        self.i=0
        self.j=0
        self.action_space = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        self.n_actions = len(self.action_space)
        self.n_features = 199
        self.S=copy.deepcopy(G_sfc)
        self.N=G.copy()
        self.N_copy=nx.Graph()
        self.sfc_i=self.S[self.i]
        self.sfc_num=sfc_sum
        self.vnf_num=vnf_sum
        self.state_first=-1
        self.next_state=-1
        self.sfc_placement = sfc_placement
        self.reward=0
        self.done=False
        self.flag_j=False
        self.net_res=net_res
        self.node_res=node_res
        self.link_res=link_res
        self.k=0
        self.l=0
        self.flag_action=False
        self.sfc_placement_row=sfc_placement_row
        self.node_path_remove=[]
        self.node_placement_remove=[]
        self.flag_SFC=False
        self.path_vnf=None
        self.flag_link=False
        self.total_link_td=0
        self.sfc_link_td=0
        self.node_link_td=0
        self.sfc_count=0
        self.ave_td=0
        self.ave_res_s = 0
        self.ave_bw_s = 0
        self.total_use_res=0
        self.total_use_bw = 0
        self.dif1=0
        self.dif2=0
        self.dif1_before=0
        self.dif2_before=0
        self.node_res_minmax = [-1 for i in range(a.shape[0])]
        self.node_bw_minmax = [-1 for i in range(G.size())]
        self.G_node_res = G_node_res
        self.flag_td = False
        self.flag_bw=False
        self.same_node_count=0
        self.sum_dif1=0
        self.sum_dif2=0
        self.ave_dif1=0
        self.ave_dif2=0
        self.sum_ave_td = 0     # 每轮的平均值总和再平均
        self.sum_ave_dif1 = 0
        self.sum_ave_dif2 = 0
        self.count_episode=-1
        self.dif1_reset = []     # 存储每一条sfc目前的dif值
        self.dif2_reset = []
        self.flag_episode_first=True

    def network_bw(self):
        node_bw = [0 for i in range(G.size())]
        k1 = 0
        for i in range(a.shape[0]):
            for j in range(i + 1, a.shape[0]):
                try:
                    if isinstance(self.N.edges[i, j]['bw'], int):
                        node_bw[k1] = self.N.edges[i, j]['bw']
                    else:
                        abc = self.N.edges[i, j]['bw'][0]
                        node_bw[k1] = abc
                    k1 += 1
                except KeyError as e:
                    continue
        node_bw = np.array(node_bw)
        y_max = node_bw.max()
        y_min = node_bw.min()
        if y_max == y_min:
            y_min -= 0.1
        node_bw = (node_bw - y_min) / (y_max - y_min)
        return node_bw

    def network_res(self):
        for i in range(a.shape[0]):
            self.node_res[i] = self.N.nodes[i]['res']
        self.node_res=np.array(self.node_res)
        self.x_max = self.node_res.max()
        self.x_min = self.node_res.min()
        self.node_res=np.around(((self.node_res - self.x_min) / (self.x_max - self.x_min)), decimals = 4)
        return self.node_res

    def reset_env(self):
        self.i = 0
        self.j = 0
        self.sfc_count = 0
        self.ave_td = 0
        self.sfc_i = self.S[self.i]
        self.N = G.copy()
        self.total_link_td=0
        self.done = False
        self.sum_dif1 = 0
        self.sum_dif2 = 0
        self.path_vnf = None
        self.flag_episode_first = True

    def reset_net(self):
        self.sfc_i = self.S[self.i]
        self.node_placement_remove = []
        self.node_path_remove = []
        self.sfc_placement_row = [-1 for i in range(a.shape[0])]
        self.sfc_link_td = 0
        self.flag_SFC=False
        self.flag_link=False
        self.flag_action = False
        self.path_vnf=None
        self.dif1_reset = []
        self.dif2_reset = []

    def reset_sfc(self):
        if self.flag_episode_first == True:
            self.N = G.copy()
        else:
            self.N = self.N_copy.copy()

        self.total_link_td -= self.sfc_link_td
        self.i += 1
        self.j = 0
        self.flag_SFC = True
        self.reward=-1

        for i in range(len(self.dif1_reset)):
            self.sum_dif1 -= self.dif1_reset[i]

        for i in range(len(self.dif2_reset)):
            self.sum_dif2 -= self.dif2_reset[i]

    def res_cal(self,action):
        try:
            if self.N.nodes[action]['res'] >= self.sfc_i.nodes[self.j]['res']:
                self.N.nodes[action]['res'] = self.N.nodes[action]['res'] - self.sfc_i.nodes[self.j]['res']
                self.sfc_placement_row[self.j]=action
                return True
            else:
                return False
        except Exception as e:
            print("错误:", e)

    def link(self,j):
        if j>1:
            self.node_placement_remove.append(self.sfc_placement_row[j-2])
        G_remove = self.N.copy()
        for c2 in range(a.shape[0]):
            if c2 in self.node_placement_remove or c2 in self.node_path_remove:
                G_remove.remove_node(c2)

        network = [[0.0 for i2 in range(a.shape[0])] for h2 in range(a.shape[0])]
        network = np.mat(network)
        for j2 in range(a.shape[0]):
            for k2 in range(a.shape[0]):
                try:
                    network[j2, k2] = G_remove.edges[j2, k2]['td']
                except KeyError as e:
                    network[j2, k2] = 0
        self.path_vnf = Dijkstra(network, self.sfc_placement_row[j-1], self.sfc_placement_row[j])

        if self.path_vnf !=None:
            for e2 in range(1, len(self.path_vnf) - 1):
                self.node_path_remove.append(self.path_vnf[e2])
        else:
            self.flag_link = True

    def total_td(self):
        for i in range(len(self.path_vnf)-1):
            self.total_link_td += self.N.edges[self.path_vnf[i],self.path_vnf[i+1]]['td']
            self.sfc_link_td += self.N.edges[self.path_vnf[i],self.path_vnf[i+1]]['td']
            self.node_link_td += self.N.edges[self.path_vnf[i],self.path_vnf[i+1]]['td']

    def cal_bw(self):
        for i in range(len(self.path_vnf) - 1):
            self.N.edges[self.path_vnf[i], self.path_vnf[i + 1]]['bw']-=bw_sfc[self.i]
            if self.N.edges[self.path_vnf[i], self.path_vnf[i + 1]]['bw']<0:
                self.flag_bw=True

    def step(self,action):
        self.dif1_before=self.dif1
        self.dif2_before=self.dif2
        self.node_link_td=0
        t2=self.res_cal(action)
        self.l+=1

        if self.l<3:
            if t2:
                if(self.j>0):
                    self.link(self.j)
                    if self.flag_link == False:
                        self.total_td()
                        self.cal_bw()

                if self.flag_link == False:
                    if self.sfc_link_td <= td_sfc[self.i]:
                        if self.flag_bw==False:
                            self.node_res_minmax = self.network_res()
                            self.net_res = np.array(self.node_res_minmax)
                            self.next_state = np.hstack((self.net_res.reshape(1, 14).squeeze()))
                            # print("self.next_state:",self.next_state)

                            for i in range(a.shape[0]):
                                self.total_use_res += (self.G_node_res[i] - self.node_res_minmax[i])
                            self.ave_res_s=self.total_use_res/a.shape[0]
                            self.total_use_res=0

                            for i in range(a.shape[0]):
                                self.dif1 += pow((self.G_node_res[i] - self.node_res_minmax[i]) - self.ave_res_s, 2)
                            self.dif1 /= a.shape[0]
                            self.dif1_reset.append(self.dif1)
                            self.sum_dif1 += self.dif1
                    else:
                        self.flag_td=True

                if self.path_vnf !=None:
                    if self.sfc_link_td <= td_sfc[self.i]:
                        if self.flag_bw==False:
                            self.node_bw_minmax = self.network_bw()
                            for i in range(G.size()):
                                self.total_use_bw += (G_node_bw[i] - self.node_bw_minmax[i])
                            self.ave_bw_s = self.total_use_bw / (G.size())
                            self.total_use_bw = 0

                            for i in range(G.size()):
                                self.dif2 += pow((G_node_bw[i] - self.node_bw_minmax[i]) - self.ave_bw_s, 2)
                            self.dif2 /= (G.size())
                            self.dif2_reset.append(self.dif2)
                            self.sum_dif2 += self.dif2
                            self.reward = -0.4 * self.node_link_td - 0.3 * (self.dif1 - self.dif1_before) - 0.3 * (self.dif2 - self.dif2_before)
                    else:
                        self.flag_td = True

                self.j += 1
                self.l=0

        else:
            self.flag_action=True
            self.l = 0

        if (self.j == self.vnf_num) and self.flag_action==False and self.flag_link==False and self.flag_td==False:
            self.flag_episode_first = False     # 只要在这一轮中有映射成功的情况，就置为False
            self.sfc_count+=1
            self.i += 1
            self.j = 0
            self.flag_SFC = True
            self.N_copy=self.N.copy()

        if self.flag_action==True:
            self.reset_sfc()
            self.flag_action = False
        if self.flag_link==True:
            self.reset_sfc()
            self.flag_link = False
        if self.flag_td==True or self.flag_bw==True:
            self.reset_sfc()
        if self.flag_td == True:
            self.flag_td = False
        if self.flag_bw == True:
            self.flag_bw = False

        if self.i==self.sfc_num:
            self.count_episode+=1
            self.ave_td=self.total_link_td/self.sfc_count
            self.ave_dif1 = self.sum_dif1/(self.sfc_count*self.vnf_num)
            self.ave_dif2 = self.sum_dif2/(self.sfc_count*(self.vnf_num-1))
            if self.count_episode>9:
                if not np.isnan(self.ave_td):
                    self.sum_ave_td+=self.ave_td
                if not np.isinf(self.ave_dif1):
                    self.sum_ave_dif1 += self.ave_dif1
                if not np.isinf(self.ave_dif1):
                    self.sum_ave_dif2 += self.ave_dif2
            self.done = True
        return self.next_state,self.reward,self.done