# 导入PyTorch和相关库
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch import FloatTensor, LongTensor, ByteTensor
from collections import namedtuple
import random 
Tensor = FloatTensor

# 设置超参数
EPSILON = 0
GAMMA = 0.9
TARGET_NETWORK_REPLACE_FREQ = 30       
MEMORY_CAPACITY = 1000
BATCH_SIZE = 80
LR = 1     

# 定义神经网络模型
class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet,self).__init__()                  
        self.linear1 = nn.Linear(35,35)  # 输入大小为35，输出大小为35的全连接层
        self.linear2 = nn.Linear(35,5)   # 输入大小为35，输出大小为5的全连接层
    def forward(self,s):
        s=torch.FloatTensor(s)  # 转换输入为PyTorch张量      
        s = s.view(s.size(0),1,35)  # 将输入变换为合适的形状
        s = self.linear1(s)  # 第一个全连接层
        s = self.linear2(s)  # 第二个全连接层
        return s           

# 定义DQN智能体类
class DQN(object):
    def __init__(self):
        self.net,self.target_net = DQNNet(),DQNNet()  # 创建DQN神经网络和目标神经网络       
        self.learn_step_counter = 0  # 学习步数
        self.memory = []  # 经验回放缓冲区
        self.position = 0  # 当前经验存储位置
        self.capacity = MEMORY_CAPACITY  # 经验回放缓冲区容量       
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)  # 优化器
        self.loss_func = nn.MSELoss()  # 损失函数

    # 选择动作的方法
    def choose_action(self,s,e):
        x=np.expand_dims(s, axis=0)  # 增加维度以匹配网络输入形状
        if np.random.uniform() < 1-e:  # 使用贪婪策略或随机策略
            actions_value = self.net.forward(x)  # 基于当前状态选择动作            
            action = torch.max(actions_value,-1)[1].data.numpy()  # 选择值最大的动作
            action = action.max()  # 转为标量           
        else: 
            action = np.random.randint(0, 5)  # 随机选择动作
        return action

    # 存储经验的方法
    def push_memory(self, s, a, r, s_):
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # 如果经验回放缓冲区未满，添加新条目
        self.memory[self.position] = Transition(torch.unsqueeze(torch.FloatTensor(s), 0),torch.unsqueeze(torch.FloatTensor(s_), 0),\
                                                torch.from_numpy(np.array([a])),torch.from_numpy(np.array([r],dtype='float32')))#
        self.position = (self.position + 1) % self.capacity  # 更新经验存储位置

    # 从经验中获取样本的方法
    def get_sample(self,batch_size):
        sample = random.sample(self.memory,batch_size)  # 随机采样一批经验样本
        return sample

    # 学习和更新网络参数的方法
    def learn(self):
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            self.target_net.load_state_dict(self.net.state_dict())  # 定期更新目标网络
        self.learn_step_counter += 1
        
        transitions = self.get_sample(BATCH_SIZE)  # 从经验回放缓冲区中获取批次样本
        batch = Transition(*zip(*transitions))  # 解包样本元组

        b_s = Variable(torch.cat(batch.state))  # 转为Tensor并封装为Variable
        b_s_ = Variable(torch.cat(batch.next_state))
        b_a = Variable(torch.cat(batch.action))
        b_r = Variable(torch.cat(batch.reward))    
             
        q_eval = self.net.forward(b_s).squeeze(1).gather(1,b_a.unsqueeze(1).to(torch.int64))  # 计算Q值
        q_next = self.target_net.forward(b_s_).detach()  # 目标网络Q值
        q_target = b_r + GAMMA * q_next.squeeze(1).max(1)[0].view(BATCH_SIZE, 1).t()  # 目标Q值           
        loss = self.loss_func(q_eval, q_target.t())  # 计算损失        
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新网络参数
        return loss

# 定义经验存储结构
Transition = namedtuple('Transition',('state', 'next_state','action', 'reward'))

# 导入Gym环境
import gymnasium as gym
import highway_env
import matplotlib
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 配置环境参数
config = \
    {
    "observation": 
         {
        "type": "Kinematics",
        "vehicles_count": 5,
        "stack_size":3,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": 
            {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
            },
        'screen_height': 1500,
        'screen_width': 6000, 
        'show_trajectories': True,
        "absolute": False,
        "order": "sorted"
        },
    "lanes_count":3,
    "simulation_frequency": 8,  
    "policy_frequency": 2,  
    'collision_reward': -10,
    # 'right_lane_reward': 0.1,
    'high_speed_reward': 2,
    #'lane_change_reward': 0,
    #'reward_speed_range': [20, 30],
    #'offroad_terminal': False,
    }

# 创建Gym环境对象
env = gym.make("highway-v0",render_mode="human")
env.unwrapped.configure(config)

# 创建DQN智能体
dqn=DQN()
count=0

reward=[]
avg_reward=0
all_reward=[]

time_=[]
all_time=[]

collision_his=[]
all_collision=[]
while True:
    done = False    
    start_time=time.time()
    s = env.reset()[0]
    
    while not done:
        e = np.exp(-count/300)  # 减小探索率
        a = dqn.choose_action(s,e)  # 选择动作
        s_, r, done, info, _ = env.step(a)  # 与环境交互并观察奖励、新状态
        env.render()  # 渲染环境
        
        dqn.push_memory(s, a, r, s_)  # 存储经验
        
        if ((dqn.position !=0)&(dqn.position % 99==0)):
            loss_=dqn.learn()  # 学习更新
            count+=1
            print('trained times:',count*5)
            
            if (count % 5 == 0):
                plt.figure(figsize=(60,10),dpi=60)
                plt.plot(reward)
                plt.title("reward")
                plt.show()
                plt.figure(figsize=(20,10),dpi=60)
                plt.plot(time_)
                plt.title("time_")
                plt.show()
                collision_rate=np.mean(collision_his)
                all_collision.append(collision_rate)
                avg_reward=np.mean(reward)
                avg_time=np.mean(time_)
                print("avg_reward:",avg_reward)
                print("avg_time",avg_time)
                print(len(collision_his)/count)
                                
                all_reward.append(avg_reward)
                all_time.append(avg_time)
                
            #    print(all_reward)  
                plt.figure(figsize=(20,10),dpi=60)
                plt.plot(all_reward)
                plt.title("all_reward")
                plt.show()
                plt.figure(figsize=(20,10),dpi=60)
                plt.plot(all_time)
                plt.title("all_time")
                plt.show()
                #plt.figure(figsize=(20,10),dpi=60)
                #plt.plot(collision_rate)
                #plt.title("collision_rate")
                #plt.show()
                plt.figure(figsize=(20,10),dpi=60)
                plt.plot(all_collision)
                plt.title("all_collision")
                plt.show()
                
                reward=[]
                time_=[]
                collision_his=[]
                
        s = s_
        reward.append(r)      

    end_time=time.time()
    episode_time=end_time-start_time
    time_.append(episode_time)
    
    print(info)
    is_collision=0 if info==True else 1
    collision_his.append(is_collision)  # 记录碰撞历史
