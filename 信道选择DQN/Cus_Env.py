import random
import numpy as np


class Channel_Env:
    def __init__(self, channel_num, adj_matrix, q_noise_net):
        self.channel_num = channel_num
        self.adj_ma = adj_matrix

        self.state = np.clip(np.random.randn(channel_num), 0, 0.3)  ### 正常的信道值很低，0.3以下
        self.channel_idx = 0
        self.q_noise_net = q_noise_net


    def step(self, act):
        """
            act是我们Agent选择的动作，理论上应该是[0~10],表示选择了第n个信道;
        """
        ###  如果客户选中的是干扰频道，那么给予惩罚
        done = False
        if self.state[act] >= 0.5:
            r_cus = -1
            r_noise = 1
            return self.state, r_cus, r_noise, done
        else:
            r_cus = 1
            r_noise = -1
            return self.state, r_cus, r_noise, done

    def reset(self):
        """
        重置状态，全部置为0，默认全0为初始状态
        """
        self.state = np.clip(np.random.randn(self.channel_num), 0, 0.3)  ### 正常的信道值很低，0.3以下
        return self.state


    def random_noise(self):
        """
        随机干扰
        :return:
        """
        channel = random.randint(0, self.channel_num) # 随机选择一个信道进行干扰
        ### 被干扰的信道，功率变高
        self.state[channel] = np.random.uniform(low=0.5, high=1.0)

    def cyclic_noise(self):
        """
        扫频干扰
        :return:
        """
        self.state[self.channel_idx] = np.random.uniform(low=0.5, high=1.0)
        self.channel_idx = (self.channel_idx + 1) % self.channel_num

    def q_learning_noise(self, act):
        """
        利用noise网络来选择信道干扰
        :return:
        """
        self.state[act] = np.random.uniform(low=0.5, high=1.0)


    def get_state_env(self, agent):
        """
        每个Agent获得的状态不一样
        :param agent: 指定的Agent
        :return: 返回对应的观测
        """
        #### 具体怎样的观测需要你自己实现了,就是针对不同Agent返回不同的观测状态

        return self.state
    # DQN
    # DDQN
    # DDQN -> VALUE, Adv
    # PRIORITY REPLAY







