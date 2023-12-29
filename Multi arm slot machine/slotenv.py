import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x,axis=1).reshape(-1,1))
    return e_x / e_x.sum(axis=1).reshape(-1,1)

### 定义智能体环境
class SlotEnv:
    def __init__(self, n):
        self.n = n
        self.action_space = np.arange(n) # 可选的动作
        ## 假定状态空间与动作空间一样大
        self.state_space = np.arange(n)
        self.st = 0  # 当前状态，初始为0
        ### 定义状态转移率P,表示在状态st下选择动作at后转移到st+1的概率，维度信息（当前状态*选择动作*下一状态）
        self.p_uniform = np.zeros((n,n,n))  ## 均匀分布,所有转换概率相等，用于第一种奖励
        ### 对概率进行归一化,获得真实的概率
        for i in range(n):
            self.p_uniform[i] = softmax(self.p_uniform[i])

        self.p_random = np.random.randn(n,n,n) ## 随机的概率，用于第二种奖励，
        ### 对概率进行归一化,获得真实的概率
        for i in range(n):
            self.p_random[i] = softmax(self.p_random[i])

        ## 策略π,表示在状态st下选择动作at的概率,维度信息（当前状态*选择动作）

        ### 均匀随机的策略，选择每个动作的概率相同
        self.pi_uniform = np.zeros((n,n)) + 1 / n
        self.pi_uniform = softmax(self.pi_uniform)

        ### 高斯分布的策略
        self.pi_random = np.random.randn(n,n)
        self.pi_random = softmax(self.pi_random)

        ### 定义奖励

        ### 上下文无关奖励,基于高斯分布，由于与上下文无关，只与所处状态有关,到达某个状态就会返回该状态的奖励
        self.r_independent = np.random.normal(0, 1, n)

        ### 与上下文有关奖励，需要考虑状态，以及动作
        """
        设定奖励规则为：
        S[n//2]为终止态，表示结束，给惩罚；
        S[n-1]是目标，达到会给与奖励，不终止
        其余奖励与上下文相关，设定为，距离S[n//2]越近，会给惩罚p;距离S[n-1]越近会给奖励r；
        其中r和p都与上下文相关
        """
        self.r_relational = np.zeros(n)
        ## 首先考虑一点，应该有终止态，这是一般的环境都会有的
        self.dead_score = -1 ## 表示结束，基于惩罚
        self.dead_point = n // 2

        self.goal_score = 1 ## 到达目标，获得奖励
        self.goal_point = n-1 # 目标带你

    ### 重置状态,设定为从S[0]开始
    def reset(self):
        self.st = 0
        return

    ### 执行动作,reward_type表示奖励模式，0表示上下文无关，1表示上下文相关
    def step(self, act, reward_type=0):
        if reward_type == 0:
            prob = self.p_uniform[self.st,act].squeeze()
            st_ = np.random.choice(a=self.state_space, size=1, replace=True, p=prob) # 根据状态转移率跳转下一状态
            rt = self.r_independent[st_] # 根据状态获取奖励
            self.st = st_

            return st_, rt, False

        else:
            prob = self.p_random[self.st, act].squeeze()
            st_ = np.random.choice(a=self.state_space, size=1, replace=True, p=prob)  # 根据状态转移率跳转下一状态
            ### 根据上一状态和当前状态计算奖励
            rt, done = self.relational_reward(self.st, st_)
            self.st = st_
            return st_, rt, done

    # 根据上一状态和当前状态计算奖励
    def relational_reward(self, st, st_):
        # 如果是终止点，直接结束
        if st_ == self.dead_point:
            return -1,True
        # 如果是目标点，返回奖励1，但并未终止
        elif st_ == self.goal_point:
            return 1,False
        else:
            dis_pre_g = abs(st - self.goal_point) # 上一状态与目标的距离
            dis_cur_g = abs(st_ - self.goal_point) # 当前状态与目标的距离
            # 离目标更近，则给奖励
            if dis_cur_g <= dis_pre_g:
                r = abs(dis_cur_g - dis_pre_g) / self.n
            else:
                r =  -abs(dis_cur_g - dis_pre_g) / self.n

            dis_pre_d = abs(st - self.dead_point) # 上一状态与终止点的距离
            dis_cur_d = abs(st_ - self.dead_point) # 上一状态与终止点的距离
            # 离终止点更近，给与惩罚
            if dis_cur_d <= dis_pre_d:
                p = abs(dis_cur_d - dis_pre_d) / self.n
            else:
                p = -abs(dis_cur_d - dis_pre_d) / self.n

            r_total = r - p
            return r_total, False

    def get_obs(self):
        return self.st









