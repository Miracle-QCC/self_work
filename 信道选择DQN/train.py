from tqdm import tqdm
from DQN import DDQN_Agent
import torch
from Cus_Env import Channel_Env
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # 信道数量
    channel_num = 10
    # 邻接矩阵
    adj_ma = None
    ### 客户智能体参数
    params_cus = {
        'gamma': 0.8,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 200,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': channel_num,
        'action_space_dim': channel_num,
    }
    N = 6 ### 客户智能体数量
    cus_agents = [DDQN_Agent(**params_cus) for i in range(N)]

    ### 干扰智能体参数

    params_noise = {
        'gamma': 0.8,
        'epsi_high': 1.0,
        'epsi_low': 0.2,
        'decay': 200,
        'lr': 0.0001,
        'capacity': 1000,
        'batch_size': 32,
        'state_space_dim': channel_num,
        'action_space_dim': channel_num,
    }
    noise_agent = DDQN_Agent(**params_noise)

    env = Channel_Env(channel_num, adj_ma, noise_agent)

    MAX_EPISODE = 100
    MAX_steps = 200
    for e in tqdm(range(MAX_EPISODE)):
        ###  重置状态
        state = env.reset()
        r_cus_ep = 0
        for s in range(MAX_steps):
            ##动作mask，用来表示哪些channel已经被选择,不能再选
            act_tion_mask = np.zeros(N)
            ### 先进行干扰
            act_noise = noise_agent.act(torch.from_numpy(state).to(device))
            env.q_learning_noise(act_noise)
            ## 返回干扰后的状态
            state = env.reset()
            ### 如果有邻接矩阵
            if adj_ma:
                states = []
                ## 先取出每个agent的观测
                for agent in cus_agents:
                    states.append(env.get_state_env(agent))
                ### 取出邻居的观测最大值
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        states[i] = np.maximum(states[i], states[j])
            else:
                states = [env.state for i in range(N)]

            for i in range(N):
                cur_agent = cus_agents[i]
                cur_agent_act = cur_agent.act(torch.from_numpy(states[i]).to(device))
                state, r_cus, r_noise, done = env.step(cur_agent_act)

                cur_agent.put(state, cur_agent_act, r_cus,state, done)

            noise_agent.put(state, act_noise, r_noise, state, done)

            ## 进行训练
            for i in range(N):
                cus_agents[i].update()

            r_cus_ep += r_cus
            if done:
                break
        noise_agent.update()

        print(f"{e} reward:", r_cus_ep)







