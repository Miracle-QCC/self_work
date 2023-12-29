from DQN import DDQN_Agent
import gym

params = {
    'gamma': 0.8,
    'epsi_high': 0.9,
    'epsi_low': 0.05,
    'decay': 200,
    'lr': 0.001,
    'capacity': 1000000,
    'batch_size': 256,
    'state_space_dim': 4,
    'action_space_dim': 2,
}

agent = DDQN_Agent(**params)
env = gym.make('CartPole-v0')
for e in range(1000):
    obs = env.reset()
    if isinstance(obs,tuple):
        obs = obs[0]
    e_r = 0
    while True:
        act = agent.act(obs)
        obs_next,r,done,_,_ = env.step(act)

        if done:
            r = -1
        agent.put(obs,act,r,obs_next,done)
        e_r += r
        agent.update()
        obs = obs_next

        if done:
            break
    print(e_r)

