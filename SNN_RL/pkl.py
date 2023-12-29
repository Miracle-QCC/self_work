import pickle


def load_reward_clock_funcs(path):
    with open(path, "rb") as f:
        clock_funcs = pickle.load(f)
    return clock_funcs

if __name__ == '__main__':
    load_reward_clock_funcs('/home/qcj/workcode/self_work/SNN_RL/Cassie_mujoco_RL-main_/Cassie_mujoco_RL-main/cassie/rewards/reward_clock_funcs/incentive_clock_smooth.pkl')