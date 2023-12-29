from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy

def off_policy_mc_prediction_ordinary_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using ordinary importance
    # sampling (Hint: Sutton Book p. 109, every-visit implementation is fine)
    #####################
    Q = np.copy(initQ)
    C = np.zeros_like(initQ)  # Cumulative sum of weights
    nS, nA = env_spec.nS, env_spec.nA
    for traj in trajs:
        G = 0  # Return (cumulative discounted reward)
        W = 1  # Importance sampling ratio


        for t,(state, action, reward, next_state) in enumerate(reversed(traj)):
            # state, action, reward, next_state = traj[t]

            G = G * env_spec.gamma + reward
            C[state, action] += W
            Q[state, action] += (W / C[state, action]) * (G - Q[state, action])
    return Q

def off_policy_mc_prediction_weighted_importance_sampling(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array
) -> np.array:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using behavior policy bpi
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        pi: evaluation target policy
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_pi$ function; numpy array shape of [nS,nA]
    """

    #####################
    # TODO: Implement Off Policy Monte-Carlo prediction algorithm using weighted importance
    # sampling (Hint: Sutton Book p. 110, every-visit implementation is fine)
    #####################
    nS, nA = env_spec.nS, env_spec.nA
    Q = initQ.copy()
    C = np.zeros_like(Q)

    for traj in trajs:
        G = 0  # Return
        W = 1  # Weight

        for t, (s, a, r, s_next) in enumerate(reversed(traj)):
            G = env_spec.gamma * G + r
            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
            # if a != pi.action(s):
            #     break
            W *= pi.action_prob(s, a) / bpi.action_prob(s, a)
    return Q




