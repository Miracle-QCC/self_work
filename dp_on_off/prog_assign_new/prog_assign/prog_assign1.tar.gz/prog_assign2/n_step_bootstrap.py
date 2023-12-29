from typing import Iterable, Tuple

import numpy as np
from env import EnvSpec
from policy import Policy


class Mypolicy(Policy):
    def __init__(self, nS, nA):
        super(Mypolicy, self).__init__()
        self.nS = nS
        self.nA = nA
        self.prob_tablle = np.zeros((nS, nA)) + 1 / nA

    def action(self, state: int) -> int:
        """
        input:
            state
        return:
            action
        """

        return np.argmax(self.prob_tablle[state])

    def action_prob(self, state: int, action: int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        return self.prob_tablle[state, action]

    def set_prob(self, state, action):
        for a in range(self.nA):
            self.prob_tablle[state, a] = 0
        self.prob_tablle[state, action] = 1

def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    #####################
    # TODO: Implement On Policy n-Step TD algorithm
    # sampling (Hint: Sutton Book p. 144)
    #####################

    V = initV.copy()
    gamma = env_spec.gamma
    for i, val in enumerate(trajs):
        T = len(val)
        for t, arr in enumerate(val):
            tau = t - n + 1
            if tau >= 0:
                G = 0
                for nxt in range(tau + 1, min(t + n, T)+1):
                    G += (gamma ** (nxt - tau - 1)) * val[nxt - 1][2]
                    if (tau + n) < T:
                        G += (gamma ** (n)) * V[val[tau + n][0]]
                    V[val[tau][0]] += alpha * (G - V[val[tau][0]])
            if tau == T - 1:
                break
    return V

def off_policy_n_step_sarsa(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    n:int,
    alpha:float,
    initQ:np.array
) -> Tuple[np.array,Policy]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        bpi: behavior policy used to generate trajectories
        n: how many steps?
        alpha: learning rate
        initQ: initial Q values; np array shape of [nS,nA]
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
        policy: $pi_star$; instance of policy class
    """

    #####################
    # TODO: Implement Off Policy n-Step SARSA algorithm
    # sampling (Hint: Sutton Book p. 149)
    #####################

    nS, nA = initQ.shape
    Q = np.copy(initQ)
    policy = Mypolicy(nS, nA)  # Initialize a policy
    for s in range(nS):
        for a in range(nA):
            policy.prob_tablle[s,a] = bpi.action_prob(s,a)
    for traj in trajs:
        T = len(traj)
        S, A, R, _ = zip(*traj)

        for t in range(T+n-1):
            tau = t - n + 1
            if tau >= 0:
                rho = 1.0
                G = 0

                for i in range(tau + 1, min(tau + n - 1, T - 1)):
                    rho *= policy.action_prob(S[i], A[i]) / bpi.action_prob(S[i], A[i])
                for i in range(tau + 1, min(tau + n, T)):
                    G += (env_spec.gamma ** (i - tau - 1)) * R[i]

                if tau + n < T:
                    G += (env_spec.gamma ** n) * Q[S[tau + n], A[tau + n]]

                Q[S[tau], A[tau]] += alpha * rho * (G - Q[S[tau], A[tau]])
                # Update policy to be greedy with respect to Q
                policy.set_prob(S[tau],np.argmax(S[tau]))

    return Q, policy