from typing import Tuple

import numpy as np
from env import EnvWithModel
from policy import Policy
class Mypolicy(Policy):
    def __init__(self, nS, nA):
        super(Mypolicy, self).__init__()
        self.nS = nS
        self.nA = nA
        self.prob_tablle = np.zeros((nS, nA)) + 1/nA
    
    def action(self,state:int) -> int:
        """
        input:
            state
        return:
            action
        """
        
        return np.argmax(self.prob_tablle[state])
    
    def action_prob(self,state:int,action:int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        return self.prob_tablle[state,action]

    def set_prob(self, state, action):
        for a in range(self.nA):
            self.prob_tablle[state,a] = 0
        self.prob_tablle[state,action] = 1

def Q2V(pi,Q,s,nA):
    v = 0
    for a in range(nA):
        v += pi.action_prob(s,a) * Q[s,a]
    return v

def value_prediction(env: EnvWithModel, pi: Policy, initV: np.array, theta: float) -> Tuple[np.array, np.array]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        pi: policy
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        V: $v_\pi$ function; numpy array shape of [nS]
        Q: $q_\pi$ function; numpy array shape of [nS,nA]
    """
    #####################
    #     # TODO: Implement Value Prediction Algorithm (Hint: Sutton Book p.75)
    #     #####################
    V = initV.copy()
    nS, nA = env._env_spec._nS, env._env_spec._nA
    Q = np.zeros((nS, nA))

    while True:
        delta = 0
        for s in range(nS):
            v = V[s]

            V[s] = sum([pi.action_prob(s, a) * sum([env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + env._env_spec.gamma * V[s_prime])
                                             for s_prime in range(nS)]) for a in range(nA)])
            delta = max(delta, abs(v - V[s]))
            for a in range(nA):
                q = Q[s, a]
                Q[s,a] = sum([env.TD[s,a,s_prime] * pi.action_prob(s, a) * (env.R[s,a,s_prime] + env._env_spec.gamma * V[s_prime])
                                       for s_prime in range(nS)])
                delta = max(delta, abs(q - Q[s,a]))
        if delta < theta:
            break
    # Calculate Q function based on the final V function
    # Q = np.zeros((nS, nA))
    #
    # while True:
    #     delta = 0
    #     for s in range(nS):
    #         for a in range(nA):
    #             q = Q[s,a]
    #             Q[s,a] = sum([env.TD[s,a,s_prime] * pi.action_prob(s, a) * (env.R[s,a,s_prime] + env._env_spec.gamma * Q2V(pi,Q,s_prime,nA))
    #                                for s_prime in range(nS)])
    #             delta = max(delta, abs(q - Q[s,a]))
    #     if delta < theta:
    #         break
    for s in range(nS):
        for a in range(nA):
            if pi.action_prob(s, a) > 0:
                Q[s,a] = Q[s,a] / pi.action_prob(s, a)
    return V, Q

def value_iteration(env: EnvWithModel, initV: np.array, theta: float) -> Tuple[np.array, Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
    """
    #####################
    #     # TODO: Implement Value Iteration Algorithm (Hint: Sutton Book p.83)
    #     #####################
    V = initV.copy()
    nS, nA = env._env_spec._nS, env._env_spec._nA
    pi = Mypolicy(nS, nA)  # Initialize a random policy initially


    while True:
        delta = 0
        for s in range(nS):
            v = V[s]
            Q_s = np.array([sum([env.TD[s, a, s_prime] * (env.R[s, a, s_prime] + env._env_spec.gamma * V[s_prime])
                                 for s_prime in range(nS)]) for a in range(nA)])
            V[s] = max(Q_s)
            pi.set_prob(s, np.argmax(Q_s))  # Update policy to be greedy

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V, pi