import numpy as np
import torch
from policy import Policy

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    for episode in range(num_episode):
        state = env.reset()
        states, rewards = [state], [0]

        T = float('inf')
        t = 0
        print(episode)
        while True:
            if t < T:
                action = pi.action(state)
                next_state, reward, done, _ = env.step(action)

                states.append(next_state)
                rewards.append(reward)

                if done:
                    T = t + 1

            tau = t - n + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n, T))])
                G = np.array(G).reshape(1,)
                with torch.no_grad():
                    if tau + n < T:
                            G += gamma ** n * V(states[tau + n])
                # delta = G - V(states[tau])
                V.update(alpha, G, states[tau])

            if tau == T - 1:
                break

            t += 1
            state = next_state

