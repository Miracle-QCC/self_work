import tensorflow as tf
import numpy as np
from typing import Iterable

class PiApproximationWithNN():
    def __init__(self, state_dims, num_actions, alpha):
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(state_dims,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_actions, activation='softmax')
        ])
        self.optimizer = tf.optimizers.Adam(alpha, beta_1=0.9, beta_2=0.999)

    def __call__(self, s):
        s = tf.convert_to_tensor([s], dtype=tf.float32)
        return self.model(s).numpy()

    def update(self, s, a, gamma_t, delta):
        with tf.GradientTape() as tape:
            pi = self(s)
            action_probs = tf.gather_nd(pi, tf.stack([tf.range(tf.shape(a)[0]), a], axis=1))
            loss = -tf.math.log(action_probs) * delta * gamma_t
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

class VApproximationWithNN():
    def __init__(self, state_dims, alpha):
        self.state_dims = state_dims
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(state_dims,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.optimizer = tf.optimizers.Adam(alpha, beta_1=0.9, beta_2=0.999)

    def __call__(self, s):
        s = tf.convert_to_tensor([s], dtype=tf.float32)
        return self.model(s).numpy()

    def update(self, s, G):
        with tf.GradientTape() as tape:
            v = self(s)
            loss = tf.reduce_mean(tf.square(G - v))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

def REINFORCE(env, gamma, num_episodes, pi, V):
    G_0 = []
    for episode in range(num_episodes):
        s = env.reset()
        done = False
        episode_rewards = []
        states = []
        actions = []
        while not done:
            action_probs = pi(s)
            a = np.random.choice(len(action_probs), p=action_probs)
            states.append(s)
            actions.append(a)
            s, reward, done, _ = env.step(a)
            episode_rewards.append(reward)
        G = np.sum([gamma**i * r for i, r in enumerate(episode_rewards)])
        G_0.append(G)
        for t in range(len(episode_rewards)):
            G_t = sum([gamma**(k-t-1) * r for k, r in enumerate(episode_rewards[t+1:], t+1)])
            delta = G_t - V(states[t])
            V.update(states[t], G_t)
            pi.update(states[t], actions[t], gamma**t, delta)
    return G_0

# 注意：此代码需要TensorFlow环境和一个OpenAI Gym环境来运行。
