import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from agent import Agent
from random import random, randrange
from memory_buffer import MemoryBuffer
from networks import tfSummary

STATE_SIZE = 3


class DQN:
    """ Deep Q-Learning Main Algorithm
    """

    def __init__(self, action_dim, state_dim, args):
        """ Initialization
        """
        # Environment and DQN parameters
        self.action_dim = action_dim
        self.state_dim = (args.consecutive_frames,) + tuple(state_dim)
        self.lr = 2.5e-4
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.buffer_size = 20000

        # Create DQN agent
        self.agent = Agent(self.state_dim, action_dim, self.lr, args.dueling)

        # Memory Buffer for Experience Replay
        self.buffer = MemoryBuffer(self.buffer_size)

        self.q, self.q_targ = 0.0, 0.0
    def policy_action(self, s):
        """ Apply an epsilon-greedy policy to pick the next action
        """
        if random() <= self.epsilon:
            return randrange(self.action_dim)
        else:
            return np.argmax(self.agent.predict(s)[0])

    def train_agent(self, batch_size, epsilon):
        """ Train the Q-network on a batch sampled from the buffer
        """
        # Sample experience from the memory buffer
        s, a, r, d, new_s ,idx= self.buffer.sample_batch(batch_size)
        s = np.squeeze(s, axis=1)
        new_s = np.squeeze(new_s, axis=1)

        # Apply the Bellman equation on the batch samples to train our DQN
        q = self.agent.predict(s)
        q_targ = self.agent.predict(new_s)
        self.q = q
        self.q_targ = q_targ
        for i in range(s.shape[0]):
            if d[i]:
                q[i, a[i]] = r[i]
            else:
                next_best_action = np.argmax(q_targ[i, :])
                q[i, a[i]] = r[i] + self.gamma * q_targ[i, next_best_action]

        # Train on the batch
        self.agent.fit(s, q)

        # Decay epsilon
        epsilon.append(self.epsilon)
        self.epsilon *= self.epsilon_decay

    def train(self, env, args, summary_writer):
        """ Main DQN Training Algorithm
        """
        results = []
        scores = []
        mean_q_values = []
        epsilon = []
        speeds1 = []
        speeds2=[]
        distances1 = []
        distances2=[]
        i = 0
        j = 0
        losses = []
        worst_reward = float('inf')
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:
            # Reset episode
            i += 1
            speed = 0.0
            distance = 0.0
            time, cumul_reward, done = 0, 0, False
            if i < args.nb_episodes -1:
                old_state = env.reset2()
                old_state = np.reshape(old_state, [1, STATE_SIZE])
                old_state = np.expand_dims(old_state, axis=0)
            else:
                old_state = env.reset()
                old_state = np.reshape(old_state, [1, STATE_SIZE])
                old_state = np.expand_dims(old_state, axis=0)
            while not done:
                j += 1
                if args.render:
                    env.render()
                # Actor picks an action (following the policy)
                a = self.policy_action(old_state)

                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a, old_state)
                if i == 1:
                    speeds1.append(speed)
                if i==args.nb_episodes:
                    speeds2.append(speed)
                speed += new_state[0]
                if i==1:
                    distances1.append(distance)
                if i==args.nb_episodes:
                    distances2.append(distance)
                distance += new_state[0] * 1
                new_state = np.reshape(new_state, [1, STATE_SIZE])
                new_state = np.expand_dims(new_state, axis=0)
                # Memorize for experience replay
                self.memorize(old_state, a, r, done, new_state)
                # Update current state
                old_state = new_state
                cumul_reward += r
                q_values = self.agent.predict(old_state)[0]
                td_error = self.q_targ - self.q
                sess = tf.Session()
                # 计算损失值
                loss = tf.reduce_mean(tf.square(td_error))
                loss = sess.run(loss)
                losses.append(loss)

                mean_q_value = np.mean(q_values)
                mean_q_values.append(mean_q_value)

                # Train DQN
                if self.buffer.size() > args.batch_size:
                    self.train_agent(args.batch_size, epsilon)

                mean = tfSummary('mean_q_values', mean_q_value)
                summary_writer.add_summary(mean, global_step=time)
                time += 1
                mean_q, stdev = np.mean(np.array(scores)), np.std(np.array(scores))
            if cumul_reward < worst_reward:
                worst_reward = cumul_reward

            scores.append(cumul_reward)
            env.close()
            results.append([e, mean_q, stdev])
            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()
            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
        print("The worst_reward is:", worst_reward)
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(scores)
        plt.title("Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        # 绘制损失曲线
        plt.subplot(2, 2, 2)
        plt.plot(losses)
        plt.title("Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")

        # 绘制平均 Q 值曲线
        plt.subplot(2, 2, 3)
        plt.plot(mean_q_values)
        plt.title("Mean Q Values")
        plt.xlabel("Episode")
        plt.ylabel("Mean Q Value")

        # 绘制 epsilon 曲线
        plt.subplot(2, 2, 4)
        plt.plot(epsilon)
        plt.title("Epsilon")
        plt.xlabel("time")
        plt.ylabel("Epsilon")

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.plot(speeds1,color='r',)
        plt.plot(speeds2,color='y')
        plt.title('speeds')
        plt.xlabel("time")
        plt.ylabel("speeds")
        plt.legend(['Before training', 'after training'])
        plt.subplot(1, 2, 2)
        plt.plot(distances1,color='r')
        plt.plot(distances2,color='y')
        plt.xlabel("time")
        plt.ylabel("distance")
        plt.title('distances')
        plt.legend(['Before training', 'after training'])
        plt.show()
        return results


    def memorize(self, state, action, reward, done, new_state):
        """ Store experience in the memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

