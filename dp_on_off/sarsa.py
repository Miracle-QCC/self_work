import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 tile_width: np.array):
        # Initialize the variables
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = np.ceil((self.state_high - self.state_low) / self.tile_width) + 1
        self.total_tiles = np.prod(self.num_tiles) * self.num_tilings
        self.dim = self.total_tiles * self.num_actions

    def feature_vector_len(self) -> int:
        # Return the dimension of the feature vector
        return int(self.dim)

    def __call__(self, s, done, a) -> np.array:
        # Compute the feature vector
        if done:
            return np.zeros(self.feature_vector_len())
        active_tiles = self.get_active_tiles(s)
        feature_vector = np.zeros(self.feature_vector_len())
        for tile in active_tiles:
            index = tile + (a * self.total_tiles)
            feature_vector[index] = 1
        return feature_vector

    def get_active_tiles(self, s: np.array) -> np.array:
        # Calculate the active tiles
        active_tiles = []
        for tiling in range(self.num_tilings):
            tiles = []
            position = s + (tiling / self.num_tilings) * self.tile_width
            for i in range(len(s)):
                tiles.append(int((position[i] - self.state_low[i]) / self.tile_width[i]))
            active_tiles.append(np.ravel_multi_index(tiles, self.num_tiles))
        return np.array(active_tiles)


def SarsaLambda(env, gamma: float, lam: float, alpha: float, X: StateActionFeatureVectorWithTile, num_episode: int) -> np.array:
    # Define the epsilon-greedy policy
    def epsilon_greedy_policy(s, done, w, epsilon=0.1):
        nA = env.action_space.n
        Q = [np.dot(w, X(s, done, a)) for a in range(nA)]
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    # Initialize the weight vector
    w = np.zeros((X.feature_vector_len()))

    # Sarsa(λ) algorithm implementation
    for episode in range(num_episode):
        s = env.reset()
        done = False
        z = np.zeros_like(w)
        Q_old = 0
        while not done:
            a = epsilon_greedy_policy(s, done, w)
            x = X(s, done, a)
            Q = np.dot(w, x)
            delta = -Q_old
            Q_old = Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q) * z - alpha * (Q - np.dot(w, x)) * x

            s, reward, done, _ = env.step(a)
            Q = np.dot(w, X(s, done, a))
            delta += reward + gamma * Q

            if done:
                w += alpha * (delta - Q_old) * z

    return w

# Assuming we have an OpenAI Gym environment, this code is ready to run.
