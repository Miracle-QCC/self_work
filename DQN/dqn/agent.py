import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


class Agent:
    """ Agent Class (Network) for DQN
    """

    def __init__(self, state_dim, action_dim, lr, dueling):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        # Initialize Deep Q-Network
        self.model = self.network(dueling)
        self.model.compile(Adam(lr), 'mse')

    def huber_loss(self, y_true, y_pred):
        return K.mean(K.sqrt(1 + K.square(y_pred - y_true)) - 1, axis=-1)

    def network(self, dueling):
        """ Build Deep Q-Network
        """
        inp = Input(shape=self.state_dim)
        x = Flatten()(inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)

        if dueling:
            # Split into value and advantage streams
            value = Dense(1)(x)
            advantage = Dense(self.action_dim)(x)
            # Combine value and advantage streams
            x = Lambda(lambda i: i[0] + i[1] - K.mean(i[1], axis=1, keepdims=True))([value, advantage])
        else:
            x = Dense(self.action_dim, activation='linear')(x)

        return Model(inputs=inp, outputs=x)

    def fit(self, inp, targ):
        """ Perform one epoch of training
        """
        self.model.fit(inp, targ, epochs=1, verbose=0)

    def predict(self, inp):
        """ Q-Value Prediction
        """
        return self.model.predict(inp)
    def save(self, path):
        if(self.dueling):
            path += '_dueling'
        self.model.save_weights(path + '.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
