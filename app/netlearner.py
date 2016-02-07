from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import numpy as np
from app.constants import Constants
from app.qlearner import Learner
import pandas as pd

class DQNLearner(Learner):
    def __init__(self):
        super().__init__()
        self._learning = True
        self._learning_rate = .1
        self._discount = .1
        self._epsilon = .9

        # Create Model
        model = Sequential()

        model.add(Dense(2, init='lecun_uniform', output_dim=10))
        model.add(Activation('relu'))

        model.add(Dense(10, init='lecun_uniform', output_dim=4))
        model.add(Activation('relu'))

        model.add(Dense(4, init='lecun_uniform', output_dim=2))
        model.add(Activation('linear'))

        rms = RMSprop()
        model.compile(loss='mse', optimizer=rms)

        self._model = model


    def get_action(self, state):
        rewards = self._model.predict([np.array([state])], batch_size=1)

        if np.random.uniform(0,1) < self._epsilon:
            if rewards[0][0] > rewards[0][1]:
                action = Constants.hit
            else:
                action = Constants.stay
        else:
            action = np.random.choice([Constants.hit, Constants.stay])

        self._last_state = state
        self._last_action = action
        self._last_target = rewards


        return action

    def update(self,new_state,reward):
        if self._learning:
            rewards = self._model.predict([np.array([new_state])], batch_size=1)
            maxQ = rewards[0][0] if rewards[0][0] > rewards[0][1] else rewards[0][1]
            new = self._discount * maxQ

            if self._last_action == Constants.hit:
                old = self._last_target[0][0]

                self._last_target[0][0] = (1-self._learning_rate)*old + self._learning_rate*(reward+new)

            else:
                old = self._last_target[0][1]
                self._last_target[0][1] = (1-self._learning_rate)*old + self._learning_rate*(reward+new)

            # Update model
            self._model.fit(np.array([self._last_state]), self._last_target, batch_size=1, nb_epoch=1, verbose=0)

    def get_optimal_strategy(self):

        index = []
        for x in range(0,21):
            for y in range(1,11):
                index.append((x,y))

        df = pd.DataFrame(index = index, columns = ['hit', 'stay'])

        for ind in index:
            outcome = self._model.predict([np.array([ind])], batch_size=1)
            df.loc[ind, 'hit'] = outcome[0][0]
            df.loc[ind, 'stay'] = outcome[0][1]


        df['optimal'] = df.apply(lambda x : 'hit' if x['hit'] >= x['stay'] else 'stay', axis=1)
        return df

