from app.player import Player
from app.constants import Constants
import pandas as pd
import numpy as np
__author__ = 'Scott'


class Learner(Player):
    def __init__(self):
        super().__init__()
        self._Q = {}
        self._last_state = None
        self._last_action = None
        self._learning_rate = .7
        self._discount = .9
        self._epsilon = .9
        self._learning = True

    def reset_hand(self):
        self._hand = []
        self._last_state = None
        self._last_action = None

    def get_action(self, state):
        if state in self._Q and np.random.uniform(0,1) < self._epsilon:
            action = max(self._Q[state], key = self._Q[state].get)
        else:
            action = np.random.choice([Constants.hit, Constants.stay])
            if state not in self._Q:
                self._Q[state] = {}
            self._Q[state][action] = 0

        self._last_state = state
        self._last_action = action

        return action

    def update(self,new_state,reward):
        if self._learning:
            old = self._Q[self._last_state][self._last_action]

            if new_state in self._Q:
                new = self._discount * self._Q[new_state][max(self._Q[new_state], key=self._Q[new_state].get)]
            else:
                new = 0

            self._Q[self._last_state][self._last_action] = (1-self._learning_rate)*old + self._learning_rate*(reward+new)

    def get_optimal_strategy(self):
        df = pd.DataFrame(self._Q).transpose()
        df['optimal'] = df.apply(lambda x : 'hit' if x['hit'] >= x['stay'] else 'stay', axis=1)
        return df