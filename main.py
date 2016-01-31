__author__ = 'Scott'
import numpy as np
import pandas as pd

class Constants:
    hit = 'hit'
    stay = 'stay'
    player1= 'you'
    player2= 'dealer'

class Game:
    def __init__(self, num_learning_rounds):
        self.p = None
        self.win = 0
        self.loss = 0
        self.game = 1
        self._num_learning_rounds = num_learning_rounds

    def run(self):
        d, p, p2, winner = self.reset_round()

        state = self.get_starting_state(p,p2)

        while True:
            p1_action = p.get_action(state)
            p2_action = p2.get_action(state)

            if p1_action == Constants.hit:
                p.hit(d)

            if p2_action == Constants.hit:
                p2.hit(d)

            if self.determine_if_bust(p):
                winner = Constants.player2
                break

            elif self.determine_if_bust(p2):
                winner = Constants.player1
                break

            if p1_action == p2_action and p1_action == Constants.stay:
                break

            state = self.get_state(p, p1_action, p2)
            p.update(state,0)


        if winner is None:
            winner = self.determine_winner(p,p2)


        if winner == Constants.player1:
            self.win += 1
            p.update(self.get_ending_state(p,p1_action,p2),1)
        else:
            self.loss += 1
            p.update(self.get_ending_state(p,p1_action,p2),-1)

        self.game += 1

        self.report()

        if self.game == self._num_learning_rounds:
            print("Turning off learning!")
            self.p._epsilon = 2
            self.p._learning_rate = 0
            self.win = 0
            self.loss = 0

    def report(self):
        if self.game % 10000 == 0:
            print(str(self.game) +" : "  +str(self.win / (self.win + self.loss)))

    def get_state(self,player1,p1_action, player2):
        return (player1.get_hand_value(), player2.get_original_showing_value())

    def get_starting_state(self,player1, player2):
        return (player1.get_hand_value(), player2.get_showing_value())

    def get_ending_state(self,player1,p1_action, player2):
        return (player1.get_hand_value(), player2.get_hand_value())

    def determine_winner(self,player1,player2):
        if player1.get_hand_value() == 21 or (player1.get_hand_value() > player2.get_hand_value() and player1.get_hand_value() <= 21):
            return Constants.player1
        else:
            return Constants.player2

    def determine_if_bust(self,player):
        if player.get_hand_value() > 21:
            return True
        else:
            return False

    def reset_round(self):
        d = Deck()
        if self.p is None:
            self.p = Learner()
        else:
            self.p.reset_hand()

        p = self.p
        p2 = Player()

        winner = None
        p.hit(d)
        p2.hit(d)
        p.hit(d)
        p2.hit(d)

        return d, p, p2, winner

class Deck:
    def __init__(self):

        self.shuffle()

    def shuffle(self):
        cards = (np.arange(0,10) + 1)
        cards = np.repeat(cards,4*3) #4 suits x 3 decks
        np.random.shuffle(cards)
        self._cards = cards.tolist()

    def draw(self):
        return self._cards.pop()


class Player:
    def __init__(self):
        self._hand = []
        self._original_showing_value = 0

    def get_hand(self):
        return self._hand

    def get_action(self, state = None):
        if self.get_hand_value() < 15:
            return Constants.hit
        else:
            return Constants.stay

    def get_hand_value(self):
        return sum(self._hand)

    def get_showing_value(self):
        showing = sum(self._hand[1:])
        self._original_showing_value = showing
        return showing

    def get_original_showing_value(self):
        return self._original_showing_value

    def hit(self, deck):
        card_value = deck.draw()
        self._hand.append(card_value)

    def stay(self):
        return True

    def reset_hand(self):
        self._hand = []

    def update(self,new_state,reward):
        pass


class Learner(Player):
    def __init__(self):
        Player.__init__(self)
        self._Q = {}
        self._last_state = None
        self._last_action = None
        self._learning_rate = .7
        self._discount = .9
        self._epsilon = .9

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
        old = self._Q[self._last_state][self._last_action]

        if new_state in self._Q:
            new = self._discount * self._Q[new_state][max(self._Q[new_state], key=self._Q[new_state].get)]
        else:
            new = 0

        self._Q[self._last_state][self._last_action] = (1-self._learning_rate)*old + self._learning_rate*(reward+new)

def main():
    num_learning_rounds = 2000000
    game = Game(num_learning_rounds)
    number_of_test_rounds = 1000000
    for k in range(0,num_learning_rounds + number_of_test_rounds):
        game.run()

    df = pd.DataFrame(game.p._Q).transpose()
    df['optimal'] = df.apply(lambda x : 'hit' if x['hit'] >= x['stay'] else 'stay', axis=1)
    print(df)
    df.to_csv('optimal_policy.csv')

if __name__ == "__main__":
    main()