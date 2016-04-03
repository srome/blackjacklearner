from app.game import Game
from app.netlearner import DQNLearner
from app.qlearner import Learner


def main():
    num_learning_rounds = 20000
    game = Game(num_learning_rounds, DQNLearner()) #Deep Q Network Learner
    #game = Game(num_learning_rounds, Learner()) #Q learner
    number_of_test_rounds = 1000
    for k in range(0,num_learning_rounds + number_of_test_rounds):
        game.run()

    df = game.p.get_optimal_strategy()
    print(df)
    df.to_csv('optimal_policy.csv')

if __name__ == "__main__":
    main()