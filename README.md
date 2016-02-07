# blackjacklearner
A example of Q-learning and Q-learning using a neural network written in Python for a simplified Blackjack implementation.

This project provides a simple implementation of Blackjack for Q-learning. The Learner is trained via the standard algorithm, 
and once training has completed, will play the game to determine a win/loss ratio for the training session. There is also an option
to use the DQNLearner which uses a variant of the Deep Q-Network from the paper "Playing Atari with Deep Reinforcement Learning" by Mnih et al.

The number of training and game rounds are parameters found in the main function. The results of the training are updated (printed to the screen)
periodically, and the win/loss ratio is reported in the same manner. The optimal strategy is saved to a CSV after training is completed.
