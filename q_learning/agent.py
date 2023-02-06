from typing import Optional

import numpy as np
from numpy import ndarray


def create_q_table(num_states: int, num_actions: int) -> ndarray:
    """
    Function that returns a q_table as an array of shape (num_states, num_actions) filled with zeros.

    :param num_states: Number of states.
    :param num_actions: Number of actions.
    :return: q_table: Initial q_table.
    """
    return np.zeros((num_states, num_actions), dtype=np.float32)


class QLearnerAgent:
    """
    The agent class for exercise 1.
    """

    def __init__(self, num_states: int,
                 num_actions: int,
                 learning_rate: float,
                 gamma: float,
                 epsilon_max: Optional[float] = None,
                 epsilon_min: Optional[float] = None,
                 epsilon_decay: Optional[float] = None,):
        """
        :param num_states: Number of states.
        :param num_actions: Number of actions.
        :param learning_rate: The learning rate.
        :param gamma: The discount factor.
        :param epsilon_max: The maximum epsilon of epsilon-greedy.
        :param epsilon_min: The minimum epsilon of epsilon-greedy.
        :param epsilon_decay: The decay factor of epsilon-greedy.
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = create_q_table(num_states, num_actions)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon = epsilon_max

    def greedy_action(self, observation: int) -> int:
        """
        Return the greedy action.

        :param observation: The observation.
        :return: The action.
        """
        action = int(np.argmax(self.q_table[observation]))
        return action

    def act(self, observation: int, training: bool = True) -> int:
        """
        Return the action.

        :param observation: The observation.
        :param training: Boolean flag for training.
        :return: The action.
        """
        if training:
            action = self.epsilon_greedy_action(observation)
        else:
            action = self.greedy_action(observation)
        return action

    def epsilon_greedy_action(self, observation: int) -> int:
        """
        Return an action using epsilon greedy exploration.

        :param observation: The observation.
        :return: The action.
        """
        if np.random.sample() < self.epsilon:
            action = np.random.choice(self.q_table.shape[-1])
        else:
            action = self.greedy_action(observation)
        return action

    def learn(self, obs: int, act: int, rew: float, done: bool, next_obs: int) -> None:
        """
        Update the Q-Value.

        :param obs: The observation.
        :param act: The action.
        :param rew: The reward.
        :param done: Done flag.
        :param next_obs: The next observation.
        """
        self.q_table[obs][act] += self.learning_rate * (
                rew + int(not done) * self.gamma * np.max(self.q_table[next_obs]) - self.q_table[obs][act])
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
