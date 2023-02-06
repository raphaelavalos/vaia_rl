from typing import Optional
import numpy as np


class BaseAgent:
    """Base agent class."""

    def __init__(self, arms: int):
        self.arms = arms

    def reset(self):
        """This method should reset the agent so it can start learning in a new run."""
        raise NotImplemented

    def greedy_action(self) -> int:
        """This method returns the current greedy action breaking ties if needed."""
        raise NotImplemented

    def act(self, t) -> int:
        """This method returns the action based on the current timestep t."""
        raise NotImplemented

    def learn(self, act: int, reward: float):
        """This method does a learning step based on the action selected by the agent and the reward obtained."""
        raise NotImplemented


class EpsilonGreedyAgent(BaseAgent):
    """Epsilon greedy agent."""

    def __init__(self, arms: int, epsilon: float, epsilon_decay: Optional[float] = None,
                 epsilon_min: Optional[float] = None):
        super().__init__(arms)
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_value = np.zeros(self.arms)
        self.arm_count = np.zeros(self.arms, dtype=np.int32)

    def reset(self):
        self.epsilon = self.initial_epsilon
        self.q_value = np.zeros(self.arms)
        self.arm_count = np.zeros(self.arms, dtype=np.int32)

    def greedy_action(self) -> int:
        action = np.argmax(self.q_value)
        return action

    def act(self, t) -> int:
        if self.epsilon > np.random.random():
            action = np.random.choice(self.arms)
        else:
            action = self.greedy_action()
        return action

    def learn(self, act: int, reward: float):
        self.arm_count[act] += 1
        step_size = 1 / self.arm_count[act]
        self.q_value[act] += step_size * (reward - self.q_value[act])
        if self.epsilon_decay:
            if self.epsilon_min:
                self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
            else:
                self.epsilon *= self.epsilon_decay

class SoftmaxAgent(BaseAgent):
    """Softmax Agent"""

    def __init__(self, arms: int, tau: float, tau_decay: Optional[float] = None, tau_min: Optional[float] = None):
        super().__init__(arms)

        # TODO: add needed variables here.

    def reset(self):
        # TODO: complete
        raise NotImplementedError()

    def act(self, t) -> int:
        # TODO: complete
        raise NotImplementedError()

    def learn(self, act: int, reward: float):
        # TODO: complete
        # Do not forget to update epsilon.
        raise NotImplementedError()


class UCBAgent(BaseAgent):
    """UCB Agent."""
    def __init__(self, arms: int, c: float):
        super().__init__(arms)
        # TODO: add needed variables here.

    def reset(self):
        # TODO: complete
        raise NotImplementedError()

    def act(self, t) -> int:
        # TODO: complete
        raise NotImplementedError()

    def learn(self, act: int, reward: float):
        # TODO: complete
        # Do not forget to update epsilon.
        raise NotImplementedError()
