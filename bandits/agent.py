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
