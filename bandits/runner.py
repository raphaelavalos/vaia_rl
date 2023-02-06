import numpy as np
from env import Env
from typing import Optional
from agent import BaseAgent, EpsilonGreedyAgent, SoftmaxAgent, UCBAgent
from numpy import ndarray
import matplotlib.pyplot as plt


def train(env: Env, agent: BaseAgent, num_steps: int) -> ndarray:
    """
    Training loop.
    :param env: The sample environment.
    :param agent: The agent.
    :param num_steps: How many actions or arms can the agent pull before ending.
    :return: Ndarray containing the cummulative average for each timestep. Size: (num_steps).
    """

    reward_list = []

    for i in range(num_steps):

        action = agent.act(i)
        reward = env.reward(action)
        agent.learn(action, reward)

        reward_list.append(reward)

    cumulative_avg = np.cumsum(reward_list)/(np.arange(num_steps) + 1)

    return cumulative_avg


def run_experiment(env: Env, agent: BaseAgent, num_steps: int, num_experiments: int) -> ndarray:
    """
    Running several experiments to obtain informative statistics
    :param env: The sample environment.
    :param agent: The agent.
    :param num_steps: How many actions or arms can the agent pull before ending.
    :param num_experiments: How many experiments will be run. 
    :return: Ndarray containing the cummulative average for each timestep of each experiment. Size: (num_experiments, num_steps). 
    """

    cum_rew_per_exp = []
    for _ in range(num_experiments):
        cum_rew_per_exp.append(train(env, agent, num_steps))        
        agent.reset()
    
    return np.array(cum_rew_per_exp)

def plot_experiment(cum_avg_rewards: ndarray, label: Optional[str] = ""):
    """
    Plot the obtained statistics of the experiments with the agents
    :param cum_avg_rewards: Ndarray containing the cummulative average for each timestep of each experiment. Size: (num_experiments, num_steps). 
    :param label: Optional string for the label of the plot
    """

    mean = cum_avg_rewards.mean(0)
    std = cum_avg_rewards.std(0)
    avg_cum_rewards = mean*np.arange(1,cum_avg_rewards.shape[1]+1,1)
    x = np.arange(0,len(mean),1)

    plt.subplot(121)
    plt.plot(mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.1)
    plt.xlabel("Number of timesteps")
    plt.ylabel("Mean reward")
    plt.legend()

    plt.subplot(122)
    plt.plot(x, avg_cum_rewards, label=label)
    plt.xlabel("Number of timesteps")
    plt.ylabel("Mean cumulative reward")
    plt.legend()



if __name__ == '__main__':
    # Initialize environment and define parameters
    # TODO: this is a base code to run the experiments once you have implemented the functions in agent.py
    # TODO: You should play around with the different parameters.
    # TODO: There are three levels of difficulty: 0, 1, 2.
    env = Env(difficulty=0)
    num_arms = env.get_arms()
    epsilon = 0.8
    epsilon_decay = 0.9995
    epsilon_min = 0.1
    c = 2
    tau = 2
    tau_decay = None
    tau_min = None
    num_steps = 200
    num_experiments = 100


    # General params
    num_steps = 200
    num_experiments = 100
    
    # Create agents
    eps_greedy_agent = EpsilonGreedyAgent(num_arms, epsilon, epsilon_decay, epsilon_min)
    ucb_agent = UCBAgent(num_arms, c)
    softmax_agent = SoftmaxAgent(num_arms, tau, tau_decay, tau_min)

    # Run experiments on eps greedy agent
    eps_greedy_exp = run_experiment(env, eps_greedy_agent, num_steps, num_experiments)
    # Run experiments on softmax agent
    softmax_exp = run_experiment(env, softmax_agent, num_steps, num_experiments)
    # Run experiments on UCB agent
    ucb_exp = run_experiment(env, ucb_agent, num_steps, num_experiments)

    plt.figure(figsize=(20, 10), dpi=80)
    # Plot statistics of experiments with eps greedy agent
    plot_experiment(eps_greedy_exp, label= f"epsilon greedy, eps = {epsilon}, decay = {epsilon_decay}, min eps = {epsilon_min}")
    # Plot statistics of experiments with eps softmax agent
    plot_experiment(softmax_exp, label= f"softmax, tau = {tau}, decay = {tau_decay}, min tau = {tau_min}")   
    # Plot statistics of experiments with greedy agent
    plot_experiment(ucb_exp, label=f"UCB, c = {c}")

    plt.show()



