
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'madrl')
sys.path.append(os.path.abspath(path))

import algos.tabular
import envs.debug
import utils

def run_qlearning():
    # const
    num_agents = 5
    num_episodes = 10000

    # env
    env = envs.debug.OneRoundDeterministicRewardMultiagentEnv(num_agents)

    # algo
    actions = list(itertools.product(range(2), repeat=num_agents))
    discount = 1
    explorationProb = .3
    stepSize = .3
    algo = algos.tabular.QLearningAlgorithm(
        actions, discount, explorationProb, stepSize)

    # train
    rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        action = algo.getAction(tuple(obs))
        next_obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        algo.incorporateFeedback(tuple(obs), action, reward, None)

    # final weights
    lines = sorted(algo.weights.items(), key=lambda (k,v): v, reverse=True)
    print("state        action      value")
    for line in lines: print(line)
    avg_rewards = utils.moving_average(rewards, num_episodes / 100)
    plt.plot(range(len(avg_rewards)), avg_rewards, alpha=.5, 
        label="average rewards")
    plt.legend(loc=8)
    plt.show()

if __name__ == '__main__':
    run_qlearning()