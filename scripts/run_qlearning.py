
import itertools
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'madrl')
sys.path.append(os.path.abspath(path))

import algos.tabular
import envs.debug

def run_qlearning():
    # const
    num_agents = 3
    num_episodes = 1000

    # env
    env = envs.debug.OneRoundDeterministicRewardMultiagentEnv(num_agents)

    # algo
    actions = list(itertools.product(range(2), repeat=num_agents))
    discount = 1
    explorationProb = .3
    stepSize = .1
    algo = algos.tabular.QLearningAlgorithm(
        actions, discount, explorationProb, stepSize)

    # train
    for episode in range(num_episodes):
        obs = env.reset()
        action = algo.getAction(tuple(obs))
        next_obs, reward, done, _ = env.step(action)
        algo.incorporateFeedback(tuple(obs), action, reward, None)

    # final weights
    lines = sorted(algo.weights.items(), key=lambda (k,v): v, reverse=True)
    print("state        action      value")
    for line in lines: print(line)

if __name__ == '__main__':
    run_qlearning()