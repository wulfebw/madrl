"""
Multiagent debugging environment
- one step MDP
- observation: each agent receives obs of 0 or 1
- actions: each agent can take action of 0 or 1
- rewards:
    + if all the agents act such that the sum of their actions equals 
        one more than the sum of the observations, they get a reward equal to 
        the sum of the actions. 
    + if that condition does not hold then they get no reward. 
- description: 
    + the idea is that the agents are encouraged to take action 1, but have to 
        coordinate in doing so.
    + rewards are fairly sparse.
    + if a learning algorithm can't find a control policy for this env, then 
        it is likely broken, or will not do well in environments that require
        coordination.

    Examples:

    obs = (1,0,0)
    act = (1,1,0)
    reward = 2

    obs = (0,0,0)
    act = (1,0,0)
    reward = 1

    obs = (1,0,0)
    act = (1,1,1)
    reward = 0

    obs = (1,0,0)
    act = (1,0,0)
    reward = 0
"""

import gym
import numpy as np
from gym import spaces

class OneRoundDeterministicRewardMultiagentEnv(gym.Env):
    def __init__(self, num_agents):
        assert num_agents > 1
        self.num_agents = num_agents
        self._reset()

    def _step(self, actions):
        sum_works = np.sum(actions) == np.sum(self.obs) + 1
        reward = np.sum(actions) if sum_works else 0
        done = True
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        self.obs = np.random.randint(2, size=self.num_agents)
        return self.obs

    def _reset(self):
        return self._get_obs()
