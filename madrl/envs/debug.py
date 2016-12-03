"""
Multiagent debugging environment
- one step MDP
- each agent receives obs of 0 or 1
- each agent can take action of 0 or 1
- if all the agents take the same action they get reward of +1 else -1
"""

import gym
import numpy as np
from gym import spaces

class OneRoundDeterministicRewardMultiagentEnv(gym.Env):
    def __init__(self, num_agents):
        assert num_agents > 1
        self.num_agents = num_agents
        self.action_space = spaces.Discrete(num_agents * 2)
        self.observation_space = spaces.Discrete(num_agents)
        self._reset()

    def _step(self, actions):
        assert all([self.action_space.contains(a) for a in actions])
        all_same = all(v1 == v2 for (v1,v2) in zip(actions, actions[1:]))
        as_reward = actions[0] == self.obs[0]
        reward = 1 if (all_same and as_reward) else -1
        done = True
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        self.obs = np.repeat(np.random.randint(2), self.num_agents)
        return self.obs

    def _reset(self):
        return self._get_obs()
