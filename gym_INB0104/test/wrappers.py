import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np
  
class ActionState(gym.Wrapper):
    def __init__(self, env, state_key = 'state'):
        super().__init__(env)
        self.state_key = state_key
        low = np.concatenate([env.observation_space[state_key].low, env.action_space.low])
        high = np.concatenate([env.observation_space[state_key].high, env.action_space.high])
        shape = env.observation_space[state_key].shape[0] + env.action_space.shape[0]
        print(low)
        print(shape)
        
        self.observation_space[state_key] = Box(low=low, high=high, shape=(shape,), dtype=np.float32)

    def reset(self):
        obs, info = self.env.reset()
        action = np.zeros(self.env.action_space.shape)
        obs[self.state_key] = np.concatenate([obs[self.state_key], action], axis=0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs[self.state_key] = np.concatenate([obs[self.state_key], action], axis=0)
        return obs, reward, terminated, truncated, info