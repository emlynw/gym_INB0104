import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
import time
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaRealEnv", render_mode=render_mode, ee_dof=4, pos_scale= 0.2, rot_scale = 0.05)
    env = TimeLimit(env, max_episode_steps=200)    
    waitkey = 1

    obs, info = env.reset()
    print(obs['state']['panda/tcp_pos'])
    for i in range(10):
        action = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs['state']['panda/tcp_pos'])
        if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
        
        
if __name__ == "__main__":
    main()
