import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
import time
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaRealEnv", render_mode=render_mode, ee_dof=4)
    env = TimeLimit(env, max_episode_steps=200)    
    waitkey = 1

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        print(obs['state']['panda/tcp_pos'])
        if render_mode == "rgb_array":
            pixels = obs["images"]["front"]
            cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
            cv2.waitKey(waitkey)
        for i in range(10):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs['state']['panda/tcp_pos'])
            if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
            time.sleep(0.1)
        # while not terminated and not truncated:
        #     if render_mode == "rgb_array":
        #         pixels = obs["images"]["front"]
        #         cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
        #         cv2.waitKey(waitkey)

        #     if i < 15:
        #         action = np.array([0.2, 0.0, 0.0, 0.0, -1.0])
        #     elif i < 50:
        #         action = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
        #     elif i < 80:
        #         action = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        #     elif i < 100:
        #         action = np.array([0.0, 0.0, 0.2, 0.0, 1.0])
        #     elif i < 120:
        #         action = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
        #     elif i < 140:
        #         action = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        #     else:
        #         action = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
            
        #     # # Random action
        #     # action = env.action_space.sample()
            
        #     obs, reward, terminated, truncated, info = env.step(action)
        #     print(obs['images']['front'].shape)
        #     i+=1
        
if __name__ == "__main__":
    main()
