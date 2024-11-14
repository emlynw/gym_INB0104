import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaEnv", render_mode=render_mode, randomize_domain=True, ee_dof=4)
    env = TimeLimit(env, max_episode_steps=100)    
    waitkey = 100
    resize_resolution = (480, 480)

    while True:
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                # cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), resize_resolution))
                # cv2.waitKey(waitkey)

            print(i)

            if i < 15:
                action = np.array([0.2, 0.0, 0.0, 0.0, -1.0])
            elif i < 50:
                action = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
            elif i < 80:
                action = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
            elif i < 100:
                action = np.array([0.0, 0.0, 0.2, 0.0, 1.0])
            elif i < 120:
                action = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
            elif i < 140:
                action = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
            else:
                action = np.array([0.0, 1.0, 0.0, 1.0, 1.0])

            # action = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            i+=1
        
if __name__ == "__main__":
    main()
