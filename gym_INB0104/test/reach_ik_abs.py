import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKAbsEnv", render_mode=render_mode, randomize_domain=False)
    env = TimeLimit(env, max_episode_steps=200)    
    waitkey = 10

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        rotate = True
        if render_mode == "rgb_array":
            pixels = obs["images"]["front"]
            cv2.resize(pixels, (224, 224))
            cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
            cv2.waitKey(waitkey)
        while not terminated and not truncated:
            if rotate:
                if i < 50:
                    action = np.array([0.12, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
                elif i < 80:
                    action = np.array([0.12, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0])
                elif i < 120:
                    action = np.array([0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                elif i < 140:
                    action = np.array([0.12, 0.1, 0.0, -0.2, -0.2, -0.2, 1.0])
                elif i < 160:
                    action = np.array([0.12, -0.1, 0.0, 0.2, 0.2, 0.2, 1.0])
            else:
                if i < 15:
                    action = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                elif i < 50:
                    action = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0])
                elif i < 80:
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                elif i < 100:
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.resize(pixels, (224, 224))
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
            i+=1
        
if __name__ == "__main__":
    main()
