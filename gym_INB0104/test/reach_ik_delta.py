import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np

def main():
    render_mode = "human"
    env = gym.make("gym_INB0104/reach_ik_delta", render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=80)    
    camera_id = 1
    waitkey = 1

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        if render_mode == "rgb_array":
            pixels = obs["images"]["front"]
            cv2.resize(pixels, (224, 224))
            cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
            cv2.waitKey(waitkey)
        while not terminated and not truncated:
            if i < 20:
                action = np.array([0.5, -0.5, 0.0, -1.0])
            elif i < 40:
                action = np.array([0.0, 0.0, -0.5, 1.0])
            elif i < 60:
                action = np.array([-0.5, 0.0, 0.0, -1.0])
            elif i < 80:
                action = np.array([0.0, 0.5, 0.0, 1.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.resize(pixels, (224, 224))
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
            i+=1
        


if __name__ == "__main__":
    main()
