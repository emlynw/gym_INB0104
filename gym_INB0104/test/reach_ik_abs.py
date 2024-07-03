import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/reach_ik_abs", render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=200)    
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
            if i < 50:
                action = np.array([0.53, 0.0, 0.02, -1.0])
            elif i < 100:
                action = np.array([0.53, 0.0, 0.02, 1.0])
            elif i < 150:
                action = np.array([0.53, 0.0, 0.3, 1.0])
            elif i < 200:
                action = np.array([0.53, 0.0, 0.3, -1.0])
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"alignment reward: {info['alignment_reward']}")

            if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.resize(pixels, (224, 224))
                # write reward on the image
                pixels = cv2.putText(pixels.copy(), f"reward: {reward:.3f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
            i+=1
        


if __name__ == "__main__":
    main()
