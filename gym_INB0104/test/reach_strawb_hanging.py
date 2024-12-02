import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    height, width = 720, 720
    env = gym.make("gym_INB0104/ReachIKDeltaStrawbHangingEnv", height=height, width=width, render_mode=render_mode, randomize_domain=True, ee_dof=6)
    env = TimeLimit(env, max_episode_steps=100)    
    waitkey = 10
    resize_resolution = (height, width)

    while True:
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            if render_mode == "rgb_array":
                pixels = obs["images"]["wrist1"]
                cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                cv2.waitKey(waitkey)
            # print(i)
            if i < 10:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.97])
            elif i < 20:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.82])
            elif i < 30:
                action = np.array([0.0, 0.0, 0, 0.0, 0.0, 0.0, -0.58])
            elif i < 40:
                action = np.array([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -0.43])
            elif i < 50:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.24])
            elif i < 60:
                action = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -0.08])
            elif i < 70:
                action = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.05])
            else:
                action = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

            # action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01])
            
            obs, reward, terminated, truncated, info = env.step(action)
            print(obs['state']['panda/gripper_vec'])
            i+=1
        
if __name__ == "__main__":
    main()
