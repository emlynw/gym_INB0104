import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
import time
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachStrawbEnv", render_mode=render_mode, randomize_domain=False, ee_dof=6, pos_scale= 0.01, rot_scale = 0.2)
    env = TimeLimit(env, max_episode_steps=50)    
    waitkey = 1

    obs, info = env.reset()
    initial_pose = obs['state']['tcp_pose']
    print(f"initial pose: {initial_pose}")
    for i in range(40):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs['state']['tcp_pose'])
        if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
    final_pose = obs['state']['tcp_pose']
    print(f"final pose: {final_pose}")
    print(f"diff: {final_pose-initial_pose}")
        
        
if __name__ == "__main__":
    main()
