import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
import time
from franka_env.envs.wrappers import Quat2EulerWrapper
np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachStrawbEnv", render_mode=render_mode, randomize_domain=False, ee_dof=6, pos_scale= 0.008, rot_scale = 0.5)
    # env = Quat2EulerWrapper(env)
    env = TimeLimit(env, max_episode_steps=50)    
    waitkey = 1

    obs, info = env.reset()
    initial_pose = obs['state']['tcp_pose']
    print(f"initial pose: {initial_pose}")
    for i in range(20):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs['state']['tcp_pose'])

    final_pose = obs['state']['tcp_pose']
    print(f"initial pose: {initial_pose}")
    print(f"final pose: {final_pose}")
    print(f"diff: {final_pose-initial_pose}")

    if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(0)
        
        
if __name__ == "__main__":
    main()
