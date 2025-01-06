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
    env = TimeLimit(env, max_episode_steps=20)    
    waitkey = 10
    resize_resolution = (height, width)

    while True:
        i=0
        terminated = False
        truncated = False
        reward = 0
        obs, info = env.reset()
        while not terminated and not truncated:
            if render_mode == "rgb_array":
                wrist1 = obs["images"]["wrist1"].copy()
                wrist2 = obs["images"]["wrist2"].copy()
                front = obs["images"]["front"].copy()
                cv2.putText(wrist1, f"{reward:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,)
                cv2.imshow("wrist1", cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR))
                cv2.putText(wrist2, f"{reward:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,)
                cv2.imshow("wrist2", cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR))
                cv2.putText(front, f"{reward:.3f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,)
                cv2.imshow("front", cv2.cvtColor(front, cv2.COLOR_RGB2BGR))
                cv2.waitKey(waitkey)

            if i < 10:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.2])
            elif i < 37:
                action = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.2])
            elif i < 45:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
            elif i < 50:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
            elif i < 60:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
            elif i < 70:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])

            obs, reward, terminated, truncated, info = env.step(action)
            # print(f"i: {i}")
            # print(f"cartesian pos: {obs['state']['panda/tcp_pos']}")
            # print(f"cartesian ori: {obs['state']['panda/tcp_orientation']}")
            # print(f"cartesian vel: {obs['state']['panda/tcp_vel']}")
            # print(f"gripper pos: {obs['state']['panda/gripper_pos']}")
            # print(f"gripper vec: {obs['state']['panda/gripper_vec']}")
            i+=1
        
if __name__ == "__main__":
    main()
