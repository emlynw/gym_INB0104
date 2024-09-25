import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
from mujoco_ar import MujocoARConnector
import time
from scipy.spatial.transform import Rotation

def main():
    render_mode = "rgb_array"
    ee_dof = 4
    env = gym.make("gym_INB0104/ReachIKAbsEnv", render_mode=render_mode, randomize_domain=True, ee_dof=ee_dof)
    env = TimeLimit(env, max_episode_steps=200)    
    waitkey = 10
    connector = MujocoARConnector()

    # Start the connector
    connector.start()
    data = connector.get_latest_data()  # Returns {"position": (3, 1), "rotation": (3, 3), "button": bool, "toggle": bool}
    time.sleep(10)

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        action = np.array([0.0]*(ee_dof+1))
        rotate = True
        if render_mode == "rgb_array":
            pixels = obs["images"]["front"]
            cv2.resize(pixels, (224, 224))
            cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
            cv2.waitKey(waitkey)
        while not terminated and not truncated:
            action[0:3] = data["position"]
            action[1] = -3.0*action[1]
            action[2] = 4.0*action[2]
            r = Rotation.from_matrix(data["rotation"])
            angles = r.as_euler("xyz", degrees=False)
            action[3] = -angles[2]
            action[4] = float(data["button"])
            
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == "rgb_array":
                pixels = obs["images"]["front"]
                cv2.resize(pixels, (224, 224))
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
                cv2.waitKey(waitkey)
            i+=1
        
if __name__ == "__main__":
    main()
