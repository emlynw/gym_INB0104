import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
from mujoco_ar import MujocoARConnector
import time
from scipy.spatial.transform import Rotation

def render(obs, waitkey):
    pixels = obs["images"]["front"]
    cv2.resize(pixels, (224, 224))
    cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), (720, 720)))
    cv2.waitKey(waitkey)

def main():
    render_mode = "rgb_array"
    ee_dof = 4
    env = gym.make("gym_INB0104/ReachIKDeltaEnv", render_mode=render_mode, randomize_domain=True, ee_dof=ee_dof)
    env = TimeLimit(env, max_episode_steps=200)    
    waitkey = 10
    connector = MujocoARConnector()

    # Start the connector
    connector.start()
    data = connector.get_latest_data()  # Returns {"position": (3, 1), "rotation": (3, 3), "button": bool, "toggle": bool}
    while data['position'] is None:
        data = connector.get_latest_data()
        time.sleep(1)
        print("Waiting for AR data...")

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        connector.reset_position()
        action = np.array([0.0]*(ee_dof+1))
        if render_mode == "rgb_array":
            render(obs, waitkey)

        while not terminated and not truncated:
            pos = data["position"]
            pos[0] = -4*pos[0]
            pos[1] = -pos[1]
            pos[2] = 4*pos[2]
            rot = []
            grasp = [float(data["button"])]
            if ee_dof == 4:
                r = Rotation.from_matrix(data["rotation"])
                angles = r.as_euler("xyz", degrees=False)
                rot = [-angles[2]]
            elif ee_dof == 6:
                r = Rotation.from_matrix(data["rotation"])
                angles = r.as_euler("xyz", degrees=False)
                rot = angles
                rot[2] = -rot[2]
            
            action = np.concatenate((pos, rot, grasp))
                        
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == "rgb_array":
                render(obs, waitkey)
            i+=1
        
if __name__ == "__main__":
    main()
