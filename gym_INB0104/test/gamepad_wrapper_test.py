import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
from gamepad_wrapper import GamepadIntervention
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaStrawbHangingEnv", render_mode=render_mode, randomize_domain=True, ee_dof=6)
    env = SERLObsWrapper(env)
    env = GamepadIntervention(env)
    env = TimeLimit(env, max_episode_steps=10)    
    waitkey = 100
    cameras = ['wrist1', 'wrist2', 'front']
    resize_resolution = (480, 480)

    while True:
        # reset the environment
        i=0
        terminated = False
        truncated = False
        obs, info = env.reset()
        rotate = True
        
        while not terminated and not truncated:
            for camera in cameras:
                cv2.imshow(camera, cv2.resize(cv2.cvtColor(obs[camera], cv2.COLOR_RGB2BGR), resize_resolution))
                cv2.waitKey(waitkey)
    
            action = np.zeros_like(env.action_space.sample())
            print(f"action: {action}")
            if "intervene_action" in info:
                print(f"i action: {info['intervene_action']}")
                action = info['intervene_action']

            print(F"action: {action}")

            
            obs, reward, terminated, truncated, info = env.step(action)
            i+=1
        
if __name__ == "__main__":
    main()
