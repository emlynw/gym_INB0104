import gymnasium as gym
from gymnasium.wrappers import PixelObservationWrapper, TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np

def main():
    env = gym.make("gym_INB0104/cartesian_reach", render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=200)    

    # reset the environment
    obs, info = env.reset()
    pixels = env.render()
    cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    cv2.waitKey(10)
    i=0

    while True:
        terminated = False
        truncated = False
        i=0
        while not terminated and not truncated:
            # action = np.zeros(len(env.action_space.sample()))
            if i < 50:
                action = np.array([0.1, -0.1, 0.1, 0])
            elif i >= 50 and i < 100:
                action = np.array([0.5, 0.1, 0.1, 0])
            elif i >= 100 and i < 150:
                action = np.array([0.5, 0.1, 0.1, 0])
            elif i >= 150 and i < 200:
                action = np.array([0.5, -0.1, 0.1, 0])
            obs, reward, terminated, truncated, info = env.step(action)
            pixels = env.render()
            # cv2.putText(
            #     pixels,
            #     f"{reward:.3f}",
            #     (10, 50 - 10),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.7,
            #     (0, 255, 0),
            #     2,
            #     cv2.LINE_AA,
            # )
            # resize
            cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)
            i+=1
        obs, info = env.reset()
        pixels = env.render()
        cv2.imshow("pixels", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
