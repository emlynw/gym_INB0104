import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
import pygame

np.set_printoptions(suppress=True)

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaStrawbHangingEnv", render_mode=render_mode, randomize_domain=False, ee_dof=6)
    env = TimeLimit(env, max_episode_steps=1000)
    resize_resolution = (480, 480)
    waitkey = 10

    # Initialize pygame for Xbox controller input
    pygame.init()
    pygame.joystick.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Using controller: {joystick.get_name()}")

    max_speed = 1.0  # Maximum speed in any direction
    rot_speed = 0.8  # Maximum rotational speed
    gripper_closed = False

    while True:
        terminated = False
        truncated = False
        obs, info = env.reset()
        i = 0

        # Dead zone threshold
        DEAD_ZONE = 0.15 # Adjust as needed

        def apply_dead_zone(value):
            return 0.0 if abs(value) < DEAD_ZONE else value

        while not terminated and not truncated:
            i+=1
            # Display the environment
            if render_mode == "rgb_array":
                pixels = obs["images"]["wrist2"]
                cv2.imshow("pixels", cv2.resize(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR), resize_resolution))
                cv2.imshow("front", cv2.resize(cv2.cvtColor(obs["images"]["front"], cv2.COLOR_RGB2BGR), resize_resolution))

            pygame.event.pump()  # Process events

            # Get joystick axes for movement and rotation
            left_stick_x = apply_dead_zone(joystick.get_axis(0)) # Left stick horizontal (left/right)
            left_stick_y = apply_dead_zone(joystick.get_axis(1)) # Left stick vertical (up/down)
            right_stick_x = apply_dead_zone(joystick.get_axis(3)) # Right stick horizontal
            right_stick_y = apply_dead_zone(joystick.get_axis(4)) # Right stick vertical
            trigger_l = apply_dead_zone(joystick.get_axis(2)) # Left trigger (gripper open/close)
            trigger_r = apply_dead_zone(joystick.get_axis(5)) # Right trigger (gripper close/open)

            # Compute actions
            move_forward_backward = -left_stick_y * max_speed  # Forward/backward
            move_left_right = left_stick_x * max_speed  # Left/right
            move_up_down = (trigger_r - trigger_l) * max_speed  # Up/down based on triggers

            # Check if left bumper (LB) is pressed
            is_yaw_mode = joystick.get_button(4)  # Left bumper (button 4)

            # Determine rotation inputs
            if is_yaw_mode:
                roll = 0.0  # Right stick horizontal controls roll
                yaw = right_stick_x * rot_speed  # Disable yaw while rolling
            else:
                roll = -right_stick_x * rot_speed
                yaw = 0.0 # Right stick horizontal controls yaw

            pitch = -right_stick_y * rot_speed  # Right stick vertical always controls pitch

            # Create the action array
            move_action = np.array([move_forward_backward, move_left_right, move_up_down, roll, pitch, yaw, 0.0])

            # Gripper control with A button (button 0) for toggling
            if joystick.get_button(0):  # A button toggles gripper
                gripper_closed = not gripper_closed
            move_action[-1] = 1.0 if gripper_closed else -0.2

            # Perform the action in the environment
            obs, reward, terminated, truncated, info = env.step(move_action)
            print(i)

            # Check for reset or exit
            key = cv2.waitKey(waitkey) & 0xFF
            if key == ord('r'):  # Reset the environment
                print("Resetting environment...")
                obs, info = env.reset()
                continue  # Restart the loop after reset
            if key == 27:  # ESC key to exit
                print("Exiting...")
                env.close()
                cv2.destroyAllWindows()
                pygame.quit()
                return

if __name__ == "__main__":
    main()
