import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
np.set_printoptions(suppress=True)
import math
import time

from detection_wrapper import DetectionAreaWrapper

# Global variables to capture mouse movement
mouse_x, mouse_y = 0, 0  # Track mouse position

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def main():

    detect_cam = "wrist1"
    control_cam = "wrist1"
    rotate_180 = "True"

    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaStrawbHangingEnv", height=720, width=720, render_mode=render_mode, randomize_domain=False, ee_dof=4)
    env = TimeLimit(env, max_episode_steps=200)    
    env = DetectionAreaWrapper(env, detect_cam=detect_cam)
    waitkey = 100
    resize_resolution = (480, 480)

    # Define the range for absolute movement control
    max_speed = 0.5  # Maximum speed in any direction
    rot_speed = 0.8  # Maximum rotation speed

    # Set up mouse callback
    cv2.namedWindow(control_cam)
    cv2.setMouseCallback(control_cam, mouse_callback)
    
    while True:
        terminated = False
        truncated = False
        obs, info = env.reset()
        
        while not terminated and not truncated:
            step_start_time = time.time()
            detection = obs["detection"]  # (found, x_norm, y_norm, area)
            found, x_norm, y_norm, area_frac = detection
            print(f"detection: {detection}")

            if render_mode == "rgb_array":
                detect_img = obs['images'][detect_cam]
                H, W, _ = detect_img.shape

                # -----------------------------------------------------
                # 1) Overlay centroid + circle if detection found
                # -----------------------------------------------------
                if found == 1.0:
                    # Convert normalized coords [-1,1] -> pixel coords
                    center_x = int((x_norm + 1.0) * 0.5 * W)
                    center_y = int((y_norm + 1.0) * 0.5 * H)

                    # Convert area fraction to approximate circle radius
                    area_pixels = area_frac * (H * W)
                    radius = int(math.sqrt(area_pixels / math.pi))

                    # -----------------------------
                    # Convert detect_img to a valid OpenCV format
                    # -----------------------------
                    # 1) Make a copy so that you don't alter the original obs.
                    detect_img_draw = detect_img.copy()
                    
                    # 2) If detect_img is float32, convert to uint8
                    #    Make sure to clamp to [0,255] before casting.
                    if detect_img_draw.dtype != np.uint8:
                        detect_img_draw = np.clip(detect_img_draw, 0, 255).astype(np.uint8)
                    
                    # 3) Convert from RGB to BGR for OpenCV
                    detect_img_draw = cv2.cvtColor(detect_img_draw, cv2.COLOR_RGB2BGR)

                    # -----------------------------
                    # Draw the circle(s)
                    # -----------------------------
                    # Draw a circle whose radius approximates the detected mask area
                    cv2.circle(detect_img_draw, (center_x, center_y), radius, (0, 255, 0), 2)
                    
                    # Draw a small filled circle at the centroid
                    cv2.circle(detect_img_draw, (center_x, center_y), 5, (0, 0, 255), -1)

                    # Optionally add text showing the area fraction
                    text = f"Area={area_frac:.3f}"
                    cv2.putText(
                        detect_img_draw,
                        text,
                        (center_x + 10, center_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                    # Now display detect_img_draw
                    if rotate_180:
                        detect_img_draw = cv2.rotate(detect_img_draw, cv2.ROTATE_180)
                    cv2.imshow(detect_cam, cv2.resize(detect_img_draw, resize_resolution))

                else:
                    # If not found, just display the original image
                    # but still ensure it's a valid format for display
                    detect_img_draw = detect_img.copy()
                    if detect_img_draw.dtype != np.uint8:
                        detect_img_draw = np.clip(detect_img_draw, 0, 255).astype(np.uint8)
                    detect_img_draw = cv2.cvtColor(detect_img_draw, cv2.COLOR_RGB2BGR)
                    if rotate_180:
                        detect_img_draw = cv2.rotate(detect_img_draw, cv2.ROTATE_180)
                    cv2.imshow(detect_cam, cv2.resize(detect_img_draw, resize_resolution))
                                
                # Optionally show other cameras resized
                if detect_cam != control_cam:
                    cv2.imshow(control_cam, cv2.resize(cv2.cvtColor(obs["images"][control_cam], cv2.COLOR_RGB2BGR), resize_resolution))

                cv2.imshow("front", cv2.resize(cv2.cvtColor(obs["images"]["front"], cv2.COLOR_RGB2BGR), resize_resolution))

            
            # Calculate movement based on absolute mouse position within window
            move_left_right = ((mouse_x / resize_resolution[0]) * 2 - 1) * max_speed
            move_up_down = -((mouse_y / resize_resolution[1]) * 2 - 1) * max_speed

            # Define movement actions for W and S keys (forward/backward)
            key = cv2.waitKey(waitkey) & 0xFF
            move_action = np.array([0, move_left_right, move_up_down, 0.0, 0.0])  # Default move

            if key == ord('w'):
                move_action[0] = max_speed  # Forward
            elif key == ord('s'):
                move_action[0] = -max_speed   # Backward
            elif key == ord('a'):
                move_action[3] = rot_speed
            elif key == ord('d'):
                move_action[3] = -rot_speed

            # Toggle gripper state with spacebar
            if key == ord(' '):
                move_action[-1] = 1.0
            elif key == ord('c'):
                move_action[-1] = -1.0

            # Perform the action in the environment
            
            step_time = time.time()-step_start_time
            if step_time < waitkey/1000:
                time.sleep(waitkey/1000 - step_time)
            obs, reward, terminated, truncated, info = env.step(move_action)

            # Reset environment on 'R' key press
            if key == ord('r'):
                print("Resetting environment...")
                obs, info = env.reset()  # Reset the environment
                continue  # Start the loop again after reset

            # Exit on 'ESC' key
            if key == 27:  # ESC key
                print("Exiting...")
                env.close()
                cv2.destroyAllWindows()
                return

if __name__ == "__main__":
    main()
