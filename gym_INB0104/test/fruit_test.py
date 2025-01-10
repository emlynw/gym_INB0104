import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
np.set_printoptions(suppress=True)

import os
import time
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

# Global variables to capture mouse movement
mouse_x, mouse_y = 0, 0  # Track mouse position

# Configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.DEVICE = "cuda"  # Use GPU if available

# Path to weights
cfg.MODEL.WEIGHTS = "/home/emlyn/Downloads/aoc_model.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Confidence threshold for predictions

# Initialize predictor
predictor = DefaultPredictor(cfg)

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaStrawbHangingEnv", height=720, width=720, render_mode=render_mode, randomize_domain=False, ee_dof=4)
    env = TimeLimit(env, max_episode_steps=200)    
    waitkey = 1
    resize_resolution = (480, 480)

    # Define the range for absolute movement control
    max_speed = 0.5  # Maximum speed in any direction
    rot_speed = 0.8  # Maximum rotation speed

    # Set up mouse callback
    cv2.namedWindow("wrist2")
    cv2.setMouseCallback("wrist2", mouse_callback)
    
    while True:
        terminated = False
        truncated = False
        obs, info = env.reset()
        
        while not terminated and not truncated:
            # Display the environment
            if render_mode == "rgb_array":
                wrist2 = obs['images']['wrist2']
                cv2.imshow("wrist2", cv2.cvtColor(wrist2, cv2.COLOR_RGB2BGR))
                # cv2.imshow("wrist2", cv2.resize(cv2.cvtColor(obs['images']['wrist2'], cv2.COLOR_RGB2BGR), resize_resolution))
                # Rotate wrist1 by 180 degrees
                # wrist1 = cv2.rotate(obs['images']['wrist1'], cv2.ROTATE_180)
                # cv2.imshow("wrist1", cv2.resize(cv2.cvtColor(wrist1, cv2.COLOR_RGB2BGR), resize_resolution))
                # cv2.imshow("front", cv2.resize(cv2.cvtColor(obs["images"]["front"], cv2.COLOR_RGB2BGR), resize_resolution))
            
            detection_time = time.time()
            outputs = predictor(wrist2)
            predictions = outputs["instances"].to("cpu")
            print("Detection time:", time.time() - detection_time)

            # Visualize predictions
            visualizer = Visualizer(wrist2, metadata=None, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
            vis_output = visualizer.draw_instance_predictions(predictions)

            # Convert visualized image to BGR format for OpenCV display
            predicted_image = cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)

            # Show the image
            cv2.imshow("Predictions", predicted_image)
            cv2.waitKey(1)
            
            # Calculate movement based on absolute mouse position within window
            move_left_right = ((mouse_x / resize_resolution[0]) * 2 - 1) * max_speed
            move_up_down = -((mouse_y / resize_resolution[1]) * 2 - 1) * max_speed

            # Define movement actions for W and S keys (forward/backward)
            key = cv2.waitKey(waitkey) & 0xFF
            move_action = np.array([0, move_left_right, move_up_down, 0.0, 1.0])  # Default move

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
