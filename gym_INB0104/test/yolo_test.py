import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
from gym_INB0104 import envs
import numpy as np
from ultralytics import YOLO

def main():
    render_mode = "rgb_array"
    env = gym.make("gym_INB0104/ReachIKDeltaEnv", render_mode=render_mode, randomize_domain=True, ee_dof=4)
    env = TimeLimit(env, max_episode_steps=100)    
    waitkey = 10
    model = YOLO("yolov8n.pt", verbose=False)
    print(model.names)

    while True:
        # reset the environment
        i = 0
        terminated = False
        truncated = False
        obs, info = env.reset()
        pixels = obs["images"]["front"]
        pixels = cv2.resize(pixels, (640, 640))
        results = model(pixels, verbose=False)  # Run inference

        if render_mode == "rgb_array":
            pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

            # Extract results from YOLOv8 output
            boxes = results[0].boxes  # YOLOv8 stores the results in a list, so access the first element
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
                confidence = box.conf[0]      # Confidence score
                cls = box.cls[0]              # Class label index
                
                label = f'{model.names[int(cls)]} {confidence:.2f}'

                # Draw rectangle for the bounding box
                cv2.rectangle(pixels_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
                
                # Put label text above the bounding box
                cv2.putText(pixels_bgr, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("pixels", cv2.resize(pixels_bgr, (720, 720)))
            cv2.waitKey(waitkey)

        while not terminated and not truncated:
            if i < 15:
                action = np.array([0.2, 0.0, 0.0, 0.0, -1.0])
            elif i < 50:
                action = np.array([0.0, 0.0, -1.0, 0.0, -1.0])
            elif i < 80:
                action = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
            elif i < 100:
                action = np.array([0.0, 0.0, 0.2, 0.0, 1.0])
            elif i < 120:
                action = np.array([0.0, 0.0, 0.0, -1.0, 1.0])
            elif i < 140:
                action = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
            else:
                action = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
            
            # # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            pixels = obs["images"]["front"]
            pixels = cv2.resize(pixels, (640, 640))
            results = model(pixels, verbose=False)
            boxes = results[0].boxes  # Access the boxes again for each step

            if render_mode == "rgb_array":
                pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

                # Draw bounding boxes again
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    cls = box.cls[0]
                    
                    label = f'{model.names[int(cls)]} {confidence:.2f}'
                    
                    # Draw rectangle for the bounding box
                    cv2.rectangle(pixels_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Put label text
                    cv2.putText(pixels_bgr, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Show the image with bounding boxes
                cv2.imshow("pixels", cv2.resize(pixels_bgr, (720, 720)))
                cv2.waitKey(waitkey)

            i += 1

if __name__ == "__main__":
    main()
