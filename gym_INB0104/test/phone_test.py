from mujoco_ar import MujocoARConnector
import time
from scipy.spatial.transform import Rotation

# Initialize the connector with your desired parameters
connector = MujocoARConnector()

# Start the connector
connector.start()
data = connector.get_latest_data()  # Returns {"position": (3, 1), "rotation": (3, 3), "button": bool, "toggle": bool}


time.sleep(10)

while True:
    # Retrieve the latest AR data (after connecting the iOS device, see the guide below)
    data = connector.get_latest_data()  # Returns {"position": (3, 1), "rotation": (3, 3), "button": bool, "toggle": bool}
    r = Rotation.from_matrix(data["rotation"])
    angles = r.as_euler("xyz", degrees=False)
    print(f"angles: {angles}")
    # print(data)

