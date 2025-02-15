import numpy as np
from scipy.spatial.transform import Rotation as R

# The provided o_t_ee values:
o_t_ee = np.array([
    0.08505778744486983,
    -0.0027123899205034707,
    0.9963626662200422,
    0.0,
    -0.005190086727178214,
    -0.9999743073512806,
    -0.0022791529784795847,
    0.0,
    0.9963624320399641,
    -0.004977444771042581,
    -0.08507131751042181,
    0.0,
    0.11562490827775448,
    -0.0008721468083857587,
    0.7989168810237183,
    1.0
])

# Reshape the flat list into a 4x4 matrix.
# Use order='F' for column-major. If your data is row-major, use order='C'.
T = o_t_ee.reshape((4, 4), order='F')

# Extract the 3x3 rotation matrix.
R_mat = T[:3, :3]

# Convert the rotation matrix to a quaternion.
r = R.from_matrix(R_mat)
quat = r.as_quat()  # returns quaternion in [x, y, z, w] order
euler = r.as_euler('xyz', degrees=True)

print("Quaternion (xyzw):", quat)
print(f"euler: {euler}")