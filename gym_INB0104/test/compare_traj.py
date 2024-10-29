import numpy as np

# First trajectory
traj1 = """[0.29954633 0.00020706 0.49759108]
[0.31709594 0.00016782 0.49364844]
[0.33010858 0.00012812 0.49204007]
[0.34383795 0.00008133 0.48965633]
[0.35762623 0.00003437 0.48738807]
"""

# Second trajectory
traj2 = """[0.31380433 0.00006014 0.48540428]
[0.3292175  0.00008267 0.48376387]
[0.34604576 0.00010493 0.48321837]
[0.3638434  0.00010881 0.4824228 ]
[0.38205257 0.00011072 0.48163006]"""

def parse_trajectory(traj_str):
    """Parse trajectory string into numpy array."""
    lines = [line.strip() for line in traj_str.strip().split('\n') if line.strip()]
    arrays = []
    for line in lines:
        # Remove brackets and convert to floats
        clean_line = line.replace('[', '').replace(']', '').strip()
        numbers = [float(x) for x in clean_line.split()]
        arrays.append(numbers)
    return np.array(arrays)

# Parse trajectories
traj1_array = parse_trajectory(traj1)
traj2_array = parse_trajectory(traj2)

# Calculate differences
differences = {
    'absolute_diff': np.abs(traj1_array - traj2_array),
    'mean_diff': np.mean(np.abs(traj1_array - traj2_array), axis=0),
    'max_diff': np.max(np.abs(traj1_array - traj2_array), axis=0),
    'rmse': np.sqrt(np.mean((traj1_array - traj2_array)**2, axis=0)),
    'euclidean_distance': np.sqrt(np.sum((traj1_array - traj2_array)**2, axis=1))
}

# Print results
print("Trajectory Comparison Results:")
print("-" * 50)

print("\nMean absolute difference per dimension:")
print(f"X: {differences['mean_diff'][0]:.6f}")
print(f"Y: {differences['mean_diff'][1]:.6f}")
print(f"Z: {differences['mean_diff'][2]:.6f}")

print("\nMaximum difference per dimension:")
print(f"X: {differences['max_diff'][0]:.6f}")
print(f"Y: {differences['max_diff'][1]:.6f}")
print(f"Z: {differences['max_diff'][2]:.6f}")

print("\nRMSE per dimension:")
print(f"X: {differences['rmse'][0]:.6f}")
print(f"Y: {differences['rmse'][1]:.6f}")
print(f"Z: {differences['rmse'][2]:.6f}")

print("\nEuclidean distance at each timestep:")
for i, dist in enumerate(differences['euclidean_distance']):
    print(f"Step {i}: {dist:.6f}")

# Calculate overall average difference
overall_mean_diff = np.mean(differences['mean_diff'])
print(f"\nOverall average difference: {overall_mean_diff:.6f}")

# Calculate total path length difference
def path_length(traj):
    diffs = np.diff(traj, axis=0)
    distances = np.sqrt(np.sum(diffs**2, axis=1))
    return np.sum(distances)

path_length1 = path_length(traj1_array)
path_length2 = path_length(traj2_array)
print(f"\nPath length trajectory 1: {path_length1:.6f}")
print(f"Path length trajectory 2: {path_length2:.6f}")
print(f"Path length difference: {abs(path_length1 - path_length2):.6f}")