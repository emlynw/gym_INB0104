import numpy as np

# First trajectory
traj1 = """[0.29862306 0.00008193 0.49772096]
[0.30466464 0.00855841 0.49660045]
[0.3146233  0.02220362 0.49415624]
[0.32585308 0.03556892 0.49177036]
[0.33700466 0.04786344 0.49058622]
[0.34843382 0.06079086 0.4887848 ]
[0.36001047 0.07383129 0.48703346]
[0.37137866 0.08705258 0.48487404]
[0.38281912 0.09886774 0.48399344]
[0.39459118 0.11244134 0.48210365]"""

# Second trajectory
traj2 = """[0.3157407  0.00722953 0.48264468]
[0.32537058 0.01719156 0.48073646]
[0.33413327 0.02619842 0.48072302]
[0.34317803 0.03556346 0.48011926]
[0.3521527  0.04479103 0.47968638]
[0.36112723 0.0540536  0.47917372]
[0.37006888 0.06328596 0.47866115]
[0.3789822  0.07250819 0.47811845]
[0.3878618  0.08170997 0.4775556 ]
[0.39670655 0.09089189 0.47696894]"""

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