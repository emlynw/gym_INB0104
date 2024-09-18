### Franka RL
Mujoco simulation of Franka B in INB0104 for use in sim to real. 
Uses SERL impedence control
Also taken from https://github.com/zichunxx/panda_mujoco_gym/blob/master/panda_mujoco_gym/envs/panda_env.py

### Installation:

Requires: mujoco, gymnasium

clone the repository
cd /gym_INB0104
pip install -e .

### Running the current simulation:
python /rl_franka/gym_INB0104/test/cartesian_velocity_test.py
python /rl_franka/gym_INB0104/test/joint_velocity_test.py

### editing and adding to the simulation:
- The gymnasium simulation environments are found in /gym_INB0104/gym_INB0104/envs
- The mujoco xml files are in /gym_INB0104/gym_INB0104/environments/envs/xmls


