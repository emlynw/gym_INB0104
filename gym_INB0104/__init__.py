from gymnasium.envs.registration import register

register( id="gym_INB0104/cartesian_push-v0", entry_point="gym_INB0104.envs:cartesian_push" , max_episode_steps=1000)
register( id="gym_INB0104/cartesian_reach-v0", entry_point="gym_INB0104.envs:cartesian_reach" , max_episode_steps=1000)
register( id="gym_INB0104/cartesian_reach_ik-v0", entry_point="gym_INB0104.envs:cartesian_reach_ik" , max_episode_steps=1000)
register( id="gym_INB0104/cartesian_reach_ik_abs-v0", entry_point="gym_INB0104.envs:cartesian_reach_ik_abs" , max_episode_steps=1000)
register( id="gym_INB0104/joint_velocity_push", entry_point="gym_INB0104.envs:joint_velocity_push" , max_episode_steps=1000)
