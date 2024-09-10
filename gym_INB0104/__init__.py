from gymnasium.envs.registration import register

register( id="gym_INB0104/cartesian_push-v0", entry_point="gym_INB0104.envs:cartesian_push" , max_episode_steps=1000)
register( id="gym_INB0104/cartesian_reach-v0", entry_point="gym_INB0104.envs:cartesian_reach" , max_episode_steps=1000)
register( id="gym_INB0104/reach_delta-v0", entry_point="gym_INB0104.envs:reach_delta" , max_episode_steps=1000)
register( id="gym_INB0104/reach_ik_delta-v0", entry_point="gym_INB0104.envs:reach_ik_delta" , max_episode_steps=1000)
register( id="gym_INB0104/reach_ik_abs-v0", entry_point="gym_INB0104.envs:reach_ik_abs" , max_episode_steps=1000)
register( id="gym_INB0104/push_ik_abs-v0", entry_point="gym_INB0104.envs:push_ik_abs" , max_episode_steps=1000)
register( id="gym_INB0104/joint_velocity_push", entry_point="gym_INB0104.envs:joint_velocity_push" , max_episode_steps=1000)
