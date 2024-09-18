from gymnasium.envs.registration import register

register( id="gym_INB0104/reach_delta-v0", entry_point="gym_INB0104.envs:reach_delta" , max_episode_steps=1000)
register( id="gym_INB0104/ReachIKDeltaEnv-v0", entry_point="gym_INB0104.envs:ReachIKDeltaEnv" , max_episode_steps=1000)
register( id="gym_INB0104/ReachIKAbsEnv-v0", entry_point="gym_INB0104.envs:ReachIKAbsEnv" , max_episode_steps=1000)
register( id="gym_INB0104/push_ik_abs-v0", entry_point="gym_INB0104.envs:push_ik_abs" , max_episode_steps=1000)
