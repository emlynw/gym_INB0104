import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
from gym_INB0104.envs import mujoco_utils
from typing import Optional, Any, SupportsFloat
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    }


class cartesian_push(MujocoEnv, utils.EzPickle):
    metadata = { 
        "render_modes": [ 
            "human",
            "rgb_array", 
            "depth_array"
        ], 
        "render_fps": 10
    }
    
    def __init__(self, render_mode=None, use_distance=False, **kwargs):
        utils.EzPickle.__init__(self, use_distance, **kwargs)
        self.use_distance = use_distance
        observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)
        p = Path(__file__).parent
        env_dir = os.path.join(p, "xmls/cartesian_push.xml")
        self.frame_skip = 50
        MujocoEnv.__init__(self, env_dir, self.frame_skip, observation_space=observation_space, default_camera_config=DEFAULT_CAMERA_CONFIG, camera_id=0, **kwargs,)
        self.render_mode = render_mode
        self._utils = mujoco_utils
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.04, 0.04])
        self.action_space = Box(
            np.array([-0.2, -0.2, -0.2, 0.0]),
            np.array([0.2, 0.2, 0.2, 255]),
            dtype=np.float32,
        )
        self.ee_low = np.array([0.2, -0.15, 0.93])
        self.ee_high = np.array([0.75, 0.15, 0.98])
        self.ep_steps = 0
        self.setup()

    def setup(self):
        self.set_joint_neutral()
        self.data.ctrl[0:7] = self.neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = np.array([0, 1, 0, 0])
        # self.grasp_site_pose = self.get_ee_orientation().copy()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        
        self._mujoco_step()
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

        self.cam_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cam0")
        self.init_cam_pos = self.model.body_pos[self.cam_body_id].copy()
        self.init_cam_quat = self.model.body_quat[self.cam_body_id].copy()

        self.light_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "light0")
        self.init_light_pos = self.model.body_pos[self.light_body_id].copy()

        self.plywood_tex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "plywood")
        self.table_tex_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "table")
        self.plywood_mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "plywood")
        self.init_plywood_rgba = self.model.mat_rgba[self.plywood_mat_id].copy()

        self.brick_mat_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "brick_wall")
        self.init_brick_rgba = self.model.mat_rgba[self.brick_mat_id].copy()

        self.left_curtain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_curtain")
        self.init_left_curtain_rgba = self.model.geom_rgba[self.left_curtain_geom_id].copy()
        self.right_curtain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_curtain")
        self.init_right_curtain_rgba = self.model.geom_rgba[self.right_curtain_geom_id].copy()
        self.back_curtain_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "back_curtain")
        self.init_back_curtain_rgba = self.model.geom_rgba[self.back_curtain_geom_id].copy()

        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.default_obj_pos = np.array([0.5, 0, 1.1])
        self.default_obs_quat = np.array([1, 0, 0, 0])

        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.init_target_site_pos = self._utils.get_site_xpos(self.model, self.data, "target_site").copy()
        self.object_center_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object_center_site")

        

    def reset_model(self):
        # Add noise to camera position and orientation
        cam_pos_noise = np.random.uniform(low=[-0.05,-0.05,-0.02], high=[0.05,0.05,0.02], size=3)
        cam_quat_noise = np.random.uniform(low=-0.01, high=0.01, size=4)
        self.model.body_pos[self.cam_body_id] = self.init_cam_pos + cam_pos_noise
        self.model.body_quat[self.cam_body_id] = self.init_cam_quat + cam_quat_noise
        # Add noise to light position
        light_pos_noise = np.random.uniform(low=[-0.8,-0.5,-0.2], high=[1.2,0.5,0.2], size=3)
        self.model.body_pos[self.light_body_id] = self.init_light_pos + light_pos_noise
        # Randomize table color
        channel = np.random.randint(0,3)
        table_color_noise = np.random.uniform(low=-0.05, high=0.2, size=1)
        self.model.mat_texid[self.plywood_mat_id] = np.random.choice([self.plywood_tex_id, self.table_tex_id])
        self.model.mat_rgba[self.plywood_mat_id] = self.init_plywood_rgba
        self.model.mat_rgba[self.plywood_mat_id][channel] = self.init_plywood_rgba[channel] + table_color_noise
        # Randomize brick color
        channel = np.random.randint(0,3)
        brick_color_noise = np.random.uniform(low=-0.1, high=0.1, size=1)
        self.model.mat_rgba[self.brick_mat_id] = self.init_brick_rgba
        self.model.mat_rgba[self.brick_mat_id][channel] = self.init_brick_rgba[channel] + brick_color_noise
        # Randomize curtain alpha
        alpha = np.random.choice([0.0, 1.0, np.random.uniform(low=0.1, high=0.9)])
        self.model.geom_rgba[self.left_curtain_geom_id][3] = alpha
        self.model.geom_rgba[self.right_curtain_geom_id][3] = alpha
        self.model.geom_rgba[self.back_curtain_geom_id][3] = alpha
        # Move target site
        self.goal_x_noise = np.random.uniform(low=0.00, high=0.08)
        self.goal_y_noise = np.random.uniform(low=-0.05, high=0.05)
        self.model.site_pos[self.target_site_id] = self.init_target_site_pos + [self.goal_x_noise, self.goal_y_noise, 0]

        # Move object
        self.object_x_noise = np.random.uniform(low=-0.15, high=0.1)
        self.object_y_noise = np.random.uniform(low=-0.1, high=0.1)
        self.object_theta_noise = np.random.uniform(low=-0.5, high=0.5)
        self.data.qpos[9] = self.default_obj_pos[0] + self.object_x_noise
        self.data.qpos[10] = self.default_obj_pos[1] + self.object_y_noise
        self.data.qpos[12] = self.default_obs_quat[0] + self.object_theta_noise
        
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
        self.set_joint_neutral()

        # Move robot
        ee_noise_x = np.random.uniform(low=0.0, high=0.10)
        ee_noise_y = np.random.uniform(low=-0.12, high=0.12)
        ee_noise = np.array([ee_noise_x, ee_noise_y, 0])
        self.initial_mocap_position = [0.25, 0.0, 0.93]
        self.set_mocap_pose(self.initial_mocap_position+ee_noise, self.grasp_site_pose)

        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_obs()

    def step(self, action):
        # Action
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._mujoco_step()

        # Observation
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        # Reward
        ee_pos = self.get_body_com("ee_center_body").copy()
        target_pos = self.data.site_xpos[self.object_center_site_id].copy()
        target_site = self._utils.get_site_xpos(self.model, self.data, "target_site").copy()
        dist_1 = np.linalg.norm(ee_pos - target_pos)
        dist_2 = np.linalg.norm(target_pos - target_site)
        ctrl = np.square(action[0:3]).sum()
        if dist_1 < 0.04:
            r_1 = 1.0
        else:
            r_1 = -dist_1
        if dist_2 < 0.04:
            r_2 = 10.0
        else:
            r_2 = -dist_2
        reward = r_1 + r_2 -ctrl
        info = dict(ee_to_obj=dist_1, obj_to_target=dist_2,  reward_ctrl=ctrl) 

        return obs, reward, False, False, info 
    
    def _set_action(self, action):
        action = action.copy()
        vel = action[:3]
        dt = 0.1
        gripper_ctrl = action[3]
        # Control gripper
        self.data.ctrl[-1] = gripper_ctrl
        # Change ee position
        new_pos = self.get_ee_position() + vel*dt
        new_pos = np.clip(new_pos, self.ee_low, self.ee_high)
        self.set_mocap_pose(new_pos, self.grasp_site_pose)

    def _get_obs(self):
        pos = self.get_body_com("ee_center_body").copy() - self.initial_mocap_position.copy()
        gripper_width = self.get_fingers_width()
        if self.use_distance:
            target_pos = self.get_body_com("target_object")
            distance = np.linalg.norm(target_pos - pos)
            return np.concatenate([pos, gripper_width, [distance]])
        else:
            return np.concatenate([pos, gripper_width])
        
    # Utils copied from https://github.com/zichunxx/panda_mujoco_gym/blob/master/panda_mujoco_gym/envs/panda_env.py
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == 1:
                    # relative pose
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(model, data)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        self.data.qpos[0:9] = self.neutral_joint_values

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat
    
    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2
