# https://github.com/rail-berkeley/serl/blob/e2065d673131af6699aa899a78159859bd17c135/franka_sim/franka_sim/envs/panda_pick_gym_env.py
import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict
import mujoco
from gym_INB0104.controllers import opspace_4 as opspace
from typing import Optional, Any, SupportsFloat
from pathlib import Path


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    }

class cartesian_reach_ik_abs(MujocoEnv, utils.EzPickle):
    metadata = { 
        "render_modes": [ 
            "human",
            "rgb_array", 
            "depth_array"
        ], 
    }
    
    def __init__(
        self,
        image_obs=True,
        control_dt=0.1,
        physics_dt=0.002,
        width=480,
        height=480,
        render_mode="rgb_array",
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, 
            image_obs=image_obs,
            **kwargs
        )

        self.image_obs = image_obs
        self.render_mode = render_mode

        if self.image_obs:
            self.observation_space = Dict(
                {
                    "state": Dict(
                        {
                            "panda/tcp_pos": Box(np.array([0.2, -0.3, 0]), np.array([0.6, 0.3, 0.5]), shape=(3,), dtype=np.float32),
                            "panda/tcp_vel": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                            "panda/gripper_pos": Box(0.0, 0.08, shape=(1,), dtype=np.float32),
                        }
                    ),
                    "images": Dict(
                        {
                            "front": Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8),
                            "wrist": Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8,),
                        }
                    ),
                }
            )
        else:
            self.observation_space = Dict(
            {
                "state": Dict(
                    {
                        "panda/tcp_pos": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "panda/tcp_vel": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                        "panda/gripper_pos": Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                        "block_pos": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    }
                ),
            }
        )
            
        p = Path(__file__).parent
        env_dir = os.path.join(p, "xmls/cartesian_reach_ik.xml")
        self._n_substeps = int(control_dt / physics_dt)
        self.frame_skip = 1

        MujocoEnv.__init__(
            self, 
            env_dir, 
            self.frame_skip, 
            observation_space=self.observation_space, 
            render_mode=self.render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG, 
            camera_id=0, 
            **kwargs,
        )
        self.model.opt.timestep = physics_dt
        self.camera_id = (0, 1)
        self.action_space = Box(
            np.array([0.2, -0.3, 0, -1.0]),
            np.array([0.6, 0.3, 0.5, 1.0]),
            dtype=np.float32,
        )
        self._viewer = MujocoRenderer(
            self.model,
            self.data,
        )
        self._viewer.render(self.render_mode)
        self.setup()

    def setup(self):
        # self._PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
        self._PANDA_HOME = np.asarray((-0.00171672, -0.786471, -0.00122413, -2.36062, 0.00499334, 1.56444, 0.772088))
        self._GRIPPER_HOME = np.asarray([0.04, 0.04])
        self._PANDA_XYZ = np.asarray([0.3, 0, 0.5])
        self._CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
        self.default_obj_pos = np.array([0.5, 0])
        self.default_obs_quat = np.array([1, 0, 0, 0])
        self._panda_dof_ids = np.asarray(
            [self.model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self.model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id
        self._pinch_site_id = self.model.site("pinch").id
        self._block_z = self.model.geom("block").size[2]
        self.action_scale: np.ndarray = np.asarray([0.1, 1])
        
        # Arm and gripper to home position
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self.data.qpos[7:9] = self._GRIPPER_HOME
        mujoco.mj_forward(self.model, self.data)
        
        # Reset mocap body to home position
        tcp_pos = self.data.sensor("pinch_pos").data
        self.data.mocap_pos[0] = tcp_pos

        mujoco.mj_step(self.model, self.data)
        self.initial_qvel = np.copy(self.data.qvel)
        self.prev_time = 0.0
        self.prev_grasp = -1.0

        # Store initial values for randomization
        self.init_cam_pos = self.model.body_pos[self.model.body('front_cam').id].copy()
        self.init_cam_quat = self.model.body_quat[self.model.body('front_cam').id].copy()
        self.init_light_pos = self.model.body_pos[self.model.body('light0').id].copy()
        self.init_plywood_rgba = self.model.mat_rgba[self.model.mat('plywood').id].copy()
        self.init_brick_rgba = self.model.mat_rgba[self.model.mat('brick_wall').id].copy()
        self.table_tex_ids = [self.model.texture('plywood').id, self.model.texture('table').id]

        self.object_center_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "object_center_site")

    def domain_randomization(self):
        # Move robot
        ee_noise_x = np.random.uniform(low=0.0, high=0.12)
        ee_noise_y = np.random.uniform(low=-0.2, high=0.2)
        ee_noise_z = np.random.uniform(low=-0.4, high=0.1)
        ee_noise = np.array([ee_noise_x, ee_noise_y, ee_noise_z])
        self.data.mocap_pos[0] = self._PANDA_XYZ + ee_noise

        # Add noise to camera position and orientation
        cam_pos_noise = np.random.uniform(low=[-0.05,-0.05,-0.02], high=[0.05,0.05,0.02], size=3)
        cam_quat_noise = np.random.uniform(low=-0.02, high=0.02, size=4)
        self.model.body_pos[self.model.body('front_cam').id] = self.init_cam_pos + cam_pos_noise
        self.model.body_quat[self.model.body('front_cam').id] = self.init_cam_quat + cam_quat_noise
        # Add noise to light position
        light_pos_noise = np.random.uniform(low=[-0.8,-0.5,-0.05], high=[1.2,0.5,0.2], size=3)
        self.model.body_pos[self.model.body('light0').id] = self.init_light_pos + light_pos_noise
        # Change light levels
        light_0_diffuse_noise = np.random.uniform(low=0.1, high=0.8, size=1)
        self.model.light_diffuse[0][:] = light_0_diffuse_noise
        # Randomize table color
        channel = np.random.randint(0,3)
        table_color_noise = np.random.uniform(low=-0.05, high=0.2, size=1)
        self.model.mat_texid[self.model.mat('plywood').id] = np.random.choice(self.table_tex_ids)
        self.model.mat_rgba[self.model.mat('plywood').id] = self.init_plywood_rgba
        self.model.mat_rgba[self.model.mat('plywood').id][channel] = self.init_plywood_rgba[channel] + table_color_noise
        # Randomize brick color
        channel = np.random.randint(0,3)
        brick_color_noise = np.random.uniform(low=-0.1, high=0.1, size=1)
        self.model.mat_rgba[self.model.mat('brick_wall').id] = self.init_brick_rgba
        self.model.mat_rgba[self.model.mat('brick_wall').id][channel] = self.init_brick_rgba[channel] + brick_color_noise

        # Move object
        self.object_x_noise = np.random.uniform(low=-0.15, high=0.1)
        self.object_y_noise = np.random.uniform(low=-0.1, high=0.1)
        self.object_theta_noise = np.random.uniform(low=-0.5, high=0.5)
        self.data.qpos[9] = self.default_obj_pos[0] + self.object_x_noise
        self.data.qpos[10] = self.default_obj_pos[1] + self.object_y_noise
        self.data.qpos[12] = self.default_obs_quat[0] + self.object_theta_noise


    def reset_model(self):
        # Reset arm to home position.
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self.data.qpos[7:9] = self._GRIPPER_HOME
        mujoco.mj_forward(self.model, self.data)

        # Reset mocap body to home position.
        tcp_pos = self.data.sensor("pinch_pos").data
        self.data.mocap_pos[0] = tcp_pos

        self.domain_randomization()
        
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
        
        mujoco.mj_forward(self.model, self.data)
        for _ in range(5*self._n_substeps):
            tau = opspace(
                model=self.model,
                data=self.data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self.data.mocap_pos[0],
                ori=self.data.mocap_quat[0],
                joint=self._PANDA_HOME,
                gravity_comp=True,
            )
            self.data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self.model, self.data)
        
        self._z_init = self.data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.2
        self.prev_time = 0.0
        
        return self._get_obs()

    def step(self, action):
        # Action
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        x, y, z, grasp = action
        npos = np.asarray([x, y, z])
        npos = np.clip(npos, *self._CARTESIAN_BOUNDS)
        self.data.mocap_pos[0] = npos
        if self.data.time - self.prev_time < 0.5:
            grasp = self.prev_grasp
        else:
            grasp = grasp
            self.prev_time = self.data.time
            self.prev_grasp = grasp
        
        if grasp > 0:
            g = 0
        else:
            g = 255
        self.data.ctrl[self._gripper_ctrl_id] = g

        for _ in range(self._n_substeps):
            tau = opspace(
                model=self.model,
                data=self.data,
                site_id=self._pinch_site_id,
                dof_ids=self._panda_dof_ids,
                pos=self.data.mocap_pos[0],
                ori=self.data.mocap_quat[0],
                joint=self._PANDA_HOME,
                gravity_comp=True,
            )
            self.data.ctrl[self._panda_ctrl_ids] = tau
            mujoco.mj_step(self.model, self.data)

        # Observation
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        # Reward
        reward, info = self._get_reward(action)

        return obs, reward, False, False, info 
    
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    def _get_obs(self):
        obs = {}
        obs["state"] = {}

        tcp_pos = self.data.sensor("pinch_pos").data
        obs["state"]["panda/tcp_pos"] = tcp_pos.astype(np.float32)

        tcp_vel = self.data.sensor("pinch_vel").data
        obs["state"]["panda/tcp_vel"] = tcp_vel.astype(np.float32)

        gripper_pos = 2*np.array(self.data.qpos[8], dtype=np.float32)
        low = self.observation_space["state"]["panda/gripper_pos"].low
        high = self.observation_space["state"]["panda/gripper_pos"].high
        gripper_pos = np.clip(gripper_pos, low, high)
        obs["state"]["panda/gripper_pos"] = gripper_pos

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["wrist"], obs["images"]["front"] = self.render()
        else:
            block_pos = self.data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_pos"] = block_pos

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs
        
    def _get_reward(self, action):
        block_pos = self.data.sensor("block_pos").data
        tcp_pos = self.data.sensor("pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        r_close = np.exp(-20 * dist)
        if block_pos[2] > self._z_init + 0.15:
            success = True
        else:
            success = False
        r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
        r_lift = np.clip(r_lift, 0.0, 1.0)
        reward = 0.3 * r_close + 2.0 * r_lift
        info = dict(reward_close=r_close, reward_lift=r_lift, success=success)
        return reward, info

