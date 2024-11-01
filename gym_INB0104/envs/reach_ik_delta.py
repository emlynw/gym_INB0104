# https://github.com/rail-berkeley/serl/blob/e2065d673131af6699aa899a78159859bd17c135/franka_sim/franka_sim/envs/panda_pick_gym_env.py
import numpy as np
import os
import mujoco

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict
from gym_INB0104.controllers import opspace_4 as opspace
from pathlib import Path
from scipy.spatial.transform import Rotation


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    }

class ReachIKDeltaEnv(MujocoEnv, utils.EzPickle):
    metadata = { 
        "render_modes": ["human", "rgb_array", "depth_array"], 
    }
    
    def __init__(
        self,
        image_obs=True,
        randomize_domain=True,
        ee_dof = 6, # 3 for position, 3 for orientation
        control_dt=0.1,
        physics_dt=0.002,
        width=640,
        height=480,
        pos_scale=0.1,
        rot_scale=0.05,
        render_mode="rgb_array",
        **kwargs,
    ):
        utils.EzPickle.__init__(self, image_obs=image_obs, **kwargs)
        self.image_obs = image_obs
        self.randomize_domain = randomize_domain
        self.ee_dof = ee_dof
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale

        state_space = Dict(
            {
                "panda/tcp_pos": Box(np.array([0.28, -0.5, 0.01]), np.array([0.75, 0.5, 0.55]), shape=(3,), dtype=np.float32),
                "panda/tcp_orientation": Box(-1, 1, shape=(4,), dtype=np.float32),  # Quaternion
                "panda/tcp_vel": Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "panda/gripper_pos": Box(-1, 1, shape=(1,), dtype=np.float32),
                "panda/gripper_vec": Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )
        if not image_obs:
            state_space["block_pos"] = Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = Dict({"state": state_space})
        if image_obs:
            self.observation_space["images"] = Dict(
                {
                    "wrist": Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
                    "front": Box(0, 255, shape=(self.height, self.width, 3), dtype=np.uint8),
                }
            )

        p = Path(__file__).parent
        env_dir = os.path.join(p, "xmls/reach_ik.xml")
        self._n_substeps = int(float(control_dt) / float(physics_dt))
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
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32,
        )
        self._viewer = MujocoRenderer(self.model, self.data,)
        self._viewer.render(self.render_mode)
        self.setup()

    def setup(self):
        # self._PANDA_HOME = np.asarray((0, -0.785, 0, -2.35, 0, 1.57, np.pi / 4))
        self._PANDA_HOME = np.array([-0.00171672, -0.786471, -0.00122413, -2.36062, 0.00499334, 1.56444, 0.772088], dtype=np.float32)
        self._GRIPPER_HOME = np.array([0.04, 0.04], dtype=np.float32)
        self._PANDA_XYZ = np.array([0.3, 0, 0.5], dtype=np.float32)
        self.center_pos = np.array([0.3, 0, 0.2], dtype=np.float32)
        self._CARTESIAN_BOUNDS = np.array([[0.28, -0.35, 0.01], [0.75, 0.35, 0.55]], dtype=np.float32)
        self._ROTATION_BOUNDS= np.array([[-np.pi/4, -np.pi/4, -np.pi/2], [np.pi/4, np.pi/4, np.pi/2]], dtype=np.float32)

        self.default_obj_pos = np.array([0.5, 0])
        self.default_obs_quat = np.array([1, 0, 0, 0])
        self._panda_dof_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id
        self._pinch_site_id = self.model.site("pinch").id
        self._block_z = self.model.geom("block").size[2]
        
        self.reset_arm_and_gripper()

        self.prev_action = np.zeros(self.action_space.shape)
        self.prev_grasp_time = 0.0
        self.prev_grasp = -1.0
        self.gripper_dict = {
            "open": np.array([1, 0, 0, 0], dtype=np.float32),
            "closed": np.array([0, 1, 0, 0], dtype=np.float32),
            "opening": np.array([0, 0, 1, 0], dtype=np.float32),
            "closing": np.array([0, 0, 0, 1], dtype=np.float32),
        }

        # Store initial values for randomization
        self.front_cam_pos = self.model.body_pos[self.model.body('front_cam').id].copy()
        self.front_cam_quat = self.model.body_quat[self.model.body('front_cam').id].copy()
        self.wrist_cam_pos = self.model.body_pos[self.model.body('wrist_cam1').id].copy()
        self.wrist_cam_quat = self.model.body_quat[self.model.body('wrist_cam1').id].copy()
        self.init_light_pos = self.model.body_pos[self.model.body('light0').id].copy()
        self.init_plywood_rgba = self.model.mat_rgba[self.model.mat('plywood').id].copy()
        self.init_brick_rgba = self.model.mat_rgba[self.model.mat('brick_wall').id].copy()
        self.table_tex_ids = [self.model.texture('plywood').id, self.model.texture('table').id]

        # Add this line to set the initial orientation
        self.initial_orientation = [0, 1, 0, 0]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)

    def domain_randomization(self):
        # Move robot
        ee_noise = np.random.uniform(low=[0.0,-0.2,-0.4], high=[0.12, 0.2, 0.1], size=3)
        self.data.mocap_pos[0] = self._PANDA_XYZ + ee_noise
        # Add noise to camera position and orientation
        front_cam_pos_noise = np.random.uniform(low=[-0.05,-0.05,-0.02], high=[0.05,0.05,0.02], size=3)
        front_cam_quat_noise = np.random.uniform(low=-0.02, high=0.02, size=4)
        self.model.body_pos[self.model.body('front_cam').id] = self.front_cam_pos + front_cam_pos_noise
        self.model.body_quat[self.model.body('front_cam').id] = self.front_cam_quat + front_cam_quat_noise
        wrist_cam_pos_noise = np.random.uniform(low=[-0.05,-0.05,-0.02], high=[0.05,0.05,0.02], size=3)
        wrist_cam_quat_noise = np.random.uniform(low=-0.02, high=0.02, size=4)
        self.model.body_pos[self.model.body('wrist_cam1').id] = self.wrist_cam_pos + wrist_cam_pos_noise
        self.model.body_quat[self.model.body('wrist_cam1').id] = self.wrist_cam_quat + wrist_cam_quat_noise
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

    def reset_arm_and_gripper(self):
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self.data.qpos[7:9] = self._GRIPPER_HOME
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.data.sensor("pinch_pos").data.copy()
        mujoco.mj_step(self.model, self.data)


    def reset_model(self):
        self.reset_arm_and_gripper()
        if self.randomize_domain:
            self.domain_randomization()
        
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
        
        self._block_init = self.data.sensor("block_pos").data
        self._z_init = self._block_init[2]
        self._z_success = self._z_init + 0.2
        self._block_success = self._block_init.copy()
        self._block_success[2] = self._z_success

        self.gripper_vec = self.gripper_dict["open"]
        self.data.ctrl[self._gripper_ctrl_id] = 255
        self.prev_grasp_time = 0.0
        self.prev_gripper_state = 0 # 0 for open, 1 for closed
        self.gripper_state = 0
        self.gripper_blocked = False

        return self._get_obs()

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Scale actions
        if self.ee_dof == 3:
            x, y, z, grasp = action
        elif self.ee_dof == 4:
            x, y, z, yaw, grasp = action
            roll, pitch = 0, 0
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        elif self.ee_dof == 6:
            x, y, z, roll, pitch, yaw, grasp = action
            drot = np.array([roll, pitch, yaw]) * self.rot_scale
        dpos = np.array([x, y, z]) * self.pos_scale
        
        # Apply position change
        pos = self.data.sensor("pinch_pos").data
        npos = np.clip(pos + dpos, *self._CARTESIAN_BOUNDS)
        self.data.mocap_pos[0] = npos

        if self.ee_dof > 3:
            # Orientation changes, ZYX because of mujoco quaternions?
            current_quat = self.data.sensor("pinch_quat").data
            current_rotation = Rotation.from_quat(current_quat)
            # Convert the action rotation to a Rotation object
            action_rotation = Rotation.from_euler('zyx', drot)
            # Apply the action rotation
            new_rotation = action_rotation * current_rotation
            # Calculate the new relative rotation
            new_relative_rotation = self.initial_rotation.inv() * new_rotation
            # Convert to euler angles and clip
            relative_euler = new_relative_rotation.as_euler('zyx')
            clipped_euler = np.clip(relative_euler, self._ROTATION_BOUNDS[0], self._ROTATION_BOUNDS[1])
            # Convert back to rotation and apply to initial orientation
            clipped_rotation = Rotation.from_euler('zyx', clipped_euler)
            final_rotation = self.initial_rotation * clipped_rotation
            # Set the final orientation
            self.data.mocap_quat[0] = final_rotation.as_quat()

        # Handle grasping
        grasp = int(grasp>0)
        if self.data.time - self.prev_grasp_time < 0.5:
            self.gripper_blocked = True
            grasp = self.prev_grasp
        else:
            self.gripper_blocked = False
            if grasp == 0 and self.gripper_state == 0:
                self.gripper_vec = self.gripper_dict["open"]
            elif grasp == 1 and self.gripper_state == 1:
                self.gripper_vec = self.gripper_dict["closed"]
            elif grasp == 0 and self.gripper_state == 1:
                self.data.ctrl[self._gripper_ctrl_id] = 255
                self.gripper_state = 0
                self.gripper_vec = self.gripper_dict["opening"]
                self.prev_grasp_time = self.data.time
                self.prev_grasp = grasp
            elif grasp == 1 and self.gripper_state == 0:
                self.data.ctrl[self._gripper_ctrl_id] = 0
                self.gripper_state = 1
                self.gripper_vec = self.gripper_dict["closing"]
                self.prev_grasp_time = self.data.time
                self.prev_grasp = grasp

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
        reward, info = self._compute_reward(action)
        self.prev_gripper_state = self.gripper_state

        return obs, reward, False, False, info 
    
    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array", camera_id=cam_id)
            )
        return rendered_frames

    def _get_obs(self):
        obs = {"state": {}}

        # Populate state observations
        obs["state"]["panda/tcp_pos"] = self.data.sensor("pinch_pos").data.astype(np.float32)
        obs["state"]["panda/tcp_orientation"] = self.data.sensor("pinch_quat").data.astype(np.float32)
        obs["state"]["panda/tcp_vel"] = self.data.sensor("pinch_vel").data.astype(np.float32)
        obs["state"]["panda/gripper_pos"] = (25*2*np.array([self.data.qpos[8]], dtype=np.float32)-1)
        obs["state"]["panda/gripper_vec"] = self.gripper_vec

        if not self.image_obs:
            obs["state"]["block_pos"] = self.data.sensor("block_pos").data.astype(np.float32)
        if self.image_obs:
            obs["images"] = {}
            obs["images"]["wrist"], obs["images"]["front"] = self.render()

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs
        
    def _compute_reward(self, action):
        block_pos = self.data.sensor("block_pos").data
        tcp_pos = self.data.sensor("pinch_pos").data
        box_target = 1 - np.tanh(5 * np.linalg.norm(block_pos - self._block_success))
        gripper_box = 1 - np.tanh(5 * np.linalg.norm(block_pos - tcp_pos))

        r_energy = -np.linalg.norm(action)

        # # Reward for staying near center_pos
        # r_move = 1 - np.tanh(5 * np.linalg.norm(tcp_pos[:2] - self.center_pos[:2]))

        # # Smoothness reward
        # action_diff = np.linalg.norm(action[:-1] - self.prev_action[:-1])/np.sqrt((len(action)-1)) # Divide by sqrt of action dimension to make it scale-invariant
        # r_smooth = 1 - np.tanh(5 * action_diff)
        # self.prev_action = action

        # # Reward for not interrupting grasp
        # if self.gripper_blocked and self.gripper_state != self.prev_gripper_state:
        #     r_grasp = 0
        # else:
        #     r_grasp = 1

        # # Check if gripper pads are in contact with the object
        # right_pad_contact = False
        # left_pad_contact = False
        # for i in range(self.data.ncon):
        #     geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom1)
        #     geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.data.contact[i].geom2)
        #     if geom1_name == None:
        #         geom1_name = ""
        #     if geom2_name == None:
        #         geom2_name = ""
        #     geom_names = geom1_name + geom2_name
        #     if "block" in geom_names:
        #         if "right_pad" in geom_names:
        #             right_pad_contact = True
        #         if "left_pad" in geom_names:
        #             left_pad_contact = True
        #     if right_pad_contact and left_pad_contact:
        #         break            

        # # Reward if both pads are in contact with the object    
        # if right_pad_contact and left_pad_contact:
        #     r_contact = 1
        # else:
        #     r_contact = 0

        # rewards = {'box_target': box_target, 'gripper_box': gripper_box, 'r_smooth': r_smooth, 'r_grasp': r_grasp, 'r_contact': r_contact, 'r_move': r_move}
        # reward_scales = {'box_target': 8.0, 'gripper_box': 4.0, 'r_smooth': 0.5, 'r_grasp': 0.5, 'r_contact': 0.5, 'r_move': 1.0}

        rewards = {'box_target': box_target, 'gripper_box': gripper_box, 'r_energy': r_energy}
        reward_scales = {'box_target': 8.0, 'gripper_box': 4.0, 'r_energy': 1.0}

        rewards = {k: v * reward_scales[k] for k, v in rewards.items()}
        reward = np.clip(sum(rewards.values()), -1e4, 1e4)
            
        # Success if
        if box_target < 0.1:
            success = True
        else:
            success = False
        
        info = rewards
        info['success'] = success
        return reward, info