# https://github.com/rail-berkeley/serl/blob/e2065d673131af6699aa899a78159859bd17c135/franka_sim/franka_sim/envs/panda_pick_gym_env.py
import numpy as np
import os
import mujoco
import random

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
from gymnasium.spaces import Box, Dict
from gym_INB0104.controllers import opspace_4 as opspace
from pathlib import Path
from scipy.spatial.transform import Rotation
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    }

class ReachIKDeltaStrawbHangingEnv(MujocoEnv, utils.EzPickle):
    metadata = { 
        "render_modes": ["human", "rgb_array", "depth_array"], 
    }
    
    def __init__(
        self,
        image_obs=True,
        randomize_domain=True,
        ee_dof = 6, # 3 for position, 3 for orientation
        control_dt=0.1,
        physics_dt=0.001,
        width=480,
        height=480,
        pos_scale=0.1,
        rot_scale=0.05,
        cameras=["wrist1", "wrist2", "front"],
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
        self.cameras = cameras

        config_path = Path(__file__).parent.parent / "configs" / "strawb_hanging.yaml"
        self.cfg = load_config(config_path)

        state_space = Dict(
            {
                "panda/tcp_pos": Box(np.array([0.28, -0.5, 0.01]), np.array([0.75, 0.5, 0.8]), shape=(3,), dtype=np.float32),
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
            self.observation_space["images"] = Dict()
            for camera in self.cameras:
                self.observation_space["images"][camera] = Box(
                    0, 255, shape=(self.height, self.width, 3), dtype=np.uint8
                )

        p = Path(__file__).parent
        env_dir = os.path.join(p, "xmls/reach_strawb_hanging.xml")
        self._n_substeps = int(float(control_dt) / float(physics_dt))
        self.frame_skip = 1

        MujocoEnv.__init__(
            self, 
            env_dir, 
            self.frame_skip, 
            observation_space=self.observation_space, 
            render_mode=self.render_mode,
            width=self.width,
            height=self.height,
            default_camera_config=DEFAULT_CAMERA_CONFIG, 
            camera_id=0, 
            **kwargs,
        )
        self.model.opt.timestep = physics_dt
        self.camera_id = ()
        for cam in self.cameras:
            self.camera_id += (self.model.camera(cam).id,)
        self.action_space = Box(
            np.array([-1.0]*(self.ee_dof+1)), 
            np.array([1.0]*(self.ee_dof+1)),
            dtype=np.float32,
        )
        self._viewer = MujocoRenderer(self.model, self.data,)
        self.setup()

    def setup(self):
        self._PANDA_HOME = np.array([0.0, -1.15, -0.12, -2.98, -0.14, 3.35, 0.84], dtype=np.float32)
        self._GRIPPER_HOME = np.array([0.04, 0.04], dtype=np.float32)
        self._PANDA_XYZ = np.array([0.3, 0, 0.7], dtype=np.float32)
        self._CARTESIAN_BOUNDS = np.array([[0.28, -0.35, 0.005], [0.8, 0.35, 0.8]], dtype=np.float32)
        self._ROTATION_BOUNDS= np.array([[-np.pi/4, -np.pi/2, -np.pi/2], [np.pi/4, np.pi/2, np.pi/2]], dtype=np.float32)

        self.default_obj_pos = np.array([0.6, 0, 0.8])
        self._panda_dof_ids = np.array([self.model.joint(f"joint{i}").id for i in range(1, 8)])
        self._panda_ctrl_ids = np.array([self.model.actuator(f"actuator{i}").id for i in range(1, 8)])
        self._gripper_ctrl_id = self.model.actuator("fingers_actuator").id
        self._pinch_site_id = self.model.site("pinch").id
        
        self.reset_arm_and_gripper()

        self.prev_action = np.zeros(self.action_space.shape)
        self.prev_grasp_time = 0.0
        self.prev_grasp = -1.0
        self.gripper_dict = {
            "moving": np.array([1, 0], dtype=np.float32),
            "grasping": np.array([0, 1], dtype=np.float32),
        }

        # Store initial values for randomization
        for camera_name in self.cameras:
            setattr(self, f"{camera_name}_pos", self.model.body_pos[self.model.body(camera_name).id].copy())
            setattr(self, f"{camera_name}_quat", self.model.body_quat[self.model.body(camera_name).id].copy())
        self.init_light_pos = self.model.body_pos[self.model.body('light0').id].copy()
        self.init_table_surface_rgba = self.model.mat_rgba[self.model.mat('table_surface').id].copy()
        self.init_front_wall_rgba = self.model.mat_rgba[self.model.mat('front_wall').id].copy()
        self.init_brick_rgba = self.model.mat_rgba[self.model.mat('brick_wall').id].copy()


        self.table_tex_ids = []
        self.skybox_tex_ids = []
        self.brick_wall_tex_ids = []
        self.front_wall_tex_ids = []
        self.floor_tex_ids = []
        for i in range(self.model.ntex):
            if i < self.model.ntex - 1:
                # For all but the last texture, use the next index
                name_start = self.model.name_texadr[i]
                name_end = self.model.name_texadr[i + 1] - 1
            else:
                # For the last texture, go until the first null byte or the end of the names array
                name_start = self.model.name_texadr[i]
                name_end = len(self.model.names)
            # Decode the name slice
            texture_name = self.model.names[name_start:name_end].split(b'\x00', 1)[0].decode('utf-8')
            if self.model.texture(texture_name).type[0] == 2:
                self.skybox_tex_ids.append(self.model.texture(texture_name).id)
            else:
                if 'Brick' in texture_name:
                    self.brick_wall_tex_ids.append(self.model.texture(texture_name).id)
                elif any(substring in texture_name for substring in ['Wood', 'Table', 'Rock', 'Plank']):
                    self.table_tex_ids.append(self.model.texture(texture_name).id)
                elif any(substring in texture_name for substring in ['Paving', 'Tiles', 'Ground', 'Gravel', 'Grass']):
                    self.floor_tex_ids.append(self.model.texture(texture_name).id)
                else:
                    self.front_wall_tex_ids.append(self.model.texture(texture_name).id)

        self.initial_vine_rotation = Rotation.from_quat(np.roll(self.model.body_quat[self.model.body("vine").id], -1))

        # Add this line to set the initial orientation
        self.initial_orientation = [0, 0.725, 0.0, 0.688]
        self.initial_rotation = Rotation.from_quat(self.initial_orientation)

        self.init_headlight_diffuse = self.model.vis.headlight.diffuse.copy()
        self.init_headlight_ambient = self.model.vis.headlight.ambient.copy()
        self.init_headlight_specular = self.model.vis.headlight.specular.copy()

    def lighting_noise(self):

        # Add noise to light position
        light_pos_noise = np.random.uniform(low=[-0.8,-0.5,-0.05], high=[1.2,0.5,0.2], size=3)
        self.model.body_pos[self.model.body('light0').id] = self.init_light_pos + light_pos_noise

        # Change light levels with varying brightness conditions
        if random.random() < 0.5:  # 50% chance to make it darker
            # Darker lighting configuration
            light_diffuse_noise = np.random.uniform(low=0.05, high=0.3, size=3)  # Lower diffuse values for dim light
            light_ambient_noise = np.random.uniform(low=0.0, high=0.2, size=3)   # Lower ambient light for a darker environment
        else:
            # Brighter lighting configuration (similar to current implementation)
            light_diffuse_noise = np.random.uniform(low=0.1, high=1.0, size=3)
            light_ambient_noise = np.random.uniform(low=0.0, high=0.5, size=3)
        self.model.light_diffuse[0] = light_diffuse_noise
        self.model.light_ambient[0] = light_ambient_noise

        # Randomize other properties like specular lighting
        light_specular_noise = np.random.uniform(low=0.0, high=0.5, size=3)
        self.model.light_specular[0] = light_specular_noise

        # Add noise to headlight
        if random.random() < 0.5:
            light = -1.0
            # Darker lighting configuration
            headlight_diffuse_noise = np.random.uniform(low=0.0, high=0.1, size=3)  # Lower diffuse values for dim light
            headlight_ambient_noise = np.random.uniform(low=0.0, high=0.1, size=3)   # Lower ambient light for a darker environment
            headlight_specular_noise = np.random.uniform(low=0.0, high=0.1, size=3)  # Lower specular light for a darker environment
        else:
            light = 1.0
            # Brighter lighting configuration (similar to current implementation)
            headlight_diffuse_noise = np.random.uniform(low=0.0, high=0.1, size=3)
            headlight_ambient_noise = np.random.uniform(low=0.0, high=0.1, size=3)
            headlight_specular_noise = np.random.uniform(low=0.0, high=0.1, size=3)

        self.model.vis.headlight.diffuse = self.init_headlight_diffuse + light * headlight_diffuse_noise
        self.model.vis.headlight.ambient = self.init_headlight_ambient + light * headlight_ambient_noise
        self.model.vis.headlight.specular = self.init_headlight_specular + light * headlight_specular_noise

    def action_scale_noise(self):
        pos_scale_range = self.cfg.get("pos_scale_range", [0.0, 0.0])
        rot_scale_range = self.cfg.get("rot_scale_range", [0.0, 0.0])
        self.pos_scale = random.uniform(*pos_scale_range)
        self.rot_scale = random.uniform(*rot_scale_range)

    def initial_state_noise(self):
        ee_noise_low = self.cfg.get("ee_noise_low", [0.0, 0.0, 0.0])
        ee_noise_high = self.cfg.get("ee_noise_high", [0.12, 0.2, 0.1])
        ee_noise = np.random.uniform(low=ee_noise_low, high=ee_noise_high, size=3)
        self.data.mocap_pos[0] = self._PANDA_XYZ + ee_noise

    def camera_noise(self):
        for cam_name in self.cameras:
            # Fetch noise ranges from configuration or use default values
            pos_noise_low = self.cfg.get(f"{cam_name}_pos_noise_low", [0.0, 0.0, 0.0])
            pos_noise_high = self.cfg.get(f"{cam_name}_pos_noise_high", [0.0, 0.0, 0.0])
            quat_noise_range = self.cfg.get(f"{cam_name}_quat_noise_range", [0.0, 0.0])

            # Randomize position
            cam_pos_noise = np.random.uniform(low=pos_noise_low, high=pos_noise_high, size=3)
            self.model.body_pos[self.model.body(cam_name).id] = getattr(self, f"{cam_name}_pos") + cam_pos_noise

            # Randomize orientation
            cam_quat_noise = np.random.uniform(low=quat_noise_range[0], high=quat_noise_range[1], size=4)
            new_cam_quat = getattr(self, f"{cam_name}_quat") + cam_quat_noise
            new_cam_quat /= np.linalg.norm(new_cam_quat)
            self.model.body_quat[self.model.body(cam_name).id] = new_cam_quat

    def table_noise(self):
        if random.random() < 0.5:
            # Make all geoms in the "workbench" body invisible
            for geom_id in range(int(self.model.body("workbench").geomadr), int(self.model.body("workbench").geomadr + self.model.body("workbench").geomnum)):
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                # Set the group to 3 (hidden group)
                self.model.geom_group[geom_id] = 3
        else:
            # Make all geoms in the "workbench" body visible
            for geom_id in range(int(self.model.body("workbench").geomadr), int(self.model.body("workbench").geomadr + int(self.model.body("workbench").geomnum))):
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 1
                # Set the group to 0 (default group)
                self.model.geom_group[geom_id] = 0

            table_tex_id = np.random.choice(self.table_tex_ids)
            material_id = self.model.mat('table_surface').id
            self.model.mat_texid[material_id] = table_tex_id

            # Randomize table color
            channel = np.random.randint(0, 3)
            table_color_noise_range = self.cfg.get("table_color_noise_range", [0.0, 0.0])
            table_color_noise = np.random.uniform(low=table_color_noise_range[0], high=table_color_noise_range[1], size=1)
            self.model.mat_rgba[material_id][channel] += table_color_noise

    def wall_noise(self):
        if random.random() < 0.7:
            for geom_id in range(int(self.model.body("walls").geomadr), int(self.model.body("walls").geomadr + self.model.body("walls").geomnum)):
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
                self.model.geom_group[geom_id] = 3
        else:
            for geom_id in range(int(self.model.body("walls").geomadr), int(self.model.body("walls").geomadr + self.model.body("walls").geomnum)):
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 1
                self.model.geom_group[geom_id] = 0
            front_wall_tex_id = np.random.choice(self.front_wall_tex_ids)
            self.model.mat_texid[self.model.mat('front_wall').id] = front_wall_tex_id
            channel = np.random.randint(0,3)
            wall_color_noise_range = self.cfg.get("wall_color_noise_range", [0.0, 0.0])
            wall_color_noise = np.random.uniform(low=wall_color_noise_range[0], high=wall_color_noise_range[1], size=1)
            self.model.mat_rgba[self.model.mat('front_wall').id] = self.init_front_wall_rgba.copy()
            self.model.mat_rgba[self.model.mat('front_wall').id][channel] = self.init_front_wall_rgba[channel] + wall_color_noise

            # Randomize brick color
            brick_wall_tex_id = np.random.choice(self.brick_wall_tex_ids)
            self.model.mat_texid[self.model.mat('brick_wall').id] = brick_wall_tex_id
            self.model.mat_rgba[self.model.mat('brick_wall').id] = self.init_brick_rgba.copy()
            self.model.mat_rgba[self.model.mat('brick_wall').id][channel] = self.init_brick_rgba[channel] + wall_color_noise

    def floor_noise(self):
        if random.random() < 0.5:
            # Set the group to 3 (hidden group)
            self.model.geom('floor').group = 3
        else:
            self.model.geom('floor').group = 0
            floor_tex_id = np.random.choice(self.floor_tex_ids)
            self.model.mat_texid[self.model.mat('floor').id] = floor_tex_id

    def skybox_noise(self):
        skybox_tex_id = np.random.choice(self.skybox_tex_ids)
        self._viewer.model.tex_adr[0] = self.model.tex_adr[skybox_tex_id]

    def object_noise(self):
        # Target pos
        target_pos_noise_low = self.cfg.get("target_pos_noise_low", [0.0, 0.0, 0.0])
        target_pos_noise_high = self.cfg.get("target_pos_noise_high", [0.0, 0.0, 0.0])
        target_pos_noise = np.random.uniform(low=target_pos_noise_low, high=target_pos_noise_high, size=3)
        target_pos = self.default_obj_pos + target_pos_noise
        self.model.body_pos[self.model.body("vine").id] = target_pos
        # Target orientation
        random_z_angle = np.random.uniform(low=-np.pi, high=np.pi)  # Random angle in radians
        z_rotation = Rotation.from_euler('z', random_z_angle)
        new_rotation = z_rotation * self.initial_vine_rotation
        new_quat = new_rotation.as_quat()
        self.model.body_quat[self.model.body("vine").id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        # Distractor 1 pos
        distract1_pos_noise_low = self.cfg.get("distract1_pos_noise_low", [0.0, 0.0, 0.0])
        distract1_pos_noise_high = self.cfg.get("distract1_pos_noise_high", [0.0, 0.0, 0.0])
        distract1_pos_noise = np.random.uniform(low=distract1_pos_noise_low, high=distract1_pos_noise_high, size=3)
        self.model.body_pos[self.model.body("vine2").id] = target_pos + distract1_pos_noise
        # Distractor 1 orientation
        random_z_angle = np.random.uniform(low=-np.pi, high=np.pi)  # Random angle in radians
        z_rotation = Rotation.from_euler('z', random_z_angle)
        new_rotation = z_rotation * self.initial_vine_rotation
        new_quat = new_rotation.as_quat()
        self.model.body_quat[self.model.body("vine2").id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        # Distractor 2 pos
        distract2_pos_noise_low = self.cfg.get("distract2_pos_noise_low", [0.0, 0.0, 0.0])
        distract2_pos_noise_high = self.cfg.get("distract2_pos_noise_high", [0.0, 0.0, 0.0])
        distract2_pos_noise = np.random.uniform(low=distract2_pos_noise_low, high=distract2_pos_noise_high, size=3)
        self.model.body_pos[self.model.body("vine3").id] = target_pos + distract2_pos_noise
        # Distractor 2 orientation
        random_z_angle = np.random.uniform(low=-np.pi, high=np.pi)  # Random angle in radians
        z_rotation = Rotation.from_euler('z', random_z_angle)
        new_rotation = z_rotation * self.initial_vine_rotation
        new_quat = new_rotation.as_quat()
        self.model.body_quat[self.model.body("vine3").id] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def domain_randomization(self):
        if self.cfg.get("apply_lighting_noise", False):
            self.lighting_noise()
        if self.cfg.get("apply_action_scale_noise", False):
            self.action_scale_noise()
        if self.cfg.get("apply_initial_state_noise", False):
            self.initial_state_noise()
        if self.cfg.get("apply_camera_noise", False):
            self.camera_noise()
        if self.cfg.get("apply_table_noise", False):
            self.table_noise()
        if self.cfg.get("apply_skybox_noise", False):
            self.skybox_noise()
        if self.cfg.get("apply_wall_noise", False):
            self.wall_noise()
        if self.cfg.get("apply_floor_noise", False):
            self.floor_noise()
        if self.cfg.get("apply_skybox_noise", False):
            self.skybox_noise()
        if self.cfg.get("apply_object_noise", False):
            self.object_noise()
        self._viewer = MujocoRenderer(self.model, self.data,)

    def reset_arm_and_gripper(self):
        self.data.qpos[self._panda_dof_ids] = self._PANDA_HOME
        self.data.qpos[7:9] = self._GRIPPER_HOME
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[0] = self.data.sensor("pinch_pos").data.copy()
        self.data.mocap_quat[0] = self.data.sensor("pinch_quat").data.copy()
        mujoco.mj_step(self.model, self.data)


    def reset_model(self):
        self.reset_arm_and_gripper()
        if self.randomize_domain:
            self.domain_randomization()

        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(1):
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
        self._x_success = self._block_init[0] - 0.1
        self._z_success = self._block_init[2] + 0.05
        self._block_success = self._block_init.copy()
        self._block_success[0] = self._x_success
        self._block_success[2] = self._z_success

        self._block2_init = self.data.sensor("block2_pos").data
        self._block3_init = self.data.sensor("block3_pos").data

        self.gripper_vec = self.gripper_dict["moving"]
        self.data.ctrl[self._gripper_ctrl_id] = 255
        self.grasp = -1.0
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
        if self.data.time - self.prev_grasp_time < 0.5:
            self.gripper_blocked = True
            grasp = self.prev_grasp
        else:
            grasp = np.round(grasp,1)
            if grasp == self.prev_grasp:
                self.gripper_blocked = False
            else:
                if -1 <= grasp <= 0:
                    self.data.ctrl[self._gripper_ctrl_id] = int(255 + (40 - 255) * (grasp + 1))
                    self.prev_grasp_time = self.data.time
                    self.prev_grasp = grasp
                    self.gripper_vec = self.gripper_dict["moving"]
                elif 0 < grasp <= 1:
                    self.data.ctrl[self._gripper_ctrl_id] = 0
                    self.prev_grasp_time = self.data.time
                    self.prev_grasp = grasp
                    self.gripper_vec = self.gripper_dict["grasping"]
        self.grasp = grasp

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
        
        # Original position and orientation observations
        tcp_pos = self.data.sensor("pinch_pos").data
        tcp_orientation = self.data.sensor("pinch_quat").data
        # Define noise parameters
        position_noise_std = 0.01  # e.g., 1 cm standard deviation
        orientation_noise_std = 0.005  # e.g., small rotations in quaternion
        # Add Gaussian noise to position and orientation
        noisy_tcp_pos = tcp_pos + np.random.normal(0, position_noise_std, size=tcp_pos.shape)
        noisy_tcp_orientation = tcp_orientation + np.random.normal(0, orientation_noise_std, size=tcp_orientation.shape)
        # Normalize orientation quaternion to keep it valid
        noisy_tcp_orientation /= np.linalg.norm(noisy_tcp_orientation)
        
        # Populate noisy observations
        obs["state"]["panda/tcp_pos"] = noisy_tcp_pos.astype(np.float32)
        obs["state"]["panda/tcp_orientation"] = noisy_tcp_orientation.astype(np.float32)
        obs["state"]["panda/tcp_vel"] = self.data.sensor("pinch_vel").data.astype(np.float32)
        obs["state"]["panda/gripper_pos"] = 25 * 2 * np.array([self.data.qpos[8]], dtype=np.float32) - 1
        obs["state"]["panda/gripper_vec"] = np.concatenate([self.gripper_vec, [self.grasp], [int(self.gripper_blocked)]]).astype(np.float32)

        if not self.image_obs:
            obs["state"]["block_pos"] = self.data.sensor("block_pos").data.astype(np.float32)
        if self.image_obs:
            obs["images"] = {}
            for cam_name in self.cameras:
                cam_id = self.model.camera(cam_name).id
                obs["images"][cam_name] = self._viewer.render(render_mode="rgb_array", camera_id=cam_id)

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs
        
    def _compute_reward(self, action):
        block_pos = self.data.sensor("block_pos").data
        tcp_pos = self.data.sensor("long_pinch_pos").data
        box_target = 1 - np.tanh(5 * np.linalg.norm(block_pos - self._block_success))
        gripper_box = 1 - np.tanh(5 * np.linalg.norm(block_pos - tcp_pos))

        block2_pos = self.data.sensor("block2_pos").data
        r_block2 = 1- np.tanh(5*np.linalg.norm(block2_pos - self._block2_init))

        block3_pos = self.data.sensor("block3_pos").data
        r_block3 = 1 - np.tanh(5*np.linalg.norm(block3_pos - self._block3_init))

        r_energy = -np.linalg.norm(action)

        # Smoothness reward
        r_smooth = -np.linalg.norm(action - self.prev_action) 
        self.prev_action = action

        rewards = {'box_target': box_target, 'gripper_box': gripper_box, 'r_block2': r_block2, 'r_block3': r_block3, 'r_energy': r_energy, 'r_smooth': r_smooth}
        reward_scales = {'box_target': 8.0, 'gripper_box': 4.0, 'r_block2': 1.0, 'r_block3': 1.0, 'r_energy': 2.0 , 'r_smooth': 1.0}

        rewards = {k: v * reward_scales[k] for k, v in rewards.items()}
        reward = np.clip(sum(rewards.values()), -1e4, 1e4)
            
        # Success if
        if box_target < 0.01:
            success = True
        else:
            success = False
        
        info = rewards
        info['success'] = success
        return reward, info