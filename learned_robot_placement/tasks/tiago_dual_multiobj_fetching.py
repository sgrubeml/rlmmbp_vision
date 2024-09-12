# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import numpy as np
import math
import random
import open3d as o3d
from collections import deque

from learned_robot_placement.tasks.base.rl_task import RLTask
from learned_robot_placement.handlers.tiagodualWBhandler import TiagoDualWBHandler
from omni.isaac.core.objects.cone import VisualCone
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.objects import FixedCuboid
from learned_robot_placement.tasks.utils.pinoc_utils import PinTiagoIKSolver # For IK
from learned_robot_placement.tasks.utils import scene_utils, pc_utils

# from omni.isaac.core.utils.prims import get_prim_at_path
# from omni.isaac.core.utils.prims import create_prim
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.physx import get_physx_interface, get_physx_simulation_interface
from omni.physx.bindings._physx import SimulationEvent
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.torch.rotations import euler_angles_to_quats, quat_diff_rad
from scipy.spatial.transform import Rotation
from pxr import PhysicsSchemaTools


from omni.isaac.core.utils.semantics import add_update_semantics 


# Base placement environment for fetching a target object among clutter
class TiagoDualMultiObjFetchingTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env
    ) -> None:
        self._env = env # temp, delete TODO
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._device = self._cfg["sim_device"]

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        
        self._gamma = self._task_cfg["env"]["gamma"]
        self._max_episode_length = self._task_cfg["env"]["horizon"]
        
        self._randomize_robot_on_reset = self._task_cfg["env"]["randomize_robot_on_reset"]
        self._face_goal_on_reset = self._task_cfg["env"]["face_goal_on_reset"]
        self._obstacle_scale= self._task_cfg["env"]["obstacle_scale"]
        self._scene_type = self._task_cfg["env"]["scene_type"]
        self._random_goal = self._task_cfg["env"]["random_goal"]

        self._n_collisions = 0

        # Get dt for integrating velocity commands and checking limit violations
        self._dt = torch.tensor(self._sim_config.task_config["sim"]["dt"]*self._sim_config.task_config["env"]["controlFrequencyInv"],device=self._device)

        # Environment object settings: (reset() randomizes the environment)
        self._obstacle_names = ["mammut", "godishus"] # ShapeNet models in usd format
        self._tabular_obstacle_mask = [True] * self._task_cfg["env"]["num_obstacles"] # Mask to denote which objects are tabular (i.e. grasp objects can be placed on them)
        if self._task_cfg["env"]["num_grasp_objects"] == 4:
            self._grasp_obj_names = ["004_sugar_box", "008_pudding_box", "010_potted_meat_can", "061_foam_brick"]
        else:
            self._grasp_obj_names = ["025_mug", "008_pudding_box", "010_potted_meat_can", "003_cracker_box", "021_bleach_cleanser", "004_sugar_box"] # YCB models in usd format

        self._num_obstacles = self._task_cfg["env"]["num_obstacles"]
        self._navigation_task = self._task_cfg["env"]["navigation_task"]
        if self._navigation_task == "simple":
            self._num_obstacles = 1
        # self._num_grasp_objs = min(self._task_cfg["env"]["num_grasp_objects"],len(self._grasp_obj_names))
        self._num_grasp_objs = self._task_cfg["env"]["num_grasp_objects"]
        self._obstacles = []
        self._obstacles_dimensions = []
        self._grasp_objs = []
        self._grasp_objs_dimensions = []

        self._move_group = self._task_cfg["env"]["move_group"]
        self._use_torso = self._task_cfg["env"]["use_torso"]
        self._use_base = self._task_cfg["env"]["use_base"]
        self._use_head = self._task_cfg["env"]["use_head"]
        # Position control. Actions are base SE2 pose (3) and discrete arm activation (2)
        self._num_actions = self._task_cfg["env"]["continous_actions"] + self._task_cfg["env"]["discrete_actions"]
        # env specific limits
        self._world_xy_radius = self._task_cfg["env"]["world_xy_radius"]
        self._action_xy_radius = self._task_cfg["env"]["action_xy_radius"]
        self._action_ang_lim = self._task_cfg["env"]["action_ang_lim"]
        self.max_arm_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_rot_vel = torch.tensor(self._task_cfg["env"]["max_rot_vel"], device=self._device)
        self.max_base_xy_vel = torch.tensor(self._task_cfg["env"]["max_base_xy_vel"], device=self._device)
        self.max_head_pan_vel = torch.tensor(self._task_cfg["env"]["max_head_pan_vel"], device=self._device)
        self.max_head_tilt_vel = torch.tensor(self._task_cfg["env"]["max_head_tilt_vel"], device=self._device)
        
        # End-effector reaching settings
        self._goal_pos_threshold = self._task_cfg["env"]["goal_pos_thresh"]
        self._goal_ang_threshold = self._task_cfg["env"]["goal_ang_thresh"]
        # For now, setting dummy goal:
        self._goals = torch.hstack((torch.tensor([[0.8,0.0,0.4+0.15]], device=self._device),euler_angles_to_quats(torch.tensor([[0.19635, 1.375, 0.19635]]),device=self._device)))[0].repeat(self.num_envs,1)
        self._goal_tf = torch.zeros((4,4),device=self._device)
        self._goal_tf[:3,:3] = torch.tensor(Rotation.from_quat(np.array([self._goals[0,3+1].cpu(),self._goals[0,3+2].cpu(),self._goals[0,3+3].cpu(),self._goals[0,3].cpu()])).as_matrix(),dtype=float,device=self._device) # Quaternion in scalar last format!!!
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0].cpu(), self._goals[0,1].cpu(), self._goals[0,2], 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self._goals[:,0:2],dim=1)  # distance from origin

        self._wall_offset = self._task_cfg["env"]["wall_offset"]
        self._wall_r = self._world_xy_radius + self._wall_offset

        # Reward settings
        self._reward_success = self._task_cfg["env"]["reward_success"]
        self._reward_dist_weight = self._task_cfg["env"]["reward_dist_weight"]
        self._reward_look_at_goal = self._task_cfg["env"]["reward_look_at_goal"]
        self._reward_new_points = self._task_cfg["env"]["reward_new_points"]
        self._reward_noIK = self._task_cfg["env"]["reward_noIK"]
        # self._reward_timeout = self._task_cfg["env"]["reward_timeout"]
        self._reward_collision = self._task_cfg["env"]["reward_collision"]
        self._terminate_on_collision = self._task_cfg["env"]["terminate_on_collision"]
        self._collided = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._ik_fails = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._is_success = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        self._get_renders = self._cfg["get_renders"]

        self._obs_representation = self._task_cfg["env"]["obs_representation"]

        if self._obs_representation != "bbox":
            self.pcl_size = self._task_cfg["obs"]["visual"]["pcl_size"]

        self.robot_state_history_len = self._task_cfg["obs"]["robot_state_history_len"]
        self.obs_history_len = self._task_cfg["obs"]["obs_history_len"]
        self.collect_action_history = self._task_cfg["obs"]["action_history"]
        self.collect_reward_history = self._task_cfg["obs"]["reward_history"]
        self.collect_robot_pose_history = self._task_cfg["obs"]["robot_pose_history"]
        self.collect_goal_pose_history = self._task_cfg["obs"]["goal_pose_history"]

        self.action_history_len = self._num_actions*self.robot_state_history_len*self.collect_action_history
        self.reward_history_len = self.robot_state_history_len*self.collect_reward_history
        self.robot_pose_history_len = 3*(self.robot_state_history_len-1)*self.collect_robot_pose_history #one element mandatory -> history_len-1
        self.goal_pose_history_len = 7*(self.robot_state_history_len-1)*self.collect_goal_pose_history #one element mandatory -> history_len-1
        self.robot_state_len = self.action_history_len + self.reward_history_len + self.robot_pose_history_len + 3 + self.goal_pose_history_len + 7

        self._downsample_strategy = self._task_cfg["obs"]["visual"]["downsampling"]


        # Choose num_obs and num_actions based on task
        if self._obs_representation == "voxel_downsample":
            self.obs_history_len = 1 # cant be other value
            # pcl_size*3 + goal pose + robot pose + history len
            self.observation_history_len = 3*self.pcl_size*(self.obs_history_len-1)
            self._o3d_pcl_map = o3d.geometry.PointCloud()
            self._num_observations = 3*self.pcl_size + self.observation_history_len + self.robot_state_len


        if self._obs_representation == "keyframe_voxel":
            self.obs_history_len = 1 # cant be other value
            self._num_keyframes = self._task_cfg["obs"]["visual"]["keyframe_voxel"]["num_keyframes"]
            self._portion_new_points = self._task_cfg["obs"]["visual"]["keyframe_voxel"]["portion_new_points"]
            self.observation_history_len = 3*self.pcl_size*(self.obs_history_len-1)
            self._num_observations = 3*self.pcl_size + self.observation_history_len + self.robot_state_len
            self._o3d_pcl_map = o3d.geometry.PointCloud()
            self._keyframes = deque([None] * self._num_keyframes)
            #self._last_robot_obs = None


        if self._obs_representation == "stack":
            self._pcl_counter = 0
            self._channels = self._task_cfg["obs"]["visual"]["stack"]["channels"]
            self.observation_history_len = self._channels*self.pcl_size*(self.obs_history_len-1)
            self._num_observations = self._channels*self.pcl_size + self.observation_history_len + self.robot_state_len

        
        if self._obs_representation == "tsdf":
            self.observation_history_len = 0
            # cut off anything above 1.50m
            self._tsdf_res = self._task_cfg["obs"]["visual"]["tsdf"]["tsdf_resolution"]
            self._voxel_length = (self._wall_r*2) / self._tsdf_res
            # set tsdf origin to bottom right corner of room
            self._tsdf_tf = torch.zeros((4,4), dtype=torch.float, device=self._device)
            self._tsdf_tf[:2,-1] = torch.tensor([[self._wall_r, self._wall_r]], dtype=torch.float, device=self._device)
            self._tsdf_tf[:2,:2] = torch.tensor([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]], dtype=torch.float, device=self._device)
            self._goal_tf_tsdf_frame = torch.matmul(self._tsdf_tf, self._goal_tf)
            self._intrinsic = None
            self._tsdf = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length = self._voxel_length,
                                                                      sdf_trunc = self._voxel_length * 4,
                                                                      color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
            self._num_observations = 4*self.pcl_size + self.robot_state_len

        
        if self._obs_representation == 'bbox':
            self._obj_states = torch.zeros((6*(4+self._num_obstacles+(self._num_grasp_objs-1)),self._num_envs),device=self._device) # All walls, obstacles and grasp objs except the target object will be used in obj state (BBox)
            self.observation_history_len = len(self._obj_states)*(self.obs_history_len-1)
            self._num_observations = len(self._obj_states) + self.obs_history_len + self.robot_state_len
            self._wall_bboxes = None


        self.action_history = torch.zeros((1, self.action_history_len), dtype=torch.float, device=self._device)
        self.reward_history = torch.zeros((1, self.reward_history_len), dtype=torch.float, device=self._device)
        self.robot_pose_history = torch.zeros((1, self.robot_pose_history_len), dtype=torch.float, device=self._device)
        self.goal_pose_history = torch.zeros((1, self.goal_pose_history_len), dtype=torch.float, device=self._device)
        self.observation_history = torch.zeros((1, self.observation_history_len), dtype=torch.float, device=self._device)

        # Handler for Tiago
        self.tiago_handler = TiagoDualWBHandler(env=env, move_group=self._move_group, use_torso=self._use_torso, use_base=self._use_base,
                                                use_head=self._use_head, sim_config=self._sim_config, num_envs=self._num_envs, device=self._device,
                                                intrinsics=[self._task_cfg["env"]["fx"], self._task_cfg["env"]["fy"], self._task_cfg["env"]["cx"], self._task_cfg["env"]["cy"]])

        RLTask.__init__(self, name, env)


    def set_up_scene(self, scene) -> None:
        from pxr import PhysxSchema
        from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
        physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(get_current_stage(), "/physicsScene")
        physxSceneAPI.CreateEnableCCDAttr(True)
        import omni
        self.tiago_handler.get_robot()
        # for n obstacles to spawn, calculate how many of each type
        dividers = [1] if self._num_obstacles == 1 else sorted(random.sample(range(1, self._num_obstacles), len(self._obstacle_names) - 1))
        n_obst_type = [a - b for a, b in zip(dividers + [self._num_obstacles], [0] + dividers)]
        # Spawn obstacles (from ShapeNet usd models):
        for i in range(len(self._obstacle_names)):
            for j in range(n_obst_type[i]):
                obst_name = self._obstacle_names[i]+str(j)
                obst = scene_utils.spawn_obstacle(self, usd_name=self._obstacle_names[i], obst_name=obst_name,
                                                  prim_path=self.tiago_handler.default_zero_env_path, device=self._device,
                                                  long_obst=(self._obstacle_scale=='long'))
                self._obstacles.append(obst) # Add to list of obstacles (Geometry Prims)
        # use this code if u want random number of objects for each type that sums up to self._num_grasp_objs
        # dividers = [1] if self._num_grasp_objs == 1 else sorted(random.sample(range(1, self._num_grasp_objs), len(self._grasp_obj_names) - 1))
        # n_grasp_type = [a - b for a, b in zip(dividers + [self._num_grasp_objs], [0] + dividers)]
        # Spawn grasp objs (from YCB usd models):
        # use more sugar boxes and cracker boxes to create occlusions
        # this is fixed for number of grasp objects 10, if you want to variate use the commented out code above
        n_grasp_type = [2, 1, 2, 3, 1, 1]
        for i in range(len(self._grasp_obj_names)):
            for j in range(n_grasp_type[i]):
                grasp_obj_name = self._grasp_obj_names[i]+str(j)
                grasp_obj = scene_utils.spawn_grasp_object(self, usd_name=self._grasp_obj_names[i], grasp_obj_name=grasp_obj_name, prim_path=self.tiago_handler.default_zero_env_path, device=self._device)
                self._grasp_objs.append(grasp_obj) # Add to list of grasp objects (Rigid Prims)
        # Goal visualizer
        goal_viz = VisualCone(prim_path=self.tiago_handler.default_zero_env_path+"/goal",
                                radius=0.05,height=0.05,color=np.array([1.0,0.0,0.0]))
        super().set_up_scene(scene)
        self._robots = self.tiago_handler.create_articulation_view()
        scene.add(self._robots)
        self._goal_vizs = GeometryPrimView(prim_paths_expr="/World/envs/.*/goal",name="goal_viz")
        scene.add(self._goal_vizs)
        # Enable object axis-aligned bounding box computations
        scene.enable_bounding_boxes_computations()
        # Add spawned objects to scene registry and store their bounding boxes:
        for obst in self._obstacles:
            scene.add(obst)
            self._obstacles_dimensions.append(scene.compute_object_AABB(obst.name)) # Axis aligned bounding box used as dimensions
        for grasp_obj in self._grasp_objs:
            scene.add(grasp_obj)
            self._grasp_objs_dimensions.append(scene.compute_object_AABB(grasp_obj.name)) # Axis aligned bounding box used as dimensions

        # add walls around the robot's field of movement
        for idx, orientation in enumerate([[0,0,torch.pi/2],[0.0,0.0,0.0],[0,0,torch.pi/2],[0.0,0.0,0.0]]):
            name = "wall_" + str(idx)
            wall_x, wall_y = self._wall_r*np.cos((np.pi/2)*idx), self._wall_r*np.sin((np.pi/2)*idx)
            wall_length = np.linalg.norm(np.array([wall_x, wall_y]))*2
            wall_position = [wall_x, wall_y, 0]
            wall_orientation = Rotation.from_euler("xyz", orientation, degrees=False).as_quat()[np.array([3,0,1,2])] 
            wall = FixedCuboid(
                prim_path= self.tiago_handler.default_zero_env_path + "/" + name,
                name= name,
                position= torch.tensor(wall_position, dtype=torch.float, device=self._device),
                orientation= torch.tensor(wall_orientation,dtype=torch.float,device=self._device),
                scale= torch.tensor([wall_length, 0.02, 0.02], device=self._device),
            )
            add_update_semantics(wall.prim, name)
            scene.add(wall)
            if self._obs_representation == 'bbox':
                if orientation[2] == 0:
                    oriented_bbox = torch.tensor([wall_position[0] - wall_length/2, wall_position[1] - 0.025,
                                                  wall_position[0] + wall_length/2, wall_position[1] + 0.025,
                                                  0.25, orientation[2],] ,dtype=torch.float,device=self._device)
                else:
                     oriented_bbox = torch.tensor([wall_position[0] - 0.025, wall_position[1] - wall_length/2,
                                                   wall_position[0] + 0.025, wall_position[1] + wall_length/2,
                                                   0.25, orientation[2],] ,dtype=torch.float,device=self._device)
                if self._wall_bboxes is None:
                    self._wall_bboxes = oriented_bbox
                else:
                    self._wall_bboxes = torch.vstack(( self._wall_bboxes, oriented_bbox ))
        
        label = "defaultGroundPlane"
        add_update_semantics(get_current_stage().GetPrimAtPath("/World/defaultGroundPlane"), semantic_label=label, type_label="class")
        
        # Optional viewport for rendering in a separate viewer
        from omni.isaac.synthetic_utils import SyntheticDataHelper
        self.viewport_api_window = get_viewport_from_window_name("Viewport")

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        self.tiago_handler.post_reset()

    def get_observations(self):
        # get robots current position
        base_tf = self.tiago_handler.get_robot_tf_world_frame()
        new_base_xy = base_tf[0:2,3].unsqueeze(dim=0)
        new_base_theta = torch.arctan2(base_tf[1,0],base_tf[0,0]).unsqueeze(dim=0).unsqueeze(dim=0)
        # get goal pose
        goal_pos_world_frame = self._goal_tf[0:3,3].unsqueeze(dim=0)
        goal_quat_world_frame = torch.tensor(Rotation.from_matrix(self._goal_tf[:3,:3].cpu()).as_quat()[[3, 0, 1, 2]],dtype=torch.float,device=self._device).unsqueeze(dim=0)
        extrinsics = self.tiago_handler.get_cam_extrinsics_o3d()
        # get robot observation
        if self._obs_representation == "bbox":
            curr_robot_obs = self._obj_bboxes.flatten().unsqueeze(dim=0)
        elif self._obs_representation == "voxel_downsample":
            if self._num_actions == 7:
                self._num_point_old_pcl = np.asarray(self._o3d_pcl_map.points).shape[0]
            curr_robot_obs = pc_utils.voxel_downsample(self, self.tiago_handler.get_pointcloud(), self._o3d_pcl_map, self.pcl_size, self._device, downsample=self._downsample_strategy, extrinsics=extrinsics)
        elif self._obs_representation == "keyframe_voxel":
            curr_robot_obs = pc_utils.voxel_downsample_keyframes(self, self.tiago_handler.get_pointcloud(), self._o3d_pcl_map, self._keyframes, self.pcl_size, self._device,
                                                                 downsample=self._downsample_strategy, dist=self._portion_new_points, extrinsics=extrinsics)
            # if curr_robot_obs is None:
            #     curr_robot_obs = self._last_robot_obs
            # else:
            #     self._last_robot_obs = curr_robot_obs
        elif self._obs_representation == "stack":
            curr_robot_obs, self._pcl_counter = pc_utils.add_pos_to_pcl_channel(self, self.tiago_handler.get_pointcloud(), self.pcl_size, self._max_episode_length, 
                                                                                self._pcl_counter, self._channels, self._device, downsample=self._downsample_strategy,
                                                                                extrinsics=extrinsics)
        elif self._obs_representation == "tsdf":
            if self._intrinsic is None:
                width, height, intrinsics_matrix = self.tiago_handler.get_cam_intrinsics()
                fx, cx, fy, cy = intrinsics_matrix[0,0], intrinsics_matrix[0,2], intrinsics_matrix[1,1], intrinsics_matrix[1,2] 
                self._intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy,
                )
            
            curr_robot_obs = pc_utils.get_pcl_from_tsdf(self, self.pcl_size, self._tsdf, self.tiago_handler.get_depth_img(), self._intrinsic, 
                                                        self.tiago_handler.get_cam_extrinsics_o3d, self._tsdf_tf, self._device)
            
        
        self.obs_buf_idx = 0
        # add observation to obs buffer
        self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+curr_robot_obs.shape[1]] = curr_robot_obs
        self.obs_buf[:,self.obs_buf_idx+curr_robot_obs.shape[1]:self.obs_buf_idx+curr_robot_obs.shape[1]+self.observation_history_len] = self.observation_history
        self.observation_history[:,:] = self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+self.observation_history_len]
        self.obs_buf_idx += curr_robot_obs.shape[1] + self.observation_history_len
        # add goal pose to observation buffer
        self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+3] = goal_pos_world_frame
        self.obs_buf[:,self.obs_buf_idx+3:self.obs_buf_idx+7] = goal_quat_world_frame
        self.obs_buf[:,self.obs_buf_idx+7:self.obs_buf_idx+7+self.goal_pose_history_len] = self.goal_pose_history
        self.goal_pose_history[:,:] = self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+self.goal_pose_history_len]
        self.obs_buf_idx += 7 + self.goal_pose_history_len
        # add robot pose to observation buffer
        self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+2] = new_base_xy
        self.obs_buf[:,self.obs_buf_idx+2] = new_base_theta
        self.obs_buf[:,self.obs_buf_idx+3:self.obs_buf_idx+3+self.robot_pose_history_len] = self.robot_pose_history
        self.robot_pose_history[:,:] = self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+self.robot_pose_history_len]
        self.obs_buf_idx += 3 + self.robot_pose_history_len

        return self.obs_buf

    def get_render(self):
        # Get ground truth viewport rgb image
        return self.tiago_handler.head_camera.get_rgba()[:,:,:3] # TODO get perspective of world frame

    def pre_physics_step(self, actions) -> None:
        # Scale actions (velocities) before sending to robot
        base_actions = actions.clone()
        base_actions = torch.clamp(base_actions,-1,1)
        base_actions[:,:2] *= self.max_base_xy_vel # scale base xy velocity joint velocities
        base_actions[:,2] *= self.max_rot_vel

        if self.collect_action_history:
            # add actions to history track
            self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+self.num_actions] = actions
            self.obs_buf[:,self.obs_buf_idx+self._num_actions:self.obs_buf_idx+self.action_history_len] = self.action_history[:,:self.action_history_len-self.num_actions]
            self.action_history[:,:] = self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+self.action_history_len]
            self.obs_buf_idx += self.action_history_len

        self.tiago_handler.apply_base_actions(base_actions[:,:3])
        
        if self._obs_representation != 'bbox' and self._num_actions == 7:
            assert self._obs_representation == "voxel_downsample" #for now active vision only for voxel_downsmaple
            base_actions[:,3] *= self.max_head_pan_vel
            base_actions[:,4] *= self.max_head_tilt_vel
            self.tiago_handler.apply_head_actions(base_actions[:,3:5])

        base_tf = self.tiago_handler.get_robot_tf_world_frame()
        # Transform goal to robot frame
        inv_base_tf = torch.linalg.inv(base_tf)
        self._curr_goal_tf = torch.matmul(inv_base_tf,self._goal_tf)

        # Discrete Arm action:
        self._ik_fails[0] = 0
        if(actions[0,-2] > actions[0,-1]): # This is the arm decision variable TODO: Parallelize
            # Compute IK to self._curr_goal_tf
            curr_goal_pos = self._curr_goal_tf[0:3,3]
            curr_goal_quat = Rotation.from_matrix(self._curr_goal_tf[:3,:3].cpu()).as_quat()[[3, 0, 1, 2]]

            success, ik_positions, _ = self.tiago_handler._ik_solver.solve_ik_pos_tiago(des_pos=curr_goal_pos.cpu().numpy(),
                                                                                        des_quat=curr_goal_quat,
                                                                                        pos_threshold=self._goal_pos_threshold,
                                                                                        angle_threshold=self._goal_ang_threshold,
                                                                                        verbose=False)
            if success:
                self._is_success[0] = 1 # Can be used for reward, termination
                # set upper body positions
                self.tiago_handler.set_upper_body_positions(jnt_positions=torch.tensor(np.array([ik_positions]),dtype=torch.float,device=self._device))
            else:
                self._ik_fails[0] = 1 # Can be used for reward
    
    def reset_idx(self, env_ids, scene):
        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        # reset dof values
        self.tiago_handler.reset(indices,randomize=self._randomize_robot_on_reset)
                
        # reset the scene objects (randomize), get target end-effector goal/grasp as well as oriented bounding boxes of all other objects
        goal_tab_index = -1
        if self._obstacle_scale == 'random':
            # set random scale of tabular objects
            goal_tab_index = scene_utils.rescale_obstacle(self, scene, self._obstacles, self._obstacles_dimensions, self._obstacle_names,
                                                        self._tabular_obstacle_mask, self.tiago_handler.default_zero_env_path, self._device, True)        
        
        self._curr_grasp_obj, self._goals[env_ids], self._obj_bboxes = scene_utils.setup_tabular_scene(
                                self, self._obstacles, self._tabular_obstacle_mask[0:self._num_obstacles], self._grasp_objs,
                                self._obstacles_dimensions, self._grasp_objs_dimensions, self._world_xy_radius, self._device,
                                self._navigation_task, random_goal_obj=self._random_goal, scene_type=self._scene_type,
                                tab_index=goal_tab_index, bboxes=(self._obs_representation == 'bbox'))
        
        self._goal_tf = torch.zeros((4,4),device=self._device)
        goal_rot = Rotation.from_quat(np.array([self._goals[0,3+1].cpu(),self._goals[0,3+2].cpu(),self._goals[0,3+3].cpu(),self._goals[0,3].cpu()])) # Quaternion in scalar last format!!!
        self._goal_tf[:3,:3] = torch.tensor(goal_rot.as_matrix(),dtype=float,device=self._device)
        self._goal_tf[:,-1] = torch.tensor([self._goals[0,0], self._goals[0,1], self._goals[0,2], 1.0],device=self._device) # x,y,z,1
        self._curr_goal_tf = self._goal_tf.clone()
        self._goals_xy_dist = torch.linalg.norm(self.tiago_handler.get_robot_obs()[:,:2] - self._goal_tf[:2,3],dim=1)
        # Pitch visualizer by 90 degrees for aesthetics
        goal_viz_rot = goal_rot * Rotation.from_euler("xyz", [0,np.pi/2.0,0])
        self._goal_vizs.set_world_poses(indices=indices,positions=self._goals[env_ids,:3],
                orientations=torch.tensor(goal_viz_rot.as_quat()[[3, 0, 1, 2]],device=self._device).unsqueeze(dim=0))
        
        if self._face_goal_on_reset:
            #make robot look at the goal at each reset
            goal_translation = self._goal_tf[:,-1][:2]
            robot_translation = self.tiago_handler.get_robot_obs()[:,:2]
            start_look_dir = torch.tensor([1,0,0,1], dtype=torch.float, device=self._device)
            current_look_dir = torch.matmul(self.tiago_handler.get_robot_tf_world_frame(), start_look_dir)[:2]
            current_look_dir_rf = (current_look_dir - robot_translation).squeeze(0)
            goal_dir_rf = (goal_translation - robot_translation).squeeze(0)
            cos_theta = torch.dot(current_look_dir_rf, goal_dir_rf) / torch.mul(torch.norm(current_look_dir_rf), torch.norm(goal_dir_rf))
            cos_theta = torch.clamp(cos_theta,-1,1)
            theta = torch.acos(cos_theta)
            # check wether turn left or turn right
            if (-1*current_look_dir_rf[0] * goal_dir_rf[1] + current_look_dir_rf[1] * goal_dir_rf[0]) > 0:
                theta *= -1
            theta += self.tiago_handler.get_robot_obs()[0,2].squeeze(0)
            self.tiago_handler.set_base_positions(torch.hstack((self.tiago_handler.get_robot_obs()[:,:2].squeeze(0), theta)))
                
        if self.collect_action_history:
            self.action_history = torch.zeros((1, self.action_history_len), dtype=torch.float, device=self._device)
        if self.collect_reward_history:
            self.reward_history = torch.zeros((1, self.reward_history_len), dtype=torch.float, device=self._device)
        self.robot_pose_history = torch.zeros((1, self.robot_pose_history_len), dtype=torch.float, device=self._device)
        self.goal_pose_history = torch.zeros((1, self.goal_pose_history_len), dtype=torch.float, device=self._device)
        self.observation_history = torch.zeros((1, self.observation_history_len), dtype=torch.float, device=self._device)

        if self._obs_representation == 'bbox':
            self._obj_bboxes = torch.vstack((self._obj_bboxes, self._wall_bboxes))
            self._curr_obj_bboxes = self._obj_bboxes.clone()
        if self._obs_representation == 'keyframe_voxel':
            self._o3d_pcl_map.clear()
            self._keyframes = deque([None] * self._num_keyframes)
            # self._last_robot_obs = None
        if self._obs_representation == 'voxel_downsample':
            self._o3d_pcl_map.clear()
        if self._obs_representation == 'stack':
            self._pcl_counter = 0
        if self._obs_representation == 'tsdf':
            # self._tsdf = o3d.pipelines.integration.UniformTSDFVolume(length=self._wall_r*2,
            #                                                         resolution=self._tsdf_res,
            #                                                         sdf_trunc=4*self._voxel_size,
            #                                                         color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
            self._tsdf = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length = self._voxel_length,
                                                          sdf_trunc = self._voxel_length * 4,
                                                          color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
            self._goal_tf_tsdf_frame = torch.matmul(self._tsdf_tf, self._goal_tf)


        # bookkeeping
        self._is_success[env_ids] = 0
        self._ik_fails[env_ids] = 0
        self._collided[env_ids] = 0
        self.cleanup()

    def _check_robot_collisions(self):
        contact_headers, contact_data = get_physx_simulation_interface().get_contact_report()
        for contact_header in contact_headers:
            c_type = str(contact_header.type)
            actor0 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            if "LOST" not in c_type:
                ground_contact = ("defaultGroundPlane" in actor0) or ("defaultGroundPlane" in actor1)
                if not ground_contact:
                    tiago_contact = ("Tiago" in actor0) or ("Tiago" in actor1) or ("wall" in actor0) or ("wall" in actor1)
                    if tiago_contact:
                        goal_contact = (self._curr_grasp_obj.name in actor0) or (self._curr_grasp_obj.name in actor1)
                        if not goal_contact:
                            return True
        return False
    
    def _look_at_goal(self):
        hom_goal_coords = np.transpose(np.expand_dims(self._goal_tf[:,-1].cpu().detach().numpy(), axis=0))
        width, height, cam_intrinsics = self.tiago_handler.get_cam_intrinsics()
        cam_intrinsics = torch.nn.functional.pad(cam_intrinsics, (0,1,0,0)).cpu().detach().numpy()
        world_to_cam = np.linalg.inv(self.tiago_handler.get_cam_extrinsics_o3d())
        projection_matrix = np.matmul(cam_intrinsics, world_to_cam)
        goal_image_plane = np.matmul(projection_matrix, hom_goal_coords)
        goal_image_plane[:2,0] /= goal_image_plane[2,0]
        if (goal_image_plane[0,0] > 0 and goal_image_plane[0,0] < width) and (goal_image_plane[1,0] > 0 and goal_image_plane[1,0] < height):
            return True
        return False
    
    def _get_bboxes_robot_frame(self, robot_frame_tf, theta_robot_frame):
        # Transform all other object oriented bounding boxes to robot frame
        for obj_num in range(self._num_obstacles+self._num_grasp_objs-1):
            min_xy_vertex = torch.hstack(( self._obj_bboxes[obj_num,0:2], torch.tensor([0.0, 1.0],device=self._device) )).T
            max_xy_vertex = torch.hstack(( self._obj_bboxes[obj_num,2:4], torch.tensor([0.0, 1.0],device=self._device) )).T
            new_min_xy_vertex = torch.matmul(robot_frame_tf,min_xy_vertex)[0:2].T.squeeze()
            new_max_xy_vertex = torch.matmul(robot_frame_tf,max_xy_vertex)[0:2].T.squeeze()
            self._curr_obj_bboxes[obj_num,0:4] = torch.hstack(( new_min_xy_vertex, new_max_xy_vertex ))
            self._curr_obj_bboxes[obj_num,5] -=  theta_robot_frame.item()# new theta
        return self._curr_obj_bboxes


    def calculate_metrics(self) -> None:
        # assuming data from obs buffer is available (get_observations() called before this function)
        if self._check_robot_collisions(): # TODO: Parallelize
            # Collision detected. Give penalty and no other rewards
            self._collided[0] = 1
            self._is_success[0] = 0 # Success isn't considered in this case
            reward = torch.tensor(self._reward_collision, device=self._device)
            self._n_collisions += 1
        else:
            # Distance reward
            prev_goal_xy_dist = self._goals_xy_dist
            curr_goal_xy_dist = torch.linalg.norm(self.tiago_handler.get_robot_obs()[:,:2] - self._goal_tf[:2,3],dim=1)

            goal_xy_dist_reduction = (prev_goal_xy_dist - curr_goal_xy_dist).clone()
            reward = self._reward_dist_weight*goal_xy_dist_reduction
            # print(f"Goal Dist reward: {reward}")
            self._goals_xy_dist = curr_goal_xy_dist

            # IK fail reward (penalty)
            reward += self._reward_noIK*self._ik_fails

            if self._obs_representation != 'bbox' and self._num_actions == 7:
                # reward for looking at goal
                if not self._look_at_goal():
                    reward += self._reward_look_at_goal
                
                if self._obs_representation == 'voxel_downsample' or self._obs_representation == 'keyframe_voxel':
                    num_new_points = np.asarray(self._o3d_pcl_map.points).shape[0] - self._num_point_old_pcl
                    reward += self._reward_new_points*num_new_points

            # Success reward
            reward += self._reward_success*self._is_success
        # print(f"Total reward: {reward}")
        self.rew_buf[:] = reward
        self.extras[:] = self._is_success.clone() # Track success

        if self.collect_reward_history:
            # add reward to history track
            self.obs_buf[:,self.obs_buf_idx] = self.rew_buf.unsqueeze(0)
            self.obs_buf[:,self.obs_buf_idx+1:self.obs_buf_idx+self.reward_history_len] = self.reward_history[:,:self.reward_history_len-1]
            self.reward_history[:,:] = self.obs_buf[:,self.obs_buf_idx:self.obs_buf_idx+self.reward_history_len]
            self.obs_buf_idx += self.reward_history_len

    def is_done(self) -> None:
        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > np.pi / 2, 1, resets)
        # resets = torch.zeros(self._num_envs, dtype=int, device=self._device)
        # reset if success OR collided OR if reached max episode length
        resets = self._is_success.clone()
        if self._terminate_on_collision:
            resets = torch.where(self._collided.bool(), 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
