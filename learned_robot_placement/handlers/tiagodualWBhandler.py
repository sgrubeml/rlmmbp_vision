import torch

# check for motion planning system early on, as pinocchio has to be imported before SimApp to avoid dependency issues

from learned_robot_placement.handlers.base.tiagohandler import TiagoBaseHandler # might be lying somewhere else
from learned_robot_placement.robots.articulations.tiago_dual_holo import TiagoDualHolo # might be lying somewhere else
from learned_robot_placement.utils.files import get_usd_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp
from omni.isaac.core.utils.stage import get_current_stage
import omni.replicator.core as rep
import numpy as np
import open3d as o3d

# from omni.isaac.sensor import _sensor
# from omni.isaac.range_sensor._range_sensor import acquire_lidar_sensor_interface
from scipy.spatial.transform import Rotation
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import BaseSensor
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.physx import get_physx_interface, get_physx_simulation_interface
from pxr import Usd, UsdGeom, PhysxSchema



# Whole Body robot handler for the dual-armed Tiago robot
class TiagoDualWBHandler(TiagoBaseHandler):
    def __init__(self, env, move_group, use_torso, use_base, use_head, sim_config, num_envs, device, usd_path=None, intrinsics=None, motion_planner='pinocchio'):
                                                                                                                                            # 'lula'
        self._env = env
        self.motion_planner = motion_planner
        if motion_planner == 'pinocchio':
            from learned_robot_placement.tasks.utils.pinoc_utils import PinTiagoIKSolver
            self._ik_solver = PinTiagoIKSolver(move_group=move_group, include_torso=use_torso, include_base=use_base, include_head=use_head, max_rot_vel=100.0)
        elif motion_planner == 'lula':
            pass #TBA

        # self.task = task_class
        self._move_group = move_group
        self._use_torso = use_torso
        self._use_base = use_base
        self._use_head = use_head
        self._sim_config = sim_config
        self._num_envs = num_envs
        self._robot_positions = torch.tensor([0, 0, 0])  # placement of the robot in the world
        self._device = device
        self.intrinsics = intrinsics

        if usd_path is None:
            # self.usd_path = "omniverse://localhost/Isaac/Robots/Tiago/tiago_dual_holobase_zed.usd"
            self.usd_path = (get_usd_path() / 'tiago_dual_holobase' / 'tiago_dual_holobase_zed.usd').as_posix()
        else:
            self.usd_path = usd_path
        # Custom default values of arm joints
        # middle of joint ranges
        # self.arm_left_start =  torch.tensor([0.19634954, 0.19634954, 1.5708,
        #                                     0.9817477, 0.0, 0.0, 0.0], device=self._device)
        # self.arm_right_start = torch.tensor([0.19634954, 0.19634954, 1.5708,
        #                                     0.9817477, 0.0, 0.0, 0.0], device=self._device)
        # Start at 'home' positions
        self.arm_left_start = torch.tensor([0.0, 1.5708, 1.5708,
                                            1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        self.arm_right_start = torch.tensor([0.0, 1.5708, 1.5708,
                                             1.5708, 1.5708, -1.5708, 1.5708], device=self._device)
        self.gripper_left_start = torch.tensor([0.045, 0.045], device=self._device)  # Opened gripper by default
        self.gripper_right_start = torch.tensor([0.045, 0.045], device=self._device)  # Opened gripper by default
        self.torso_fixed_state = torch.tensor([0.25], device=self._device)

        self.default_zero_env_path = "/World/envs/env_0"

        # self.max_arm_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_rot_vel = torch.tensor(self._sim_config.task_config["env"]["max_rot_vel"], device=self._device)
        # self.max_base_xy_vel = torch.tensor(self._sim_config.task_config["env"]["max_base_xy_vel"], device=self._device)
        # Get dt for integrating velocity commands
        self.dt = torch.tensor(
            self._sim_config.task_config["sim"]["dt"] * self._sim_config.task_config["env"]["controlFrequencyInv"],
            device=self._device)

        # articulation View will be created later
        self.robots = None

        # joint names
        self._base_joint_names = ["X",
                                  "Y",
                                  "R"]
        self._torso_joint_name = ["torso_lift_joint"]
        self._arm_left_names = []
        self._arm_right_names = []
        for i in range(7):
            self._arm_left_names.append(f"arm_left_{i + 1}_joint")
            self._arm_right_names.append(f"arm_right_{i + 1}_joint")

        # Future: Use end-effector link names and get their poses and velocities from Isaac
        self.ee_left_prim = ["gripper_left_grasping_frame"]
        self.ee_right_prim = ["gripper_right_grasping_frame"]
        # self.ee_left_tf =  torch.tensor([[ 0., 0.,  1.,  0.      ], # left_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [-1., 0.,  0., -0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        # self.ee_right_tf = torch.tensor([[ 0., 0., -1.,  0.      ], # right_7_link to ee tf
        #                                  [ 0., 1.,  0.,  0.      ],
        #                                  [ 1., 0.,  0.,  0.196575],
        #                                  [ 0., 0.,  0.,  1.      ]])
        self._gripper_left_names = ["gripper_left_left_finger_joint", "gripper_left_right_finger_joint"]
        self._gripper_right_names = ["gripper_right_left_finger_joint", "gripper_right_right_finger_joint"]
        self._head_names = ["head_1_joint",
                            "head_2_joint"]
        # set starting position of head joints
        self.head_start = torch.tensor((0.0, -0.5), device=self._device)
        # values are set in post_reset after model is loaded
        self.base_dof_idxs = []
        self.torso_dof_idx = []
        self.arm_left_dof_idxs = []
        self.arm_right_dof_idxs = []
        self.gripper_left_dof_idxs = []
        self.gripper_right_dof_idxs = []
        self.head_dof_idxs = []
        self.upper_body_dof_idxs = []
        self.combined_dof_idxs = []

        # dof joint position limits
        self.torso_dof_lower = []
        self.torso_dof_upper = []
        self.arm_left_dof_lower = []
        self.arm_left_dof_upper = []
        self.arm_right_dof_lower = []
        self.arm_right_dof_upper = []
        self.head_dof_higher = []
        self.head_dof_lower = []

        self._camera_path = "/TiagoDualHolo/zed_camera/zed_camera_center/ZedCamera"
        self.camera_path = self.default_zero_env_path + self._camera_path

        self._obs_representation = self._sim_config.task_config["env"]["obs_representation"]

        if self._obs_representation != "bbox":
            self.pcl_size = self._sim_config.task_config["obs"]["visual"]["pcl_size"]

        

        # transforms from isaac sim camera coordinate convention (+x right, +y up, -z forward) to camera coordinate convention (+x right, +y down, +z forward)
        self._isaac_to_cam_tf = np.array([[1,0,0], [0,-1,-np.sin(np.pi)], [0, np.sin(np.pi), -1]])

    def _generate_camera(self):
        camera = Camera(prim_path=self.camera_path,
                        name=f"zed_camera",
                        frequency=120)
        if camera.is_valid():
            camera.initialize()
            if self._obs_representation != "tsdf":
                camera.add_pointcloud_to_frame()
            else:
                camera.add_distance_to_image_plane_to_frame()
            camera.add_semantic_segmentation_to_frame()
                # camera.add_instance_id_segmentation_to_frame()
            #camera.set_clipping_range(0.05, 4)
            self.head_camera = camera
            if self.intrinsics is not None:
                self.set_intrinsics(self.intrinsics)
        else:
            RuntimeError("Camera Path is not valid")

    def get_robot(self):
        #from pxr import Gf, UsdPhysics
        ## make it in task and use handler as getter for path
        #rigid_api = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(self.default_zero_env_path + "/TiagoDualHolo", name="TiagoDualHolo"))
        #rigid_api.CreateRigidBodyEnabledAttr(True)
        self.tiago = TiagoDualHolo(prim_path=self.default_zero_env_path + "/TiagoDualHolo", name="TiagoDualHolo",
                              usd_path=self.usd_path, translation=self._robot_positions)
        # Optional: Apply additional articulation settings
        self._sim_config.apply_articulation_settings("TiagoDualHolo", get_prim_at_path(self.tiago.prim_path),
                                                     self._sim_config.parse_actor_config("TiagoDualHolo"))
        # Add contact sensing to tiago
        contactReportAPI = PhysxSchema.PhysxContactReportAPI.Apply(self.tiago.prim)
        contactReportAPI.CreateThresholdAttr().Set(1)

    # call it in setup_up_scene in Task
    def create_articulation_view(self):
        self.robots = ArticulationView(prim_paths_expr="/World/envs/.*/TiagoDualHolo", name="tiago_dual_holo_view")
        return self.robots

    def post_reset(self):
        # reset that takes place when the isaac world is reset (typically happens only once)
        # add dof indexes
        self._set_dof_idxs()
        # set dof limits
        self._set_dof_limits()
        # set new default state for reset
        self._set_default_state()
        if self._obs_representation != "bbox":
            # generate camera after loading sim
            self._generate_camera()
        # get stage
        self._stage = get_current_stage()

    def _set_dof_idxs(self):
        [self.base_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._base_joint_names]
        [self.torso_dof_idx.append(self.robots.get_dof_index(name)) for name in self._torso_joint_name]
        [self.arm_left_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_left_names]
        [self.arm_right_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._arm_right_names]
        [self.gripper_left_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._gripper_left_names]
        [self.gripper_right_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._gripper_right_names]
        [self.head_dof_idxs.append(self.robots.get_dof_index(name)) for name in self._head_names]
        self.upper_body_dof_idxs = []
        if self._use_torso:
            self.upper_body_dof_idxs += self.torso_dof_idx

        if self._move_group == "arm_left":
            self.upper_body_dof_idxs += self.arm_left_dof_idxs
        elif self._move_group == "arm_right":
            self.upper_body_dof_idxs += self.arm_right_dof_idxs
        elif self._move_group == "both_arms":
            self.upper_body_dof_idxs += self.arm_left_dof_idxs + self.arm_right_dof_idxs
        else:
            raise ValueError('move_group not defined')
        # Future: Add end-effector prim paths
        self.combined_dof_idxs = self.base_dof_idxs + self.upper_body_dof_idxs

    def _set_dof_limits(self):  # dof position limits
        # (num_envs, num_dofs, 2)
        dof_limits = self.robots.get_dof_limits()
        dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        dof_limits_upper = dof_limits[0, :, 1].to(self._device)

        # set relevant joint position limit values
        self.torso_dof_lower = dof_limits_upper[self.torso_dof_idx]
        self.torso_dof_upper = dof_limits_upper[self.torso_dof_idx]
        self.arm_left_dof_lower = dof_limits_lower[self.arm_left_dof_idxs]
        self.arm_left_dof_upper = dof_limits_upper[self.arm_left_dof_idxs]
        self.arm_right_dof_lower = dof_limits_lower[self.arm_right_dof_idxs]
        self.arm_right_dof_upper = dof_limits_upper[self.arm_right_dof_idxs]
        # self.gripper_dof_lower = dof_limits_lower[self.gripper_idxs]
        # self.gripper_dof_upper = dof_limits_upper[self.gripper_idxs]
        self.head_lower = dof_limits_lower[self.head_dof_idxs]
        self.head_upper = dof_limits_upper[self.head_dof_idxs]

        # Holo base has no limits

    def _set_default_state(self):
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions
        jt_pos[:, self.torso_dof_idx] = self.torso_fixed_state
        jt_pos[:, self.arm_left_dof_idxs] = self.arm_left_start
        jt_pos[:, self.arm_right_dof_idxs] = self.arm_right_start
        jt_pos[:, self.gripper_left_dof_idxs] = self.gripper_left_start
        jt_pos[:, self.gripper_right_dof_idxs] = self.gripper_right_start
        jt_pos[:, self.head_dof_idxs] = self.head_start

        self.robots.set_joints_default_state(positions=jt_pos)

    def apply_actions(self, actions):
        # Actions are velocity commands
        # The first three actions are the base velocities
        self.apply_base_actions(actions=actions[:, :3])
        self.apply_upper_body_actions(actions=actions[:, 3:])

    def apply_upper_body_actions(self, actions):
        # Apply actions as per the selected upper_body_dof_idxs (move_group)
        # Velocity commands (rad/s) will be converted to next-position (rad) targets
        jt_pos = self.robots.get_joint_positions(joint_indices=self.upper_body_dof_idxs, clone=True)
        jt_pos += actions * self.dt  # create new position targets
        # self.robots.set_joint_position_targets(positions=jt_pos, joint_indices=self.upper_body_dof_idxs)
        # TEMP: Use direct joint positions
        self.robots.set_joint_positions(positions=jt_pos, joint_indices=self.upper_body_dof_idxs)
        if not self._use_torso:
            # Hack to avoid torso falling when it isn't controlled
            pos = self.torso_fixed_state.unsqueeze(dim=0)
            self.robots.set_joint_positions(positions=pos, joint_indices=self.torso_dof_idx)
        if self._move_group == "arm_left":  # Hack to avoid arm falling when it isn't controlled
            pos = self.arm_right_start.unsqueeze(dim=0)
            self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_right_dof_idxs)
        elif self._move_group == "arm_right":  # Hack to avoid arm falling when it isn't controlled
            pos = self.arm_left_start.unsqueeze(dim=0)
            self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_left_dof_idxs)
        elif self._move_group == "both_arms":
            raise NotImplementedError('both_arms not implemented')

    def apply_base_actions(self, actions):
        base_actions = actions.clone()
        # self.robots.set_joint_velocity_targets(velocities=base_actions, joint_indices=self.base_dof_idxs)
        # TEMP: Use direct joint positions
        jt_pos = self.robots.get_joint_positions(joint_indices=self.base_dof_idxs, clone=True)
        jt_pos += base_actions * self.dt  # create new position targets
        limits = (jt_pos[:, 2] > torch.pi)
        jt_pos[limits, 2] -= 2 * torch.pi
        limits = (jt_pos[:, 2] < -torch.pi)
        jt_pos[limits, 2] += 2 * torch.pi
        self.robots.set_joint_positions(positions=jt_pos, joint_indices=self.base_dof_idxs)

    def apply_head_actions(self, actions):
        head_actions = actions.clone()
        # param torso_position: torch.tensor of shape (2,)
        jt_pos = self.robots.get_joint_positions(joint_indices=self.head_dof_idxs, clone=True)
        jt_pos += head_actions * self.dt  # create new position targets
        self.robots.set_joint_positions(positions=jt_pos, joint_indices=self.head_dof_idxs)

    def set_upper_body_positions(self, jnt_positions):
        # Set upper body joints to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.upper_body_dof_idxs)
        # if not self._use_torso:
        #     # Hack to avoid torso falling when it isn't controlled
        #     pos = self.torso_fixed_state.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.torso_dof_idx)
        # if self._move_group == "arm_left": # Hack to avoid arm falling when it isn't controlled
        #     pos = self.arm_right_start.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_right_dof_idxs)
        # elif self._move_group == "arm_right": # Hack to avoid arm falling when it isn't controlled
        #     pos = self.arm_left_start.unsqueeze(dim=0)
        #     self.robots.set_joint_positions(positions=pos, joint_indices=self.arm_left_dof_idxs)
        # elif self._move_group == "both_arms":
        #     pass

    def set_base_positions(self, jnt_positions):
        # Set base joints to specific positions
        self.robots.set_joint_positions(positions=jnt_positions, joint_indices=self.base_dof_idxs)

    def get_robot_obs(self):
        # return positions and velocities of upper body and base joints
        combined_pos = self.robots.get_joint_positions(joint_indices=self.combined_dof_idxs, clone=True)
        # Base rotation continuous joint should be in range -pi to pi
        limits = (combined_pos[:, 2] > torch.pi)
        combined_pos[limits, 2] -= 2 * torch.pi
        limits = (combined_pos[:, 2] < -torch.pi)
        combined_pos[limits, 2] += 2 * torch.pi
        # NOTE: Velocities here will only be correct if the correct control mode is used!!
        combined_vel = self.robots.get_joint_velocities(joint_indices=self.combined_dof_idxs, clone=True)
        # Future: Add pose and velocity of end-effector from Isaac prims
        return torch.hstack((combined_pos, combined_vel))
    
    def get_robot_tf_world_frame(self):
        # Get current base positions
        base_joint_pos = self.get_robot_obs()[:,:3] # First three are always base positions
        base_tf = torch.zeros((4,4),device=self._device)
        base_tf[:2,:2] = torch.tensor([[torch.cos(base_joint_pos[0,2]), -torch.sin(base_joint_pos[0,2])],[torch.sin(base_joint_pos[0,2]), torch.cos(base_joint_pos[0,2])]]) # rotation about z axis
        base_tf[2,2] = 1.0 # No rotation here
        base_tf[:,-1] = torch.tensor([base_joint_pos[0,0], base_joint_pos[0,1], 0.0, 1.0]) # x,y,z,1
        return base_tf

    def get_arms_dof_pos(self):
        # (num_envs, num_dof)
        dof_pos = self.robots.get_joint_positions(clone=False)
        # left arm
        arm_left_pos = dof_pos[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_pos = dof_pos[:, self.arm_right_dof_idxs]
        return arm_left_pos, arm_right_pos

    def get_arms_dof_vel(self):
        # (num_envs, num_dof)
        dof_vel = self.robots.get_joint_velocities(clone=False)
        # left arm
        arm_left_vel = dof_vel[:, self.arm_left_dof_idxs]
        # right arm
        arm_right_vel = dof_vel[:, self.arm_right_dof_idxs]
        return arm_left_vel, arm_right_vel

    def get_base_dof_values(self):
        base_pos = self.robots.get_joint_positions(joint_indices=self.base_dof_idxs, clone=False)
        base_vel = self.robots.get_joint_velocities(joint_indices=self.base_dof_idxs, clone=False)
        return base_pos, base_vel

    def get_torso_dof_values(self):
        torso_pos = self.robots.get_joint_positions(joint_indices=self.torso_dof_idx, clone=False)
        torso_vel = self.robots.get_joint_velocities(joint_indices=self.torso_dof_idx, clone=False)
        return torso_pos, torso_vel

    def get_head_dof_values(self):
        head_pos = self.robots.get_joint_positions(joint_indices=self.head_dof_idxs, clone=False)
        head_vel = self.robots.get_joint_velocities(joint_indices=self.head_dof_idxs, clone=False)
        return head_pos, head_vel

    def set_intrinsics(self, int_coefficient):
        # param int_coefficient = [fx, fy, cx, cy]

        cam = self.head_camera
        self._set_cam_intrinsics(cam, int_coefficient[0], int_coefficient[1], int_coefficient[2], int_coefficient[3])

    def _set_cam_intrinsics(self, cam, fx, fy, cx, cy):
        focal_length = cam.get_focal_length()
        horiz_aperture = cam.get_horizontal_aperture()
        width, height = cam.get_resolution()
        # Pixels are square so we can do
        horiz_aperture_old = cam.get_horizontal_aperture()
        width_old, height_old = cam.get_resolution()
        width, height = round(cx * 2), round(cy * 2)

        # adjust aperture relative to with/height, as pixel size won't change
        horiz_aperture = width / width_old * horiz_aperture_old
        vert_aperture = height / width * horiz_aperture
        # should be the same as fy * vert_aperture / height
        focal_length = fx * horiz_aperture / width

        cam.set_horizontal_aperture(horiz_aperture)
        cam.set_vertical_aperture(vert_aperture)
        cam.set_focal_length(focal_length)
        cam.set_resolution([width, height])

    # only differns in input arguments in case you want to set it indirectly
    def _set_cam_intrinsics_other(self, focal_length, horizontal_aperture, width, height):
        cam = self.head_camera
        cam.set_focal_length(focal_length)
        cam.set_horizontal_aperture(horizontal_aperture)
        cam.set_resolution(width, height)

    def get_pointcloud(self):
        for i in range(5): self._env._world.render()
        
        data = self.head_camera.get_current_frame()
        point_cloud = data["pointcloud"]
        id_to_labels = data['semantic_segmentation']['info']['idToLabels']
        point_labels = data["pointcloud"]['info']['pointSemantic']

        if point_cloud is None:
            return np.zeros((self.pcl_size, 3), dtype='float32')
        else:
            # filter_labels_idx = []
            # for key, value in id_to_labels.items():
            #     if value['class'] == 'defaultgroundplane':# or 'wall' in value['class']:
            #         filter_labels_idx.append(int(key))
            # print(point_cloud['data'].size)
            # print(point_cloud['data'].shape)

            # filtered_idx = np.argwhere(np.isin(point_labels, filter_labels_idx)).ravel()
            # filtered_pcl = np.delete(point_cloud['data'], filtered_idx, axis=0)
            # if filtered_idx.size != 0 or point_cloud['data'].shape[0] != 0:
            #     filtered_pcl = np.delete(point_cloud['data'], filtered_idx, axis=0)
            # else:
            #     filtered_pcl = point_cloud['data']
            # # if filtered_pcl.shape[0] < self.pcl_size:
            # #     pcl = np.zeros((self.pcl_size, 3), dtype='float32')
            # #     pcl[:filtered_pcl.shape[0],:3] = filtered_pcl
            # # else:
            # # pcl = filtered_pcl

            filtered_pcl = point_cloud['data'][point_cloud['data'][:,2] > 0.01]

            if filtered_pcl.shape[0] < self.pcl_size:
                pcl = np.zeros((self.pcl_size, 3), dtype='float32')
                pcl[:filtered_pcl.shape[0],:3] = filtered_pcl
            else:
                pcl = filtered_pcl

            return filtered_pcl
    
    # get cam extrinsics in open3d cam convention (+x right, +y down, +z forward)
    def get_cam_extrinsics_o3d(self): 
        # camera_params = rep.annotators.get("CameraParams").attach(self.head_camera._render_product_path)
        # cam_world_to_local = camera_params.get_data()["cameraViewTransform"].reshape(4, 4)
        # cam_local_to_world = np.linalg.inv(cam_world_to_local).T
        # # transform isaac sim camera coordinate convention (+x right, +y up, -z forward) to camera coordinate convention (+x right, +y down, +z forward)
        # cam_local_to_world[:3,:3] = np.matmul(cam_local_to_world[:3,:3], self._isaac_to_cam_tf)
        trans, orient = BaseSensor.get_world_pose(self.head_camera)
        o3d_orient = torch.roll(orient,-1) # transform from isaac quat (w,x,y,z) to open3d quat(x,y,z,w)
        # change isaac sim camera coordinate convention (+x right, +y up, -z forward) to open3d camera coordinate convention (+x right, +y down, +z forward)
        orient = (Rotation.from_quat(o3d_orient.cpu().detach().numpy()) * Rotation.from_quat([1,0,0,0])).as_quat()
        extrinsic = TiagoDualWBHandler.pos_quat_to_hom(trans, orient)
        return extrinsic
        
    @staticmethod
    def pos_quat_to_hom(pos, quat):
        # pos: (3,) np array
        # quat: (4,) np array
        # returns: (4, 4) np array homogenous transformation matrix
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([ quat[0], quat[1], quat[2], quat[3]]).as_matrix() # Quaternion in scalar last format!!!
        T[:3, 3] = pos
        return T


    def get_cam_intrinsics(self):
        width, height = self.head_camera.get_resolution()
        cam_intrinsics_matrix = self.head_camera.get_intrinsics_matrix()
        return width, height, cam_intrinsics_matrix

    
    def get_depth_img(self, torch_tensor=False):
        for i in range(5): self._env._world.render()
        data = self.head_camera.get_current_frame()
        depth = data["distance_to_image_plane"]
        semantic_segmentation_mask = data["semantic_segmentation"]

        if depth is None:
            return None

        filter_ground_seg_idx = -1
        for key, value in semantic_segmentation_mask['info']['idToLabels'].items():
            if value['class'] == 'defaultgroundplane':
                filter_ground_seg_idx = key
                break

        depth = np.where(semantic_segmentation_mask['data'][:,:] != int(filter_ground_seg_idx), depth, 0)

        if torch_tensor:
            depth = torch.from_numpy(depth)

        return depth

    def reset(self, indices, randomize=False):
        num_resets = len(indices)
        joint_states = self.robots.get_joints_default_state()
        jt_pos = joint_states.positions.clone()
        jt_pos = jt_pos[0:num_resets]  # we need only num_resets rows
        if randomize:
            noise = torch_rand_float(-0.75, 0.75, jt_pos[:, self.upper_body_dof_idxs].shape, device=self._device)
            # Clip needed? dof_pos[:] = tensor_clamp(self.initial_dof_pos[env_ids] + dof_pos, self.dof_limits_lower, self.dof_limits_upper)
            # jt_pos[:, self.upper_body_dof_idxs] = noise
            jt_pos[:, self.upper_body_dof_idxs] += noise  # Optional: Add to default instead
        self.robots.set_joint_positions(jt_pos, indices=indices)
    