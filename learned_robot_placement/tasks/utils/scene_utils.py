# from typing import Optional
# import numpy as np
import os
import numpy as np
import torch
import random
import copy
import math
from pxr import Gf, UsdPhysics, PhysxSchema
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.utils.semantics import add_update_semantics
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from learned_robot_placement.utils.files import get_usd_path

# Utility functions to build a scene with obstacles and objects to be grasped (grasp objects)


def spawn_obstacle(self, usd_name, obst_name, prim_path, device, random_scale=False, long_obst=False):
    # Spawn Shapenet obstacle model from usd path
    object_usd_path = os.path.join(get_usd_path(),'Props','Shapenet',usd_name,'models','model_normalized.usd')
    add_reference_to_stage(object_usd_path, prim_path + "/obstacle/" + obst_name)
    scale = [0.01,0.01,0.01]
    if random_scale:
        scale_x = random.choice([0.01,0.015,0.02,0.025])
        scale_y = random.choice([0.01,0.015])
        scale = [scale_x, 0.01, scale_y]
    if long_obst:
        scale = [0.02,0.01,0.015]
    obj = GeometryPrim(
        prim_path=prim_path + "/obstacle/" + obst_name,
        name=obst_name,
        position= torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation= torch.tensor([0.707106, 0.707106, 0.0, 0.0], device=device), # Shapenet model may be downward facing. Rotate in X direction by 90 degrees,
        visible=True,
        scale=scale, # Has to be scaled down to metres. Default usd units for these objects is cms
        collision=True
    )
    # Attach rigid body and enable tight collision approximation
    obj.set_collision_approximation("convexDecomposition")
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(obj.prim_path))
    rigid_api.CreateRigidBodyEnabledAttr(True)
    # Add semantic information to object
    add_update_semantics(obj.prim, obst_name)
    return obj


def spawn_grasp_object(self, usd_name, grasp_obj_name, prim_path, device):
    # Spawn YCB object model from usd path
    object_usd_path = os.path.join(get_usd_path(),'Props','YCB','Axis_Aligned',usd_name+'.usd')
    add_reference_to_stage(object_usd_path, prim_path + "/grasp_obj/ycb_" + grasp_obj_name)

    obj = GeometryPrim(
        prim_path=prim_path + "/grasp_obj/ycb_" + grasp_obj_name,
        name=grasp_obj_name,
        position= torch.tensor([0.0, 0.0, 0.0], device=device),
        orientation= torch.tensor([0.707106, -0.707106, 0.0, 0.0], device=device), # YCB model may be downward facing. Rotate in X direction by -90 degrees,
        visible=True,
        scale=[0.01,0.01,0.01], # Has to be scaled down to metres. Default usd units for these objects is cms
        collision=True
    )
    # Attach rigid body and enable tight collision approximation
    obj.set_collision_approximation("convexDecomposition")
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(get_current_stage().GetPrimAtPath(obj.prim_path))
    rigid_api.CreateRigidBodyEnabledAttr(True)
    # Add semantic information to object
    add_update_semantics(obj.prim, grasp_obj_name)
    return obj

# TODO remove semantics, necessary???
def rescale_obstacle(self, scene, obstacles, obst_dimensions, obstacle_names, tabular_obstacle_mask, prim_path, device, tab_index=-1):
    from omni.isaac.core.utils.stage import get_current_stage
    if tab_index < 0:
        tab_index = np.random.choice(np.nonzero(tabular_obstacle_mask)[0])
    obst_name = obstacles[tab_index].name
    usd_name = ""
    for obst_usd_name in obstacle_names:
        if obst_usd_name in obst_name:
            usd_name = obst_usd_name
            break
    scene.remove_object(obst_name)
    obstacles.pop(tab_index)
    obst_dimensions.pop(tab_index)
    obst = spawn_obstacle(self, usd_name=usd_name, obst_name=obst_name, prim_path=prim_path, device=device, random_scale=True)
    scene.add(obst)
    obstacles.insert(tab_index, obst)
    obst_dimensions.insert(tab_index, scene.compute_object_AABB(obst.name))
    return tab_index


def setup_tabular_scene(self, obstacles, tabular_obstacle_mask, grasp_objs, obstacles_dimensions, grasp_objs_dimensions,
                        world_xy_radius, device, navigation_task, random_goal_obj=True, scene_type='simple',
                        tab_index=-1, bboxes=False):
    # Randomly arrange the objects in the environment. Ensure no overlaps, collisions!
    # Grasp objects will be placed on a random tabular obstacle.
    # Returns: target grasp object, target grasp/goal, all object's oriented bboxes
    # TODO: Add support for circular tables
    object_positions, object_yaws, objects_dimensions = [], [], []
    obst_aabboxes, grasp_obj_aabboxes = [], []
    robot_radius = 0.45 # metres. To exclude circle at origin where the robot (Tiago) is

    # Choose one tabular obstacle to place grasp objects on
    if tab_index < 0:
        tab_index = np.random.choice(np.nonzero(tabular_obstacle_mask)[0])

    # Place tabular obstacle at random location on the ground plane
    tab_xyz_size = obstacles_dimensions[tab_index][1] - obstacles_dimensions[tab_index][0]
    tab_z_to_ground = - obstacles_dimensions[tab_index][0,2]

    # polar co-ords
    tab_phi = np.random.uniform(-np.pi,np.pi)
    if navigation_task == "simple":
        tab_r = np.random.uniform(0,world_xy_radius - (np.max(tab_xyz_size[0:2]/2 + 0.01)))
    else:
        robot_phi = random.random() * 2 * math.pi
        robot_x, robot_y = math.cos(robot_phi) * (self._world_xy_radius+1.15), math.sin(robot_phi) * (self._world_xy_radius+1.15) #add one to avoid collision with obstacle at scene set
        robot_theta = random.uniform(0, 2*math.pi)
        robot_coords = np.array([robot_x, robot_y])
        self.tiago_handler.set_base_positions(torch.tensor((robot_x, robot_y, robot_theta), device=self._device))
        # robot_phi = torch.atan(robot_coords[0,1]/robot_coords[0,0]).cpu().detach().numpy()
        # robot starting position is on the circunference of world_xy_radius
        tab_r = np.random.uniform(0,world_xy_radius)
    tab_x, tab_y = tab_r*np.cos(tab_phi), tab_r*np.sin(tab_phi)
    tab_position = [tab_x,tab_y,tab_z_to_ground]
    tab_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
    tab_orientation = Rotation.from_euler("xyz", [torch.pi/2,0,tab_yaw], degrees=False).as_quat()[np.array([3,0,1,2])] 
    obstacles[tab_index].set_world_pose(position=torch.tensor(tab_position,dtype=torch.float,device=device),
                                        orientation=torch.tensor(tab_orientation,device=device, dtype=torch.float))
    
    # Store tabular obstacle position, orientation, dimensions and AABBox
    object_positions.append(tab_position)
    object_yaws.append(tab_yaw)
    objects_dimensions.append(obstacles_dimensions[tab_index])
    self._scene._bbox_cache.Clear()
    goal_obst_bbox = self._scene.compute_object_AABB(obstacles[tab_index].name)
    # if multiple obstacles, enlarge bounding box so that robot can navigate around obstacle without colliding with other obstacles
    if len(obstacles) > 1:
        goal_obst_bbox[0,0] -= 0.325
        goal_obst_bbox[1,0] += 0.325
        goal_obst_bbox[0,1] -= 0.325
        goal_obst_bbox[1,1] += 0.325
    obst_aabboxes.append(goal_obst_bbox)
    
    # now that tabular obstacle with goal object has been set, set robot around table if navigation_task == simple
    if navigation_task == "simple":
        robot_coords = place_robot_around_tabular_obstacle(self, robot_radius, obstacles_dimensions[tab_index], tab_yaw, tab_position, device)
    
    # Now we need to place all the other obstacles (without overlaps):
    for idx, _ in enumerate(obstacles):
        if (idx == tab_index) or navigation_task == "simple": continue # Skip this since we have already placed tabular obstacle
        obst_xyz_size = obstacles_dimensions[idx][1] - obstacles_dimensions[idx][0]
        obst_z_to_ground = - obstacles_dimensions[idx][0,2]
        
        while(1): # Be careful about infinite loops!
            # Place obstacle at random position and orientation on the ground plane
            # polar co-ords
            obst_phi = np.random.uniform(-np.pi,np.pi)
            # robot starting position is on the circunference of world_xy_radius
            # make sure that table position doesnt collide with robot
            obst_r = np.random.uniform(0, world_xy_radius)
            obst_x, obst_y = obst_r*np.cos(obst_phi), obst_r*np.sin(obst_phi)
            obst_position = [obst_x.item(),obst_y.item(),obst_z_to_ground.item()]
            obst_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
            obst_orientation = Rotation.from_euler("xyz", [torch.pi/2,0,obst_yaw], degrees=False).as_quat()[np.array([3,0,1,2])]
            obstacles[idx].set_world_pose(position=torch.tensor(obst_position,dtype=torch.float,device=device),
                                          orientation=torch.tensor(obst_orientation,dtype=torch.float,device=device))
            # compute new AxisAligned bbox
            self._scene._bbox_cache.Clear()
            obst_aabbox = self._scene.compute_object_AABB(obstacles[idx].name)
            obst_aabbox[0,0] -= 0.325
            obst_aabbox[1,0] += 0.325
            obst_aabbox[0,1] -= 0.325
            obst_aabbox[1,1] += 0.325
            # Check for overlap with all existing grasp objects
            overlap = False
            for other_aabbox in obst_aabboxes: # loop over existing AAbboxes
                obst_range = Gf.Range3d(Gf.Vec3d(obst_aabbox[0,0],obst_aabbox[0,1],obst_aabbox[0,2]),Gf.Vec3d(obst_aabbox[1,0],obst_aabbox[1,1],obst_aabbox[1,2]))
                other_obst_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
                intersec = Gf.Range3d.GetIntersection(obst_range, other_obst_range)
                if (not intersec.IsEmpty()):
                    overlap = True # Failed. Try another pose
                    break
            if (overlap):
                continue # Failed. Try another pose
            else:
                # Success. Add this valid AAbbox to the list
                obst_aabboxes.append(obst_aabbox)
                # Store obstacle position, orientation (yaw) and dimensions
                object_positions.append(obst_position)
                object_yaws.append(obst_yaw)
                objects_dimensions.append(obstacles_dimensions[idx])
                break

    # use obstacle that is the farthest to put objects on
    if navigation_task == "complex":
        positions_to_consider = np.array(object_positions)
        positions_to_consider = positions_to_consider[:,:2] # only need x-y
        tab_index = np.argmax(np.sum((positions_to_consider - robot_coords)**2, axis=1))
        tab_xyz_size = objects_dimensions[tab_index][1] - objects_dimensions[tab_index][0]
        tab_z_to_ground = - objects_dimensions[tab_index][0,2]
        tab_x, tab_y = object_positions[tab_index][0], object_positions[tab_index][1]
        tab_position = [tab_x,tab_y,tab_z_to_ground]
        tab_yaw = object_yaws[tab_index]

    # Place all grasp objects on the tabular obstacle (without overlaps)
    for idx, _ in enumerate(grasp_objs):
        grasp_obj_z_to_ground = - grasp_objs_dimensions[idx][0,2]
        while(1): # Be careful about infinite loops!
            # Add random orientation (yaw) to object
            grasp_obj_yaw = np.random.uniform(-np.pi,np.pi) # random yaw
            grasp_object_orientation = Rotation.from_euler("xyz", [torch.pi/2,0,grasp_obj_yaw], degrees=False).as_quat()[np.array([3,0,1,2])]
            # Place object at height of tabular obstacle and in the x-y range of the tabular obstacle
            # add small offset so that objects are not placed too close to the edge and dont fall off
            grasp_obj_x = tab_x + np.random.uniform((-tab_xyz_size[0]-grasp_objs_dimensions[idx][0,0])/2.0 + 0.1, (tab_xyz_size[0]-grasp_objs_dimensions[idx][1,0])/2.0 - 0.1)
            grasp_obj_y = tab_y + np.random.uniform((-tab_xyz_size[1]-grasp_objs_dimensions[idx][0,1])/2.0 + 0.1, (tab_xyz_size[1]-grasp_objs_dimensions[idx][1,1])/2.0 - 0.1)
            grasp_obj_z = tab_xyz_size[2] + grasp_obj_z_to_ground # Place on top of tabular obstacle
            grasp_obj_position = [grasp_obj_x,grasp_obj_y,grasp_obj_z]
            grasp_objs[idx].set_world_pose(position=torch.tensor(grasp_obj_position,dtype=torch.float,device=device),
                                           orientation=torch.tensor(grasp_object_orientation,dtype=torch.float,device=device)) # YCB needs X -90 deg rotation
            # compute new AxisAligned bbox
            self._scene._bbox_cache.Clear()
            grasp_obj_aabbox = self._scene.compute_object_AABB(grasp_objs[idx].name)
            # Check for overlap with all existing grasp objects
            overlap = False
            for other_aabbox in grasp_obj_aabboxes: # loop over existing AAbboxes
                grasp_obj_range = Gf.Range3d(Gf.Vec3d(grasp_obj_aabbox[0,0],grasp_obj_aabbox[0,1],grasp_obj_aabbox[0,2]),Gf.Vec3d(grasp_obj_aabbox[1,0],grasp_obj_aabbox[1,1],grasp_obj_aabbox[1,2]))
                other_obj_range = Gf.Range3d(Gf.Vec3d(other_aabbox[0,0],other_aabbox[0,1],other_aabbox[0,2]),Gf.Vec3d(other_aabbox[1,0],other_aabbox[1,1],other_aabbox[1,2]))
                intersec = Gf.Range3d.GetIntersection(grasp_obj_range, other_obj_range)
                if (not intersec.IsEmpty()):
                    overlap = True # Failed. Try another pose
                    break
            if (overlap):
                continue # Failed. Try another pose
            else:
                # Success. Add this valid AAbbox to the list
                grasp_obj_aabboxes.append(grasp_obj_aabbox)
                # Store grasp object position, orientation (yaw), dimensions
                object_positions.append(grasp_obj_position)
                object_yaws.append(grasp_obj_yaw)
                objects_dimensions.append(grasp_objs_dimensions[idx])
                break
    
    for idx in range(len(obstacles), len(obstacles) + len(grasp_objs)):
        object_yaws[idx] += tab_yaw # Add orientation that was just added to tabular obstacle
        if (object_yaws[idx] < -np.pi): object_yaws[idx] + 2*np.pi, # ensure within -pi to pi
        if (object_yaws[idx] >  np.pi): object_yaws[idx] - 2*np.pi, # ensure within -pi to pi
        # modify x-y positions of grasp objects accordingly
        curr_rel_x, curr_rel_y = object_positions[idx][0] - tab_position[0], object_positions[idx][1] - tab_position[1] # Get relative co-ords
        modify_x, modify_y = curr_rel_x*np.cos(tab_yaw) - curr_rel_y*np.sin(tab_yaw), curr_rel_x*np.sin(tab_yaw) + curr_rel_y*np.cos(tab_yaw)
        new_x, new_y = modify_x + tab_position[0], modify_y + tab_position[1]
        object_positions[idx] = [new_x, new_y, object_positions[idx][2]] # new x and y but z is unchanged
        obj_orientation = Rotation.from_euler("xyz", [torch.pi/2,0,object_yaws[idx]], degrees=False).as_quat()[np.array([3,0,1,2])]
        grasp_objs[idx - len(obstacles)].set_world_pose(position=torch.tensor(object_positions[idx],dtype=torch.float,device=device),
                                       orientation=torch.tensor(obj_orientation,dtype=torch.float,device=device))


    # All objects placed in the scene!
    # Pick one object to be the grasp object and compute its grasp:
    if random_goal_obj:
        if scene_type == 'complex': # TODO adjust
            # Don't pick the big objects as the goal object
            goal_obj_index = np.random.randint(len(grasp_objs)-3) # TODO Why -2?
        else:
            goal_obj_index = np.random.randint(len(grasp_objs))
    else:
        # Optional: try to select the most occluded object as the goal object
        # simply choose the one furthest away from origin
        positions_to_consider = np.array(object_positions)[len(obstacles):]
        positions_to_consider = positions_to_consider[:,:2] # only need x-y
        goal_obj_index = np.argmax(np.sum((positions_to_consider - robot_coords)**2, axis=1)) + len(obstacles)
    
    if scene_type == 'complex':
        #random roll and random pitch
        goal_roll = np.random.uniform(-np.pi,np.pi)
        goal_pitch = np.random.uniform(0,np.pi/2.0)
    else:
        # For now, generating only top grasps: no roll, pitch 90, same yaw as object
        goal_roll = 0.0 # np.random.uniform(-np.pi,np.pi)
        goal_pitch = np.pi/2.0 # np.random.uniform(0,np.pi/2.0)
    
    goal_yaw = object_yaws[goal_obj_index]
    goal_position = np.array(object_positions[goal_obj_index])
    goal_position[2] = (grasp_obj_aabboxes[goal_obj_index-len(obstacles)][1,2] + np.random.uniform(0.05,0.15)) # Add (random) z offset to object top (5 to 15 cms)
    goal_orientation = Rotation.from_euler("xyz", [goal_roll,goal_pitch,goal_yaw], degrees=False).as_quat()[np.array([3,0,1,2])]
    goal_pose = torch.hstack(( torch.tensor(goal_position,dtype=torch.float,device=device),
                        torch.tensor(goal_orientation,dtype=torch.float,device=device)))
    
    object_oriented_bboxes = None
        
    if bboxes:
        object_oriented_bboxes = get_bboxes(self, object_positions, objects_dimensions, object_yaws, goal_obj_index, device)

    # Remove the goal object from obj_positions and yaws list (for computing oriented bboxes)
    del object_positions[goal_obj_index], object_yaws[goal_obj_index]

    return grasp_objs[goal_obj_index-len(obstacles)], goal_pose, object_oriented_bboxes

def place_robot_around_tabular_obstacle(self, robot_radius, obstacle_dimension, object_yaw, object_position, device):
        # bounding box will be used to place robot around table, enlarge bounding box dimensions to avoid collision on reset
        goal_obj_dim = copy.deepcopy(obstacle_dimension)
        goal_obj_dim[0][0:2] -= robot_radius
        goal_obj_dim[1][0:2] += robot_radius
        # get all 4 vertices of the bounding box
        goal_obj_oriented_bbox = _compute_bboxes(self, [object_position], [goal_obj_dim], [object_yaw], 0, device, True)
        # position robot randomly around tabular obstacle
        # first, sample one edge along the perimeter of the bounding box at which to place the robot
        edge_idx_start = np.random.choice([0,2,4,6])
        x1, y1 = goal_obj_oriented_bbox[edge_idx_start], goal_obj_oriented_bbox[edge_idx_start + 1]
        x2, y2 = goal_obj_oriented_bbox[(edge_idx_start + 2) % 8], goal_obj_oriented_bbox[(edge_idx_start + 3) % 8]
        norm = torch.sqrt(torch.square(x2 - x1) + torch.square(y2 - y1))
        t = random.random()
        # compute unit vector between the two vertices of the edge
        robot_x = x1 + (t * (x2 - x1)) / norm
        robot_y = y1 + (t * (y2 - y1)) / norm
        robot_theta = random.uniform(0, 2*math.pi)
        self.tiago_handler.set_base_positions(torch.tensor((robot_x, robot_y, robot_theta), device=self._device))
        return np.array([robot_x, robot_y])

def get_bboxes(self, object_positions, objects_dimensions, object_yaws, goal_obj_index, device):
    object_oriented_bboxes = None
    # Compute oriented bounding boxes for all remaining objects
    for idx in range(len(object_positions)):
        if idx == goal_obj_index:
            continue
        # Oriented bbox: x0, y0, x1, y1, z_to_ground, yaw
        oriented_bbox = _compute_bboxes(self, object_positions, objects_dimensions, object_yaws, idx, device)
        if object_oriented_bboxes is None:
            object_oriented_bboxes = oriented_bbox
        else:
            object_oriented_bboxes = torch.vstack(( object_oriented_bboxes, oriented_bbox ))
        
    return object_oriented_bboxes

def _compute_bboxes(self, object_positions, objects_dimensions, object_yaws, idx, device, all_corners=False):
    bbox_tf = np.zeros((3,3))
    bbox_tf[:2,:2] = np.array([[np.cos(object_yaws[idx]), -np.sin(object_yaws[idx])],[np.sin(object_yaws[idx]), np.cos(object_yaws[idx])]])
    bbox_tf[:,-1] = np.array([object_positions[idx][0], object_positions[idx][1], 1.0]) # x,y,1
    min_xy_vertex = np.array([[objects_dimensions[idx][0,0],objects_dimensions[idx][0,1],1.0]]).T
    max_xy_vertex = np.array([[objects_dimensions[idx][1,0],objects_dimensions[idx][1,1],1.0]]).T
    new_min_xy_vertex = (bbox_tf @ min_xy_vertex)[0:2].T.squeeze()
    new_max_xy_vertex = (bbox_tf @ max_xy_vertex)[0:2].T.squeeze()
    z_top_to_ground = object_positions[idx][2] + objects_dimensions[idx][1,2] # z position plus distance to object top
    if all_corners:
        third_vertex = np.array([[objects_dimensions[idx][0,0],objects_dimensions[idx][1,1],1.0]]).T
        fourth_vertex = np.array([[objects_dimensions[idx][1,0],objects_dimensions[idx][0,1],1.0]]).T
        new_third_vertex = (bbox_tf @ third_vertex)[0:2].T.squeeze()
        new_fourth_vertex = (bbox_tf @ fourth_vertex)[0:2].T.squeeze()
        oriented_bbox = torch.tensor([new_min_xy_vertex[0], new_min_xy_vertex[1],
                                      new_third_vertex[0], new_third_vertex[1],
                                      new_max_xy_vertex[0], new_max_xy_vertex[1],
                                      new_fourth_vertex[0], new_fourth_vertex[1],
                                      z_top_to_ground, object_yaws[idx],] ,dtype=torch.float,device=device)
    else:
        # Oriented bbox: x0, y0, x1, y1, z_to_ground, yaw
        oriented_bbox = torch.tensor([new_min_xy_vertex[0], new_min_xy_vertex[1],
                                      new_max_xy_vertex[0], new_max_xy_vertex[1],
                                      z_top_to_ground, object_yaws[idx],] ,dtype=torch.float,device=device)
    return oriented_bbox
