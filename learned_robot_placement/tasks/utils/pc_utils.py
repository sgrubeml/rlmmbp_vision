import numpy as np
import open3d as o3d
import torch

def distance_downsampling(self, num_points, pcl, extrinsics, dist_thresh=1):
    probs = np.zeros(pcl.shape[0])
    origin = extrinsics[:3,-1]
    sum_distances = 0
    for i in range(pcl.shape[0]):
        probs[i] = 1 / (dist_thresh*np.square(np.linalg.norm(origin - pcl[i])))
        sum_distances += probs[i]
    probs = probs / sum_distances
    sample_indices = np.random.choice(pcl.shape[0], num_points, False, probs)
    return pcl[sample_indices]

# use uniform downsampling
def uniform_downsampling(self, num_points, pcl):
    sample_indices = np.random.choice(np.arange(len(pcl)), num_points, replace=False)
    return pcl[sample_indices]

def pc_normalize(self, pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def voxel_downsample(self, pcl, o3d_pcl_map, pcl_size, device, downsample="uniform", extrinsics=None):    
    o3d_pcl = o3d.geometry.PointCloud()
    o3d_pcl.points = o3d.utility.Vector3dVector(pcl)
    
    # pcl map in world frame, merge pcls and voxel downsample
    o3d_pcl_map += o3d_pcl
    o3d_pcl_map = o3d_pcl_map.voxel_down_sample(voxel_size=0.01)
    #o3d.visualization.draw_geometries([o3d_pcl_map])
    # downsample point cloud to fixed size if bigger than number of observatins, if smaller add 0's 
    if len(o3d_pcl_map.points) > pcl_size:
        if downsample == 'distance':
            pcl_map_res = distance_downsampling(self, pcl_size, np.asarray(o3d_pcl_map.points, dtype=np.float32), extrinsics)
        if downsample == 'uniform':
            pcl_map_res = uniform_downsampling(self, pcl_size,  np.asarray(o3d_pcl_map.points, dtype=np.float32))
        if downsample == 'farthest':
            pcl_map_res = np.asarray(o3d_pcl_map.farthest_point_down_sample(pcl_size).points,  dtype=np.float32)
    else:
        pcl_map_res = np.zeros((pcl_size, 3), dtype='float32')
        pcl_map_res[:len(o3d_pcl_map.points),:3] = np.asarray(o3d_pcl_map.points, dtype=np.float32)
    if np.any(pcl_map_res):
        pcl_map_res = pc_normalize(self, pcl_map_res)
    return torch.from_numpy(pcl_map_res).flatten().unsqueeze(0).to(device)

def voxel_downsample_keyframes(self, pcl, keyframes_map, keyframes, pcl_size, device, downsample="uniform", dist=0.2, extrinsics=None, dist_thres=1):
    # for now always forget oldest keyframe and create voxel each time a new keyframe is added
    # check if keeping old frames in voxel and adding newest keyframe better
    o3d_pcl = o3d.geometry.PointCloud()
    o3d_pcl.points = o3d.utility.Vector3dVector(pcl)
    
    new_keyframe = add_if_keyframe_distance(self, keyframes, keyframes_map, o3d_pcl, dist)
    
    # if not new_keyframe:
    #     return None

    if new_keyframe:
        # pcl map in world frame, merge pcls and voxel downsample
        keyframes_map.clear()
        for i, keyframe in enumerate(keyframes):
            if keyframe is not None:
                keyframes_map += keyframes[i]

    downsampled_map = keyframes_map.voxel_down_sample(voxel_size=0.01)

    # downsample point cloud to fixed size if bigger than number of observatins, if smaller add 0's 
    if len(downsampled_map.points) > pcl_size:
        if downsample == 'distance':
            pcl_map_res = distance_downsampling(self, pcl_size, np.asarray(downsampled_map.points, dtype=np.float32), extrinsics, dist_thres)
        if downsample == 'uniform':
            pcl_map_res = uniform_downsampling(self, pcl_size,  np.asarray(downsampled_map.points, dtype=np.float32))
    else:
        pcl_map_res = np.zeros((pcl_size, 3), dtype='float32')
        pcl_map_res[:len(downsampled_map.points),:3] = np.asarray(downsampled_map.points, dtype=np.float32)

    return torch.from_numpy(pcl_map_res).flatten().unsqueeze(0).to(device)     


def add_if_keyframe_distance(self, keyframes, keyframes_map, new_frame, dist=0.2) -> bool:
    
    # # first frame always keyframe
    # if len(keyframes_map.points) == 0:
    #     keyframes.popleft()
    #     keyframes.append(new_frame)
    #     return True
    
    # dists = new_frame.compute_point_cloud_distance(keyframes_map)
    # dists = np.asarray(dists)
    # n_new_points = len(np.where(dists >= 0.01)[0])

    # if n_new_points < dist * len(new_frame.points):
    #     return False

    keyframes.popleft()
    keyframes.append(new_frame)
    return True

def add_if_keyframe_kd(self, keyframes, keyframes_map, new_frame) -> bool:
    
    # first frame always keyframe
    if len(keyframes_map.points) == 0:
        keyframes.popleft()
        keyframes.append(new_frame)
        return True
    
    kdtree = o3d.geometry.KDTreeFlann(keyframes_map)
    # search radius
    radius = 0.05
    # number of unseen points in new point cloud
    n_new_points = 0
    # iterate over new points, check if they are in the search radius of a an old point with a kd tree
    for i in range(len(new_frame.points)):
        [_, idx, _] = kdtree.search_radius_vector_3d(new_frame.points[i], radius)
        idx = np.asarray(idx, dtype=int)
        # if len(idx)==0, then new point
        if len(idx) == 0:
            n_new_points += 1

    if n_new_points < 0.2 * len(new_frame.points):
        return False

    keyframes.popleft()
    keyframes.append(new_frame)
    return True
    

def get_pcl_from_tsdf(self, pcl_size, tsdf, depth_img, intrinsic, extrinsics, tsdf_tf, device, downsample="uniform"):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=o3d.geometry.Image(np.empty_like(depth_img)),
            depth=o3d.geometry.Image(depth_img),
            depth_scale=1.0,
            depth_trunc=4.0,
            convert_rgb_to_intensity=False,
    )
    # transform extrinsics into tsdf frame (bottom right corner of room)
    extrinsic = np.matmul(np.linalg.inv(extrinsics), tsdf_tf)
    tsdf.integrate(rgbd, intrinsic, np.linalg.inv(extrinsics))
    pcl = tsdf.extract_voxel_point_cloud()

    points = np.asarray(pcl.points)
    distances = np.asarray(pcl.colors)[:, [0]]

    pcl_with_tsdf_feat = np.zeros((points.shape[0], 4), dtype=np.float32)
    pcl_with_tsdf_feat[:,:3] = points
    pcl_with_tsdf_feat[:,3:] = distances

    if downsample == 'distance':
        downsampled_pcl_with_tsdf_feat = distance_downsampling(self, pcl_size, pcl_with_tsdf_feat, extrinsics)
    if downsample == 'uniform':
        downsampled_pcl_with_tsdf_feat = uniform_downsampling(self, pcl_size,  pcl_with_tsdf_feat)

    #grid = np.zeros((1, tsdf.resolution, tsdf.resolution, tsdf_res_z), dtype=np.float32)

    # for idx, point in enumerate(points):
    #     i, j, k = np.floor(point / ((tsdf.length) / tsdf.resolution)).astype(int)
    #     if k < self._tsdf_res_z:
    #         grid[0, i, j, k] = distances[idx]
    return torch.from_numpy(downsampled_pcl_with_tsdf_feat).flatten().unsqueeze(0).to(self._device)



def visualize_tsdf(self, pcl):
    pcl = pcl.transform(self._tsdf_tf)

    import random
    from omni.isaac.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()
    draw.clear_points()
    
    pcl = np.asarray(pcl.points)
    colors = [(255,0,0,1)]*len(pcl)
    if type(colors) is tuple: colors = [colors]*len(pcl)
    voxel_size = (self._wall_r*2 ) / self._tsdf_res
    
    box_template = [np.array([-voxel_size / 2, -voxel_size / 2, -voxel_size / 2]),
                    np.array([-voxel_size / 2, -voxel_size / 2,  voxel_size / 2]),
                    np.array([-voxel_size / 2,  voxel_size / 2, -voxel_size / 2]),
                    np.array([-voxel_size / 2,  voxel_size / 2,  voxel_size / 2]),
                    np.array([ voxel_size / 2, -voxel_size / 2, -voxel_size / 2]),
                    np.array([ voxel_size / 2, -voxel_size / 2,  voxel_size / 2]),
                    np.array([ voxel_size / 2,  voxel_size / 2, -voxel_size / 2]),
                    np.array([ voxel_size / 2,  voxel_size / 2,  voxel_size / 2])]
    
    for point, color in zip(pcl, colors):
        m_m_m, m_m_p, m_p_m, m_p_p, p_m_m, p_m_p, p_p_m, p_p_p = point + box_template
        point_list_start =  [m_m_m, m_m_m, m_m_m, m_m_p, m_m_p, m_p_m, m_p_m, p_m_m, p_m_m, m_p_p, p_m_p, p_p_m]
        point_list_end =    [m_m_p, m_p_m, p_m_m, m_p_p, p_m_p, m_p_p, p_p_m, p_m_p, p_p_m, p_p_p, p_p_p, p_p_p]

        draw.draw_lines(point_list_start, point_list_end, [color]*len(point_list_start), [5]*len(point_list_start))

def add_pos_to_pcl_channel(self, pcl, pcl_size, max_sequence_length, pcl_counter, channels, device, downsample, extrinsics=None):
    if pcl.shape[0] < self.pcl_size:
            exp_pcl = np.zeros((self.pcl_size, 3), dtype='float32')
            exp_pcl[:pcl.shape[0],:3] = pcl
            pcl = exp_pcl
    if downsample == 'distance':
        pcl = distance_downsampling(self, pcl_size, pcl, extrinsics)
    if downsample == 'uniform':
        pcl = uniform_downsampling(self, pcl_size, pcl)
    if downsample == 'farthest':
        o3d_pcl = o3d.geometry.PointCloud()
        o3d_pcl.points = o3d.utility.Vector3dVector(pcl)
        pcl = np.asarray(o3d_pcl.farthest_point_down_sample(pcl_size).points,  dtype=np.float32)
    if np.any(pcl):
        pcl = pc_normalize(self, pcl)
    if channels == 4:
        pcl_pos_encoding = np.zeros((pcl_size, 4), dtype=np.float32)
        pcl_pos_encoding[:,:3] = pcl
        pcl_pos_encoding[:pcl_size,3:] = pcl_counter / max_sequence_length
        pcl_pos_encoding = torch.from_numpy(pcl_pos_encoding).float().flatten().unsqueeze(0).to(device)
        pcl_counter += 1
    else:
        pcl_pos_encoding = torch.from_numpy(pcl).flatten().float().unsqueeze(0).to(self._device)
    
    return pcl_pos_encoding, pcl_counter
