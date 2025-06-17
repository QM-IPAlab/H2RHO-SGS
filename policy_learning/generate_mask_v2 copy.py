# generate hand and object mask from trajectory 
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.pyplot as plt
from plyfile import PlyData

def load_pointcloud_from_ply(file_path):
    """Load point cloud data from a PLY file."""
    plydata = PlyData.read(file_path)
    # Extract x, y, z coordinates from the vertex data
    points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    # import trimesh
    # pc = trimesh.load(file_path)
    # pc_np = np.array(pc.vertices)

    return points

def get_mask(pose_file, hand_ply_file, object_pcl_file,output_dir):
    """
    Generate projected images for hand and object point clouds based on a given pose.

    Parameters:
        pose_file (str): Path to the .npy file containing the pose matrix.
        hand_ply_file (str): Path to the .ply file for the hand point cloud.
        object_pcl_file (str): Path to the .ply file for the object point cloud.
        output_dir (str): Directory to save the generated images.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    extrinsics =  np.load(pose_file)
    # print(extrinsics)
    # extrinsics = np.array([
    #                                 [-0.9286197279370463,  0.28499809028787526, -0.23757417666571096,  0.35575658082962036],
    #                                 [-0.19175885377662133,  0.17951273203290213,  0.9648853408754638, -1.0083527565002441],
    #                                 [ 0.31763806901000186,  0.9415685145385586, -0.1120481572595004,  0.9933208227157593],
    #                                 [0,0,0,1]
    #                             ])
    extrinsics = np.linalg.inv(extrinsics)                           


    hand_ply  = load_pointcloud_from_ply(hand_ply_file)
    object_ply = load_pointcloud_from_ply(object_pcl_file)
 
    # Extract intrinsic parameters
    width=640
    height=480
    fx=621
    fy=621
    cx=302
    cy=236
    depth_scale = 1

    # Transform point cloud using extrinsics
    hand_ply_homogeneous = np.hstack((hand_ply, np.ones((hand_ply.shape[0], 1))))  # Convert to homogeneous coordinates
    hand_transformed_points = (extrinsics @ hand_ply_homogeneous.T).T[:, :3]  # Apply extrinsic transformation
    object_ply_homogeneous = np.hstack((object_ply, np.ones((object_ply.shape[0], 1))))  # Convert to homogeneous coordinates
    object_transformed_points = (extrinsics @ object_ply_homogeneous.T).T[:, :3]  # Apply extrinsic transformation
    z_h = hand_transformed_points[:, 2] / depth_scale
    z_o = object_transformed_points[:,2] / depth_scale

    # hand Project points onto image plane
    x_proj_h = (hand_transformed_points[:, 0] * fx / z_h) + cx
    y_proj_h = (hand_transformed_points[:, 1] * fy / z_h) + cy
    valid_points_h = (x_proj_h >= 0) & (x_proj_h < width) & (y_proj_h >= 0) & (y_proj_h < height)
    x_proj_valid_h = x_proj_h[valid_points_h].astype(int)
    y_proj_valid_h = y_proj_h[valid_points_h].astype(int)
   
    

    # object Project points onto image plane
    x_proj_o = (object_transformed_points[:, 0] * fx / z_o) + cx
    y_proj_o = (object_transformed_points[:, 1] * fy / z_o) + cy
    valid_points_o = (x_proj_o >= 0) & (x_proj_o < width) & (y_proj_o >= 0) & (y_proj_o < height)
    x_proj_valid_o = x_proj_o[valid_points_o].astype(int)
    y_proj_valid_o = y_proj_o[valid_points_o].astype(int)



    hand_mask = np.zeros((height, width), dtype=np.uint8)
    hand_mask[y_proj_valid_h, x_proj_valid_h] = 255  # Set points to white (255) on the mask
    # Save the hand mask image
    hand_mask_image = Image.fromarray(hand_mask)
    hand_mask_image.save(os.path.join(output_dir, 'hand_mask.png'))

    object_mask = np.zeros((height, width), dtype=np.uint8)
    object_mask[y_proj_valid_o, x_proj_valid_o] = 255  # Set points to white (255) on the mask
    # Save the hand mask image
    object_mask_image = Image.fromarray(object_mask)
    object_mask_image.save(os.path.join(output_dir, 'object_mask.png'))


    return 0

   
# pose = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_155021/12_frame/trajectory_v4/0/0_0.npy'
# hand_ply_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_155021/12_frame/handover_3D/hand.ply'
# object_pcl_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_155021/12_frame/handover_3D/object.ply'
# output_dir= '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_155021/12_frame/mask'
pose = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_153453/60_frame/trajectory_v4/3/3_0.npy'
hand_ply_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_153453/60_frame/handover_3D/hand.ply'
object_pcl_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_153453/60_frame/handover_3D/object.ply'
output_dir= '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_153453/60_frame/mask'
get_mask(pose,hand_ply_file,object_pcl_file,output_dir)