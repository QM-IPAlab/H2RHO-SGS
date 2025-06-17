# 根据给定的步长在初始位姿和目标位姿之间进行采样。
# 先仅改变位置，直到距离目标点小于a米，然后改变位置和旋转。
# 距离target pose最近pre grasp, rotation== target pose
# pre grasp is same direction to object with grasp pose
# sample poses always face to object, if it is more than 40 degree, it will chage the rotation,to let is face the object
# initial pose cannot be too far away from training camera
import argparse
import grasp_estimator_yik
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
from utils import utils
from data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import trimesh
from plyfile import PlyData
import argparse
import os
import shutil
import copy
from scipy.spatial.transform import Rotation as R

def parse_colmap_images(file_path):
    """
    Parses COLMAP images.txt file to extract camera poses and convert them to world coordinates.
    
    Args:
        file_path (str): Path to the COLMAP images.txt file.
    
    Returns:
        numpy.ndarray: An (N x 4 x 4) NumPy array containing camera poses in world coordinates.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    camera_poses = []
    for line in lines:
        # Skip comments and empty lines
        if line.startswith('#') or not line.strip():
            continue

        # COLMAP image format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        elements = line.split()
        if len(elements) < 8:  # Ensure it's a valid image line
            continue

        # Extract quaternion (QW, QX, QY, QZ) and translation (TX, TY, TZ)
        qw, qx, qy, qz = map(float, elements[1:5])
        tx, ty, tz = map(float, elements[5:8])

        # Convert quaternion to rotation matrix
        rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()

        # Create 4x4 transformation matrix (world to camera)
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = [tx, ty, tz]

        # Invert the transformation to get camera to world
        camera_to_world = np.linalg.inv(world_to_camera)

        # Store the camera-to-world pose
        camera_poses.append(camera_to_world)

    # Convert to NumPy array
    camera_poses = np.array(camera_poses)

    return camera_poses

def adjust_rotation_to_face_object(new_pose, object_position):
    """
    调整给定位姿的旋转矩阵，使其Z轴朝向物体位置。
    """
    # 计算方向向量
    direction_to_object = object_position - new_pose[:3, 3]
    direction_to_object /= np.linalg.norm(direction_to_object)  # 单位化

    current_z_axis = new_pose[:3, 2]
    dot_product = np.dot(current_z_axis, direction_to_object)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    if angle_deg > 40.0:
            z_axis = direction_to_object
            temp_x_axis = np.array([1, 0, 0]) if not np.isclose(z_axis[0], 1.0) else np.array([0, 1, 0])
            x_axis = np.cross(temp_x_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)

            y_axis = np.cross(z_axis, x_axis)
            new_rotation = np.column_stack((x_axis, y_axis, z_axis))
            new_pose[:3, :3] = new_rotation

    return new_pose, angle_deg > 40.0

def is_too_close(new_pose, point_cloud, min_distance):
    """检查新位姿与点云中任一点的距离是否小于最小距离"""
    new_position = new_pose[:3, 3]
    distances = np.linalg.norm(point_cloud - new_position, axis=1)
    return np.any(distances < min_distance)

def linear_interpolate(start, end, t):
    """线性插值函数"""
    return (1 - t) * start + t * end

def slerp(start_rot, end_rot, t):
    """球面线性插值函数"""
    key_rots = R.from_matrix([start_rot, end_rot])
    # key_rots = R.random(2, random_state=2342345)
    key_times = [0, 1] 

    # # 使用 Slerp 进行插值
    slerp_obj = Slerp(key_times, key_rots)

    times = [t]
    interp_rots = slerp_obj(times)
    return interp_rots.as_matrix()

def sample_poses(start_pose, target_pose, handover_pcl, step_distance, min_distance_pointcloud, near_dis, min_distance):
    """
    根据给定的步长在初始位姿和目标位姿之间进行采样。
    最后一个点是 pre_grasp_pose，位置距离 target_pose 为 min_distance，旋转与 target_pose 一致。
    """
    # 提取位置和旋转部分
    start_position = start_pose[:3, 3]
    target_position = target_pose[:3, 3]

    start_rotation = copy.deepcopy(start_pose[:3, :3])
    target_rotation = copy.deepcopy(target_pose[:3, :3])

    # 计算到 pre_grasp_pose 的目标位置
    # 计算抓取姿势的 z 轴（工具方向）
    z_axis = copy.deepcopy(target_rotation[:, 2])  # 第三列表示 z 轴方向

    # 计算预抓取的平移位置
    pre_grasp_translation = copy.deepcopy(target_position) - min_distance * z_axis

    # 构造预抓取的 4x4 齐次变换矩阵
    pre_grasp_pose = np.eye(4)
    pre_grasp_pose[:3, :3] = copy.deepcopy(target_rotation)  # 旋转矩阵保持不变
    pre_grasp_pose[:3, 3] = pre_grasp_translation  # 更新平移向量
    

    # direction = copy.deepcopy(target_position)
    # direction /= np.linalg.norm(direction)  # 单位化方向向量
    # pre_grasp_position = target_position + direction * min_distance

    # 计算总距离（从 start 到 pre_grasp）
    total_distance = np.linalg.norm(pre_grasp_translation - start_position)

    # 计算需要的插值步数
    num_steps = int(np.ceil(total_distance / step_distance))

    # 生成采样点
    sampled_poses = [start_pose]
    begin_rotate = 2000
    for i in range(1, num_steps + 1):
        t = i / num_steps

        # 线性插值计算当前位置
        position = linear_interpolate(start_position, pre_grasp_translation, t)

        if np.linalg.norm(position - target_position) > near_dis:
            # 仅改变位置
            rotation = start_rotation
        else:
            # 改变位置和旋转
            if begin_rotate == 2000 or begin_rotate == 3000:
                begin_rotate = copy.deepcopy(i)
                num_steps_rotate = num_steps - begin_rotate 
            t_rotate =  (i - begin_rotate )/num_steps_rotate 
            if np.isclose(t_rotate,1.0):
                rotation = target_rotation
            else:
                rotation = slerp(start_rotation, target_rotation, t_rotate)

        # 构建新的 4x4 位姿矩阵
        sampled_pose = np.eye(4)
        sampled_pose[:3, 3] = position
        sampled_pose[:3, :3] = rotation

        object_position = [0,0,0]
        sampled_pose, adjusted = adjust_rotation_to_face_object(sampled_pose, object_position)

        if adjusted:
            if begin_rotate == 2000 :
                start_rotation = sampled_pose[:3, :3]
                begin_rotate = 2000
            else:
                start_rotation = sampled_pose[:3, :3]
                begin_rotate = 3000

                   

        # 检查是否与点云太近c
        if is_too_close(sampled_pose, handover_pcl, min_distance_pointcloud):
            # print(f"Sampled pose is too close to hand or object at step {i}, t={t}.")
            continue
        else:
            sampled_poses.append(sampled_pose)

    return sampled_poses


def sample_initial_pose_sphere(target_pose_o, radius, num_samples,  handover_pcl, max_rotation_angle, min_distance, dir_hono, mean, train_camera,threshold_train_cam ):
    samples = []
    i = 0
    j = 0
    target_rotation = target_pose_o[:3, :3]
    
    for rt_cam in train_camera:
        rt_cam[:3, 3] -= mean
    
    while i < num_samples:
        j = j+1
        if j>200000:
            break
        # 生成随机方向
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)

        direction_to_object = -copy.deepcopy(mean)
        direction_to_object /= np.linalg.norm(mean)
        
        # 球面坐标转笛卡尔坐标
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        # 生成新的位姿
        new_pose = copy.deepcopy(target_pose_o)
        new_pose[:3, 3] += np.array([x, y, z])  # 更新平移部分

        if is_too_close(new_pose, handover_pcl, min_distance):
            continue

        direction_to_object =  - new_pose[:3, 3]
        direction_to_object /= np.linalg.norm(direction_to_object)  # 单位化
        z_axis = direction_to_object  # 新位姿的Z轴朝向物体
        x_axis = np.cross(np.array([0, 0, 1]), z_axis)  # 假设初始Z轴为[0,0,1]
        x_axis /= np.linalg.norm(x_axis)  # 单位化
        y_axis = np.cross(z_axis, x_axis)  # Y轴垂直于X和Z轴

        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))  # 构建旋转矩阵
        new_pose[:3, :3] = rotation_matrix  # 更新旋转部分

        # # new_pose移动至target_pose_o 的方向，不能和target_pose_o->object 的rotation方向大于60度
        # xyz轴不是直直的new_pose[0,3] * target_pose_w[0,3]) < 0 or (new_pose[1,3] * target_pose_w[1,3]) < 0 or (new_pose[3,3] * target_pose_w[3,3]) < 0
        # dir_o2p = new_pose[:3, 3] - target_pose_o[:3, 3]
        # dir_pose = target_pose_o[:3, 3]
        # a1 = np.array(dir_o2p)
        # b1 = np.array(dir_pose)
        # dot_product = np.dot(a1, b1)
        # norm_a1 = np.linalg.norm(a1)
        # norm_b1 = np.linalg.norm(b1)
        # cos_theta1 = dot_product / (norm_a1 * norm_b1)
        # angle_rad1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))  # 使用 clip 防止数值误差
        # angle_deg1 = np.degrees(angle_rad1)  
        
        current_z_axis = new_pose[:3, 2]
        target_z_axis = target_pose_o[:3,2]
        dot_product = np.dot(current_z_axis, target_z_axis)
        angle_rad_initial_target = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg_initial_target = np.degrees(angle_rad_initial_target)

        # not the same side with hand
        dir_sampleono = new_pose[:3, 3] 
        a = np.array(dir_sampleono)
        b = np.array(dir_hono)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_theta = dot_product / (norm_a * norm_b)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 使用 clip 防止数值误差
        angle_deg = np.degrees(angle_rad)  

        # angle_deg>90 or target_pose_w[0, 3]*dir_o2p[0]<0 or target_pose_w[1, 3]*dir_o2p[1]<0 or target_pose_w[2, 3]*dir_o2p[2]<0
      
        # if(angle_deg <120 ):  or angle_deg1 > max_rotation_angle
        if(angle_deg <120  or angle_deg_initial_target > max_rotation_angle):
            continue

        #need to be close to origin camera
        train_positions = copy.deepcopy(train_camera[:, :3, 3])   # 训练相机的位置 (N x 3)
        new_position = copy.deepcopy(new_pose[:3, 3] )             # 新相机的位置 (1 x 3)
        # 计算 new_pose 和每个 train camera 的欧几里得距离
        distances = np.linalg.norm(train_positions - new_position, axis=1)
        if np.all(distances > threshold_train_cam):
            continue

        else:
            i += 1
            # 给new pose设计初始的旋转 在原始的旋转上加上一些小的扰动
            np_random_rotation = np.random.uniform(-1,1,size = (3,))
            samples.append(new_pose)
    print(j)
    return samples


def readPly(object_pcl, hand_pcl, ply_file, whole_scene):
    plydata = PlyData.read(ply_file)
    pc = plydata['vertex'].data
    points = pc[['x', 'y', 'z']]
    points_array = np.array(points.tolist())
    pc = points_array.astype(np.float32)
    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    pc_colors[:, :] = [0, 0, 255]

    plydata_whole = PlyData.read(whole_scene)
    pc_whole_data = plydata_whole['vertex'].data
    pc_whole_colors = np.array(pc_whole_data[['red', 'green', 'blue']])
    pc_whole_colors = np.array(pc_whole_colors.tolist())
    pc_whole_colors = pc_whole_colors.astype(np.uint8)
    points_whole = pc_whole_data[['x', 'y', 'z']]
    points_array_whole = np.array(points_whole.tolist())
    pc_whole = points_array_whole.astype(np.float32)
   
    plydata_hand = PlyData.read(hand_pcl)
    pc_h = plydata_hand['vertex'].data
    points_h = pc_h[['x', 'y', 'z']]
    points_array_h = np.array(points_h.tolist())
    pc_h = points_array_h.astype(np.float32)
    mean_hand = np.mean(pc_h, axis=0)
   
    plydata_o = PlyData.read(object_pcl)
    pc_o = plydata_o['vertex'].data
    points_o = pc_o[['x', 'y', 'z']]
    points_array_o = np.array(points_o.tolist())
    pc_o = points_array_o.astype(np.float32)

    mean = np.mean(pc_o, axis=0)
    pc_o = pc_o - mean
    pc_h = pc_h - mean
    pc = pc - mean
    pc_whole = pc_whole -mean

    dir_hono =  mean_hand - mean
    return pc, pc_h, pc_o, mean, pc_whole, pc_whole_colors, dir_hono,pc_colors


def main(args):
    sample_initial_radius = args.sample_initial_radius
    num_samples = args.num_samples
    max_rotation_angle = args.max_rotation_angle
    min_distance_initial_pose = args.min_distance_initial_pose
    step_distance = args.step_distance
    min_distance_sample_pose = args.min_distance_sample_pose
    min_distance_pointcloud = args.min_distance_pointcloud
    near_dis = args.near_dis
    threshold_train_cam = args.threshold_train_cam

    # 以物体为原点，主相机为坐标系方向，的坐标系里
    base_dir = args.base_dir
    object_pcl = os.path.join(base_dir,'handover_3D', 'object.ply')
    hand_pcl = os.path.join(base_dir, 'handover_3D','hand.ply')
    ply_file = os.path.join(base_dir,'handover_3D' ,'handover.ply')
    whole_scene = os.path.join(base_dir, 'points3D.ply')
    dataset = base_dir
    npz_path = os.path.join(dataset,'gpw.npy')
    target_pose_w = np.load(npz_path)

    # all in object coodinate
    pc, pc_h, pc_o, mean, pc_whole, pc_whole_colors,dir_hono,pc_color = readPly(object_pcl, hand_pcl, ply_file, whole_scene)
    target_pose_o = copy.deepcopy(target_pose_w)
    target_pose_o[:3, 3] -= mean
   
    npz_data = os.path.join(dataset,'trajectory_v4')
    if os.path.exists(npz_data):
        shutil.rmtree(npz_data)
    os.makedirs(npz_data,exist_ok = True)

    train_camera = parse_colmap_images(os.path.join(dataset,'sparse','0','images.txt'))
    
    # initial poses
    sphere_samples = sample_initial_pose_sphere(target_pose_o, radius=sample_initial_radius,
                                                    num_samples=num_samples,
                                                    handover_pcl=pc,
                                                    max_rotation_angle=max_rotation_angle,
                                                    min_distance=min_distance_initial_pose,dir_hono = dir_hono,mean=mean,
                                                    train_camera = train_camera,
                                                    threshold_train_cam=threshold_train_cam)
    sphere_target_samples = copy.deepcopy(sphere_samples)
    sphere_target_samples.append(target_pose_o)
   
    grasp_score = np.ones(len(sphere_target_samples))

    # 2:right up


    
    for sample in train_camera:
        sample[:3, 3] -= mean
    train_camera_score = np.zeros(len(train_camera))
    sphere_target_samples.extend(train_camera)
    grasp_score = np.concatenate((grasp_score,train_camera_score))
    # mlab.figure(bgcolor=(1, 1, 1),size=(1000, 800))
    # draw_scene(
    #     pc,
    #     pc_color =  pc_color,
    #     grasps = sphere_target_samples,
    #     grasp_scores = grasp_score,
    # )
    
    # print('close the window to continue to next object . . .')
    # mlab.show()

    # interpolated poses
    for j, sample in enumerate(sphere_samples):
        sample_j_path = os.path.join(npz_data,str(j))
        os.makedirs(sample_j_path,exist_ok = True)
        start_pose = copy.deepcopy(sample)
        # interpolated_poses contain start pose, not contain target pose
        interpolated_poses = sample_poses(start_pose, target_pose_o,
                                            handover_pcl=pc,
                                            step_distance=step_distance,
                                            min_distance_pointcloud=min_distance_pointcloud,
                                            near_dis=near_dis,
                                            min_distance=min_distance_sample_pose)
       
        initial_interpolated_poses = copy.deepcopy(interpolated_poses)
        initial_interpolated_poses.append(target_pose_o)
        initial_interpolated_poses_score = np.zeros(len(initial_interpolated_poses))
        # mlab.figure(bgcolor=(1, 1, 1),size=(1000, 1000))
        # draw_scene(
        #     pc,
        #     pc_color =  pc_color,
        #     grasps = initial_interpolated_poses,
        #     grasp_scores = initial_interpolated_poses_score,
        # )
        # mlab.show()

        # transfer pose in object coordinate to world coordinate and save
        start_pose_w = copy.deepcopy(start_pose)
        start_pose_w[:3, 3] += mean
        target_pose_w = copy.deepcopy(target_pose_o)
        target_pose_w[:3, 3] += mean
        np.save(f'{sample_j_path}/{j}_target.npy', target_pose_w)

        interpolated_poses_world = copy.deepcopy(interpolated_poses)
        for i, pose_w in enumerate(interpolated_poses_world):
            pose_w[:3, 3] += mean
            np.save(f'{sample_j_path}/{j}_{i}.npy', pose_w)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample poses for handover task.')
    
    # 添加参数及其默认值
    parser.add_argument('--sample_initial_radius', type=float, default=1,
                        help='Initial radius for sampling poses (default: 10.0)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate (default: 10)')
    parser.add_argument('--max_rotation_angle', type=float, default=70.0,
                        help='not use Maximum rotation angle in degrees  (default: 70.0) for initial sample')
    parser.add_argument('--min_distance_initial_pose', type=float, default=0.7,
                        help='Minimum distance for initial pose (default: 0.7)')
    parser.add_argument('--step_distance', type=float, default=0.05,
                        help='Step distance for interpolation (default: 0.5)')
    parser.add_argument('--min_distance_sample_pose', type=float, default=0.2,
                        help='pre grasp distance from target pose ')
    parser.add_argument('--base_dir', type=str, default='/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_safe_grasp/54_frame_handover_3D',
                        help='base directory for the 3D data files')
    parser.add_argument('--near_dis', type=float,default=0.4)
    parser.add_argument('--min_distance_pointcloud', type=float,default=0.05,
                        help='Minimum distance for pose and pointcloud of hand and object')
    parser.add_argument('--threshold_train_cam', type=float,default=0.3)
                        
    

    args = parser.parse_args()
    
    main(args)


    
