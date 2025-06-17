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

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from plyfile import PlyData
import argparse
import os
import shutil
import copy

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

def sample_poses(start_pose, target_pose, handover_pcl ,step_distance, min_distance):
    """根据给定的步长在初始位姿和目标位姿之间进行采样"""
    
    # 提取位置和旋转部分
    start_position = start_pose[:3, 3]
    target_position = target_pose[:3, 3]
    
    start_rotation = start_pose[:3, :3]
    target_rotation = target_pose[:3, :3]
    
    # 计算位置之间的距离
    distance = np.linalg.norm(target_position - start_position)
    
    # 计算需要的插值步数
    num_steps = int(np.ceil(distance / step_distance))
    if(num_steps == 0):
        import pdb
        pdb.set_trace()
    
    # 生成采样点
    sampled_poses = [start_pose]
    
    for i in range(1,num_steps):
        t = i / num_steps
        
        # 线性插值位置
        position = linear_interpolate(start_position, target_position, t)
        
        # Slerp 插值旋转
        rotation = slerp(start_rotation, target_rotation, t)
        
        # 构建新的4x4位姿矩阵
        sampled_pose = np.eye(4)
        sampled_pose[:3, 3] = position
        sampled_pose[:3, :3] = rotation

        if is_too_close(sampled_pose, handover_pcl, min_distance):
            print(f'sample pose is too close to hand and object, t is {t}, start_pose is {start_pose}')
        else:
            sampled_poses.append(sampled_pose)
    return sampled_poses

def sample_initial_pose_sphere(target_pose_o, radius, num_samples,  handover_pcl, max_rotation_angle, min_distance, dir_hono, mean ):
    samples = []
    i = 0
    target_rotation = target_pose_o[:3, :3]
    while i < num_samples:
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

        # # new_pose移动至target_pose_o 的方向，不能和target_pose_o->object 的rotation方向大于60度
        # xyz轴不是直直的new_pose[0,3] * target_pose_w[0,3]) < 0 or (new_pose[1,3] * target_pose_w[1,3]) < 0 or (new_pose[3,3] * target_pose_w[3,3]) < 0
        dir_o2p = new_pose[:3, 3] - target_pose_o[:3, 3]
        dir_pose = target_pose_o[:3, 3]
        a1 = np.array(dir_o2p)
        b1 = np.array(dir_pose)
        dot_product = np.dot(a1, b1)
        norm_a1 = np.linalg.norm(a1)
        norm_b1 = np.linalg.norm(b1)
        cos_theta1 = dot_product / (norm_a1 * norm_b1)
        angle_rad1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))  # 使用 clip 防止数值误差
        angle_deg1 = np.degrees(angle_rad1)  
        print(angle_deg1)

        dir_sampleono = new_pose[:3, 3] 
        a = np.array(dir_sampleono)
        b = np.array(dir_hono)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        cos_theta = dot_product / (norm_a * norm_b)
        angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 使用 clip 防止数值误差
        angle_deg = np.degrees(angle_rad)  
        print(angle_deg)

        # angle_deg>90 or target_pose_w[0, 3]*dir_o2p[0]<0 or target_pose_w[1, 3]*dir_o2p[1]<0 or target_pose_w[2, 3]*dir_o2p[2]<0
      
        if(angle_deg <120 or angle_deg1 > 50):
            continue
        else:
            i += 1
            # 给new pose设计初始的旋转 在原始的旋转上加上一些小的扰动
            np_random_rotation = np.random.uniform(-1,1,size = (3,))
            
            direction_to_object =  - new_pose[:3, 3]
            direction_to_object /= np.linalg.norm(direction_to_object)  # 单位化
            z_axis = direction_to_object  # 新位姿的Z轴朝向物体
            x_axis = np.cross(np.array([0, 0, 1]), z_axis)  # 假设初始Z轴为[0,0,1]
            x_axis /= np.linalg.norm(x_axis)  # 单位化
            y_axis = np.cross(z_axis, x_axis)  # Y轴垂直于X和Z轴

            rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))  # 构建旋转矩阵
            new_pose[:3, :3] = rotation_matrix  # 更新旋转部分

          
           

            # perturbation = R.from_rotvec(max_rotation_angle * np_random_rotation, degrees = True) 
            # p2 = perturbation.as_euler('xyz', degrees=True)

            # r_original = R.from_matrix(target_rotation)
            # p1 = r_original.as_euler('xyz', degrees=True)
            # # print(f'original pose {p1}')

            # new_rotation = perturbation*r_original
            # p3 = new_rotation.as_euler('xyz', degrees=True)
            # # print(f'new_rotation {p3}')

            # new_pose[:3, :3] = new_rotation.as_matrix()
        
            samples.append(new_pose)

        
    return samples

def sample_pose_rectangle(target_pose, width, height, num_samples):
    samples = []
    for _ in range(num_samples):
        # 在矩形区域内随机生成偏移量
        dx = np.random.uniform(-width / 2, width / 2)
        dy = np.random.uniform(-height / 2, height / 2)
        
        # 生成新的位姿
        new_pose = copy.deepcopy(target_pose)
        new_pose[0, 3] += dx  # 更新x轴平移
        new_pose[1, 3] += dy  # 更新y轴平移
        
        samples.append(new_pose)
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

    # 以物体为原点，主相机为坐标系方向，的坐标系里
    # target_pose_w = np.array(  [[ 0.67889947, -0.72975671, -0.08093569,  0.12466926],
    #                             [ 0.61408573,  0.50392175,  0.60742211, -0.4597988 ],
    #                             [-0.40248513, -0.46208,     0.79024541,  0.6746785 ],
    #                             [ 0.,          0.,          0.,          1.        ]] )
    
    # object_pcl = '/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_safe_grasp/54_frame_handover_3D/object.ply'
    # hand_pcl = '/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_safe_grasp/54_frame_handover_3D/hand.ply'
    # ply_file = '/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_safe_grasp/54_frame_handover_3D/handover.ply'
    # whole_scene = '/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_safe_grasp/54_frame_handover_3D/points3D.ply'
    # dataset = '/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_safe_grasp/54_frame_handover_3D'
    
    # base_dir = '/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_safe_grasp/54_frame_handover_3D'
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
   
    npz_data = os.path.join(dataset,'trajectory')
    if os.path.exists(npz_data):
        shutil.rmtree(npz_data)
    os.makedirs(npz_data,exist_ok = True)
    
    # initial poses
    sphere_samples = sample_initial_pose_sphere(target_pose_o, radius=sample_initial_radius,
                                                    num_samples=num_samples,
                                                    handover_pcl=pc,
                                                    max_rotation_angle=max_rotation_angle,
                                                    min_distance=min_distance_initial_pose,dir_hono = dir_hono,mean=mean)
    sphere_target_samples = copy.deepcopy(sphere_samples)
    sphere_target_samples.append(target_pose_o)
   
    grasp_score = np.ones(len(sphere_target_samples))

    train_camera = np.array([
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0, 0, 0, 1]
    ],
    [
        [-0.893451988697052, -0.03789233788847923, -0.4475576877593994, 0.6386182308197021],
        [-0.3177464008331299, 0.7575904130935669, 0.5701702237129211, -0.5989090204238892],
        [0.3174603581428528, 0.6516294479370117, -0.688910722732544, 1.391905665397644],
        [0, 0, 0, 1]
    ],
    [
        [-0.2367057204246521, 0.26019808650016785, 0.9360916018486023, -0.6338921785354614],
        [0.7574113011360168, 0.6528606414794922, 0.010052978061139584, -0.08515024185180664],
        [-0.6085217595100403, 0.7113859057426453, -0.35161277651786804, 1.0669708251953125],
        [0, 0, 0, 1]
    ],
  
    [
        [0.5951622619388115, 0.6853450488717701, -0.4196236956498394, 0.7148191928863525],
        [-0.6514453394194257, 0.7172168040858102, 0.2474247879369531, -0.3177867531776428],
        [0.47053251929339285, 0.12610400439275565, 0.8733252134019291, 0.18353232741355896],
        [0, 0, 0, 1]
    ],
    [
        [-0.4766446146572601, 0.75390154568288, -0.4521530390644259, 0.41736653447151184],
        [-0.6333187685169396, 0.06223099033979218, 0.7713848853105145, -0.6476160287857056],
        [0.609686188758134, 0.6540334572927389, 0.447797932084649, 0.40139156579971313],
        [0, 0, 0, 1]
    ],
    [
        [0.4486923103476053, -0.6708688872680653, 0.5904321694577928, -0.1706404834985733],
        [0.6349333263878278, 0.7042357829222984, 0.3176659142777285, -0.46829313039779663],
        [-0.6289156395551123, 0.2323508083642598, 0.7419421946320743, 0.28708332777023315],
        [0, 0, 0, 1]
    ],
    [
        [-0.2469573303563285, -0.5030590752630304, 0.8282171477208013, -0.5300791263580322],
        [0.6828783108268728, 0.516055556015601, 0.5170724085731763, -0.5804263353347778],
        [-0.6875240283695634, 0.6932663484555798, 2e-01 , .7049697637557983],
        [ 0  , 0  ,0   ,  1 ]
    ],
    [
        [-0.9286197279370463, 0.28499809028787526, -0.23757417666571096, 0.35575658082962036],
        [-0.19175885377662133, 0.17951273203290213, 0.9648853408754638, -1.0083527565002441],
        [0.31763806901000186, 0.9415685145385586, -0.1120481572595004, 0.9933208227157593],
        [0, 0, 0, 1]
    ]
    ])
    for sample in train_camera:
        sample[:3, 3] -= mean
    train_camera_score = np.zeros(len(train_camera))
    sphere_target_samples.extend(train_camera)
    grasp_score = np.concatenate((grasp_score,train_camera_score))
    # mlab.figure(bgcolor=(1, 1, 1))
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
                                            min_distance=min_distance_sample_pose)
        

        initial_interpolated_poses = copy.deepcopy(interpolated_poses)
        initial_interpolated_poses.append(target_pose_o)
        initial_interpolated_poses_score = np.zeros(len(initial_interpolated_poses))
        # mlab.figure(bgcolor=(1, 1, 1))
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
    parser.add_argument('--sample_initial_radius', type=float, default=1.5,
                        help='Initial radius for sampling poses (default: 10.0)')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to generate (default: 10)')
    parser.add_argument('--max_rotation_angle', type=float, default=30.0,
                        help='Maximum rotation angle in degrees (default: 40.0)')
    parser.add_argument('--min_distance_initial_pose', type=float, default=0.4,
                        help='Minimum distance for initial pose (default: 0.7)')
    parser.add_argument('--step_distance', type=float, default=0.1,
                        help='Step distance for interpolation (default: 0.5)')
    parser.add_argument('--min_distance_sample_pose', type=float, default=0.2,
                        help='Minimum distance for sampled poses (default: 0.2)')
    parser.add_argument('--base_dir', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/70_frame', help='base directory for the 3D data files')

    args = parser.parse_args()
    
    main(args)


    
