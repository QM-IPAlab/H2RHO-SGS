# generate hand and object mask from trajectory 
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage

import numpy as np
import scipy.ndimage

def fill_mask(mask_np):
    """
    改进的二值化 mask 处理：
    - 更平滑的形态学闭运算
    - 避免错误填充

    参数：
        mask_np (np.ndarray): 形状为 (H, W) 的二值化图像（0 和 255）

    返回：
        np.ndarray: 经过优化处理的 mask
    """
    # 归一化 mask（转换为 0/1）
    mask_binary = mask_np > 0  
    structure_element = np.ones((7, 7))  # 5x5 结构元素
    mask_closed = scipy.ndimage.binary_closing(mask_binary, structure=structure_element,iterations=3)
    mask_filled = scipy.ndimage.binary_fill_holes(mask_closed)

    # **3. 转换回 0-255 格式**
    mask_final = (mask_filled.astype(np.uint8)) * 255

    return mask_final


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

    # Load pose matrix
    pose = np.load(pose_file)  # Should be a 4x4 transformation matrix
    pose = np.linalg.inv(pose) 
    pose = o3d.core.Tensor(pose,dtype=o3d.core.Dtype.Float32)


    # Load hand and object point clouds
    hand_pcl = o3d.io.read_point_cloud(hand_ply_file)
    object_pcl = o3d.io.read_point_cloud(object_pcl_file)

    hand_pcl = o3d.t.geometry.PointCloud.from_legacy(hand_pcl)
    object_pcl = o3d.t.geometry.PointCloud.from_legacy(object_pcl)

    width= 640
    height= 480
    fx=621
    fy=621
    cx=302
    cy=236
    fx = 607.3981323242188
    fy = 607.3981323242188
    cx = 321.79571533203125
    cy = 242.15435791015625

    # Set up a virtual camera (intrinsic parameters)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    intrinsic_tensor = o3d.core.Tensor(intrinsic.intrinsic_matrix,dtype=o3d.core.Dtype.Float32)

    # 投影点云到2D图像
    hand_img = hand_pcl.project_to_depth_image(width, height, intrinsic_tensor, pose )
    hand_img_np = np.asarray(hand_img.to_legacy())
    threshold = 0.1 
    # 二值化，深度小于阈值的为 255，大于阈值的为 0
    hand_img_np = np.where(hand_img_np > threshold, 255, 0).astype(np.uint8)
    hand_mask_save_path = os.path.join(output_dir,'hand_mask.png')
    plt.imsave(hand_mask_save_path, hand_img_np, cmap='gray')
    mask_filled_h = fill_mask(hand_img_np)
    fill_hand_mask_save_path = os.path.join(output_dir,'hand_mask_filled.png')
    plt.imsave(fill_hand_mask_save_path, mask_filled_h, cmap='gray')


    object_img = object_pcl.project_to_depth_image(width, height, intrinsic_tensor,  pose)
    object_img_np = np.asarray(object_img.to_legacy())
    object_img_np = np.where(object_img_np > threshold, 255, 0).astype(np.uint8)
    object_mask_save_path = os.path.join(output_dir,'object_mask.png')
    plt.imsave(object_mask_save_path, object_img_np, cmap='gray')
    mask_filled_o = fill_mask(object_img_np)
    fill_object_mask_save_path = os.path.join(output_dir,'object_mask_filled.png')
    plt.imsave(fill_object_mask_save_path, mask_filled_o, cmap='gray')

    hand_pil = Image.open(fill_hand_mask_save_path).convert('L')
    object_pil = Image.open(fill_object_mask_save_path).convert('L')
    # print(hand_pil.mode)  # 查看图像模式，检查是否为 RGBA
    # print(hand_pil.getbands())  # 查看图像的通道名称
    # indices = np.nonzero(np.array(hand_pil))
    # values = np.array(hand_pil)[indices]
    # print(f'----------------values:{values}')
    # object_mask_save_path_ = os.path.join(output_dir,'object_mask_again.png')
    # plt.imsave(object_mask_save_path_,hand_pil,cmap='gray')

    return hand_pil,object_pil


# pose = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_154103/60_frame/trajectory_v4/0/0_0.npy'
# hand_ply_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_154103/60_frame/handover_3D/hand.ply'
# object_pcl_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_154103/60_frame/handover_3D/object.ply'
# output_dir= '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_154103/60_frame/mask'
# get_mask(pose,hand_ply_file,object_pcl_file,output_dir)