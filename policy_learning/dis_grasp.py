#compute the rot and trans differece between grasp pose and pre grasp pose
import argparse
import numpy as np
import transforms3d

def parse_args():
    parser = argparse.ArgumentParser(description="Compute Euler angles and translation difference between two 4x4 pose matrices.")
    parser.add_argument("--pre_grasp_file", type=str, required=True, help="First pose matrix npz file")
    parser.add_argument("--grasp_pose_file",  type=str, required=True, help="Second pose matrix npz file")
    return parser.parse_args()

def matrix_to_euler_and_translation(pose_matrix):
    # 从4x4矩阵提取欧拉角和位移
    rotation_matrix = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    
    # 将旋转矩阵转换为欧拉角 (假设使用xyz顺序)
    euler_angles = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
    euler_angles = np.array(euler_angles)

    return euler_angles, translation

def main():
    args = parse_args()
    
    # Reshape input matrices
    pre_grasp_file = np.load(args.pre_grasp_file)
    grasp_pose_file = np.load(args.grasp_pose_file)
    
    # Compute Euler angles and translations
    euler1, trans1 = matrix_to_euler_and_translation(pre_grasp_file)
    euler2, trans2 = matrix_to_euler_and_translation(grasp_pose_file)
    
    # Compute differences
    euler_diff = euler2 - euler1
    translation_diff = np.linalg.norm(trans2 - trans1)
    
    # Print results
    print("Euler angles of pre_grasp_file (XYZ, degrees):", euler1)
    print("Euler angles of grasp_pose_file (XYZ, degrees):", euler2)
    print("Difference in Euler angles (degrees):", euler_diff)
    print("Translation of pre_grasp_file:", trans1)
    print("Translation of grasp_pose_file:", trans2)
    print("Translation distance difference:", translation_diff)

if __name__ == "__main__":
    main()
