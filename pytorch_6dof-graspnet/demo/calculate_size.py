import trimesh
import glob
import os

# 指定包含点云文件的文件夹
ply_folder = '/home/robot_tutorial/vgn_ws/src/wyk/pytorch_6dof-graspnet/demo/data_ply/'

# 遍历所有 PLY 文件
for ply_file in glob.glob(os.path.join(ply_folder, '*.ply')):
    # 加载点云数据
    pc = trimesh.load(ply_file)
    
    # 获取顶点数组
    vertices = pc.vertices
    
    # 打印点云的大小（顶点数量）
    print(f'Point cloud file: {ply_file}, Number of points: {vertices.shape[0]}')