# 检查每个路径下的 trajectory_v4 文件夹是否为空，然后将非空的路径存入另一个 txt 文件
import os

# 输入 TXT 文件路径
input_txt_path = "/home/e/eez095/project/pytorch_6dof-graspnet/log/completed_datasets.txt"  
output_txt_path = "/home/e/eez095/project/pytorch_6dof-graspnet/log/non_empty_paths_v2.txt" 
empty_data_path = "/home/e/eez095/project/pytorch_6dof-graspnet/log/empty_traj_paths_v2.txt"

# 读取路径列表
with open(input_txt_path, "r") as file:
    paths = [line.strip() for line in file.readlines()]

# 过滤非空的 trajectory_v4 文件夹
non_empty_paths = []
empty_path = []
for path in paths:
    trajectory_path = os.path.join(path, "trajectory_v4")
    
    # 检查 trajectory_v4 是否存在且不为空
    if os.path.exists(trajectory_path) and os.listdir(trajectory_path):
        non_empty_paths.append(path)
    else:
        empty_path.append(path)

# 存入新的 TXT 文件
with open(output_txt_path, "w") as file:
    for path in non_empty_paths:
        file.write(path + "\n")
with open(empty_data_path,"w") as ef:
    for path in empty_path:
        ef.write(path + "\n")
print(f"筛选完成，{len(non_empty_paths)} 个路径已保存至 {output_txt_path}")
