import os
import time
import subprocess

# object on the up
des_dir = '/home/e/eez095/project/motion/data/seen/obj10/2'
os.makedirs(des_dir,exist_ok=True)
output_file = f"{des_dir}/log.txt"
src_path = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_152401/60_frame'
policy = 'policy_v12_T_norm'
model_path = '/home/e/eez095/project/policy_learning/model/0311_norm_sep_lucky_v6000.pth'
hand_ply_file = f'{src_path}/handover_3D/hand.ply'
obj_ply_file = f'{src_path}/handover_3D/object.ply'
ini_test_img_path = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_152401/60_frame/FSGS_output/video/sample_v610000/4/4_0.png'
ini_test_traj_path = f'{src_path}/trajectory_v6/4/4_0.npy'
# pre_grasp_path = f'{src_path}/trajectory_v6/3/3_23.npy'
grasp_pose_file = f'{src_path}/trajectory_v6/4/4_target.npy'
first_pth = f'{des_dir}/1.npy'


def run_command(command, output_file):
    with open(output_file, "a+") as f:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="")  # 输出到终端
            f.write(line)  # 写入文件
        process.wait()


command1 = f"""python {policy}.py \
            --mode test  \
            --model_path {model_path}  \
            --hand_ply_file {hand_ply_file}   \
            --obj_ply_file {obj_ply_file}  \
            --mask_dir {des_dir} \
            --test_image_path {ini_test_img_path}   \
            --test_traj_path {ini_test_traj_path}   \
            --next_pose_path {first_pth}"""

command3 = f""" 
            python dis_grasp.py \
            --predict_grasp_file {des_dir}/26.npy \
           
            --object_pcl_file {obj_ply_file} \
            --grasp_pose_file {grasp_pose_file}\
            --hand_pcl_file {hand_ply_file}
            """
           

print(f"---------Running command1...")
run_command(command1, output_file)
print(f'-----------command1 finish!')

total_iterations = 27
for i in range(1, total_iterations + 1):
    print(i)
    while not os.path.exists(f'{des_dir}/{i}.png'):
        print(f'wait for fsgs side')
        time.sleep(20)
        
    command2 = f"""python {policy}.py \
            --mode test  \
            --model_path  {model_path} \
            --hand_ply_file {hand_ply_file}   \
            --obj_ply_file {obj_ply_file}  \
            --mask_dir {des_dir}  \
            --test_image_path {des_dir}/{i}.png   \
            --test_traj_path {des_dir}/{i}.npy \
            --next_pose_path {des_dir}/{i+1}.npy"""
    print(f"----------Running command2 for i={i}...")
    run_command(command2, output_file)
print(f"---------Running distance meature...")
run_command(command3, output_file)