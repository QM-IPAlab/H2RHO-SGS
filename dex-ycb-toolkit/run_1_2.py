
import subprocess
import os
import re
import argparse
from tqdm import tqdm



def execute_commands(dataSet,dataset_path):
    # Define the commands
    common_commands = [
        # f"python examples/create_dataset.py",
        f"python examples/get_pointcloud.py --name {dataSet}"
    ]
    frame_specific_command = "python examples/visualize_pose.py --src {}"

    # Execute common commands once
    for cmd in common_commands:
        print(f"--------Executing: {cmd}-----------")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {cmd}")
            print(f"Error message: {e}")
            return False
        print(f"Command completed: {cmd}\n")
    print('hi')
    data_frames = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    # Execute frame-specific command for each data_frame
    for data_frame in data_frames:
        if not re.match(r'\d+_frame$', os.path.basename(data_frame)):
            continue
        cmd = frame_specific_command.format(data_frame)
        print(f"---------Executing: {cmd}------------")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {cmd}")
            print(f"Error message: {e}")
            return False
        print(f"Command completed: {cmd}\n")

    print("All commands executed.")
    return True

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process dataset commands.')
    parser.add_argument('--dataSet', type=str, default='20200813-subject-02/20200813_145341', help='The dataset to process')
    parser.add_argument('--dataSetBig', type=str, default = '/home/e/eez095/dexycb_data/20200813-subject-02', help="dataset")
    args = parser.parse_args()
    dataSetBig = args.dataSetBig
    dataSet = args.dataSet
    # dataSet = '20200813-subject-02/20200813_145341'
    base_path = '/home/e/eez095/dexycb_data'
    dataset_path = os.path.join(base_path, dataSet)

    
    #process dataset like /home/e/eez095/dexycb_data/20200813-subject-02
    # datas = [f for f in os.listdir(dataSetBig) if os.path.isdir(os.path.join(dataSetBig, f))]
    
    # # for data in tqdm(datas,desc = "processing data in 1-2 step"):
    # #     full_path = os.path.join(dataSetBig,data)
    # #     a_path = os.path.relpath(full_path, base_path)
    # #     success = execute_commands(a_path, full_path)
    # #     print(f'dealing with {data} now ----- ')
    # #     if success:
    # #         continue
    # #     else:
    # #         print(f"There was an error during execution{a_path}.")
        


    # process dataset like 20200813-subject-02/20200813_145341
    success = execute_commands(dataSet,dataset_path)
    if success:
        print("All commands executed successfully.")
    else:
        print("There was an error during execution.")
 