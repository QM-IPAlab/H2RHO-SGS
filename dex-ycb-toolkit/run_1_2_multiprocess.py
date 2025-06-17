import subprocess
import os
import re
import argparse
from tqdm import tqdm
import concurrent.futures

# 定义执行命令的函数
def execute_commands(dataSet, dataset_path):
    common_commands = [
        f"python examples/get_pointcloud.py --name {dataSet}"
    ]
    frame_specific_command = "python examples/visualize_pose.py --src {}"
    
    for cmd in common_commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {cmd}")
            print(f"Error message: {e}")
            return False

    data_frames = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    for data_frame in data_frames:
        if not re.match(r'\d+_frame$', os.path.basename(data_frame)):
            continue
        cmd = frame_specific_command.format(data_frame)
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {cmd}")
            print(f"Error message: {e}")
            return False
    
    return True

def process_data(dataSetBig, base_path):
    datas = [f for f in os.listdir(dataSetBig) if os.path.isdir(os.path.join(dataSetBig, f))]
    
    # 设置最大并发工作数
    max_workers = 6  # 你可以调整这个数字
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {executor.submit(execute_commands, os.path.relpath(os.path.join(dataSetBig, data), base_path), os.path.join(dataSetBig, data)): data for data in datas}
        
        # 监控执行状态并获取结果
        for future in tqdm(concurrent.futures.as_completed(future_to_data), desc="Processing data"):
            data = future_to_data[future]
            try:
                result = future.result()
                if result:
                    print(f"Successfully processed {data}")
                else:
                    print(f"Error processing {data}")
            except Exception as e:
                print(f"Error processing {data}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset commands.')
    parser.add_argument('--dataSet', type=str, default='20200813-subject-02/20200813_145341', help='The dataset to process')
    parser.add_argument('--dataSetBig', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02', help="dataset")
    args = parser.parse_args()
    dataSetBig = args.dataSetBig
    dataSet = args.dataSet
    base_path = '/home/e/eez095/dexycb_data'

    # 处理数据集
    process_data(dataSetBig, base_path)
