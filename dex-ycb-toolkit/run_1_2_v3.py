# multiprocess for dataset like /home/e/eez095/dexycb_data/20200813-subject-02
# have error.txt and complete.txt
import subprocess
import os
import re
import argparse
from tqdm import tqdm
import concurrent.futures

# 定义执行命令的函数
def execute_commands(dataSet, dataset_path, completed_file, error_file):
    res = True
    common_commands = [
        f"python examples/get_pointcloud.py --name {dataSet}"
    ]
    frame_specific_command = "python examples/visualize_pose.py --src {}"

    # 执行命令，捕获异常并写入错误日志文件
    for cmd in common_commands:
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            with open(error_file, 'a') as ef:
                ef.write(f"Error executing command: {cmd}\nError message: {e}\n\n")
            return False

    data_frames = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    for data_frame in data_frames:
        if not re.match(r'\d+_frame$', os.path.basename(data_frame)):
            continue
        cmd = frame_specific_command.format(data_frame)
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            error_details = traceback.format_exc()
            with open(error_file, 'a') as ef:
                ef.write(f"Error executing command: {cmd}\nError message: {e} {error_details}\n\n")
            res = False
    
   
    with open(completed_file, 'a') as cf:
        cf.write(f"{dataset_path}\n")
    
    return res

def process_data(dataSetBig, base_path, completed_file, error_file):
    datas = [f for f in os.listdir(dataSetBig) if os.path.isdir(os.path.join(dataSetBig, f))]

    # 从已完成文件读取已处理的数据集路径
    if os.path.exists(completed_file):
        with open(completed_file, 'r') as cf:
            completed_datasets = {line.strip() for line in cf.readlines()}
    else:
        completed_datasets = set()

    # 设置最大并发工作数
    max_workers = 8  # 你可以调整这个数字
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_data = {
            executor.submit(execute_commands, os.path.relpath(os.path.join(dataSetBig, data), base_path), os.path.join(dataSetBig, data), completed_file, error_file): data
            for data in datas if os.path.join(dataSetBig, data) not in completed_datasets
        }
        
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
    parser.add_argument('--completed_file', type=str, default='/home/e/eez095/project/dex-ycb-toolkit/completed_datasets.txt', help="File to store completed dataset paths")
    parser.add_argument('--error_file', type=str, default='/home/e/eez095/project/dex-ycb-toolkit/error_log.txt', help="File to store error messages")
    args = parser.parse_args()
    
    dataSetBig = args.dataSetBig
    dataSet = args.dataSet
    base_path = '/home/e/eez095/dexycb_data'
    completed_file = args.completed_file
    error_file = args.error_file

    # 处理数据集
    process_data(dataSetBig, base_path, completed_file, error_file)
