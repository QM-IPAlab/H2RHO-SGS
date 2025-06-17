# multi thread process data like /home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341
import subprocess
import argparse
import os
import re
from multiprocessing import Pool

# 用于执行命令的函数
def execute_commands(safe_grasp_folder, processed_frames, errors_file):
    # 如果该帧已经处理过，则跳过
    if safe_grasp_folder in processed_frames:
        print(f"Skipping already processed frame: {safe_grasp_folder}")
        return True

    # 定义要执行的命令
    commands = [
        f"python -m demo.main --safe_grasp_folder {safe_grasp_folder}",
        f"python -m demo.sample_v3 --base_dir {safe_grasp_folder}"
    ]

    # 执行命令
    for cmd in commands:
        print(f"Executing: {cmd}")
        try:
            # 使用 shell=True 来允许 conda activate 命令正常工作
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            # 如果有错误发生，记录到错误文件，并返回 False
            with open(errors_file, 'a') as ef:
                ef.write(f"Error executing command: {cmd}\nError message: {e}\nFor folder: {safe_grasp_folder}\n\n")
            print(f"Error executing command: {cmd}")
            return False  # 如果有错误发生，返回 False
        print(f"Command completed: {cmd}\n")

    # 将当前处理的帧写入 processed_frames.txt 文件
    with open('./6dof_processed_frames.txt', 'a') as pf:
        pf.write(safe_grasp_folder + "\n")
    
    print("All commands executed successfully.")
    return True  # 如果所有命令都成功执行，返回 True

def process_dataset(dataset_path, max_threads):
    # 读取已经处理过的帧
    processed_frames = set()
    if os.path.exists('./6dof_processed_frames.txt'):
        with open('./6dof_processed_frames.txt', 'r') as pf:
            processed_frames = set(pf.read().splitlines())
    
    # 获取所有数据帧
    data_frames = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    tasks = []

    # 将每个数据帧的路径添加到任务列表
    for data_frame in data_frames:
        if re.match(r'\d+_frame$', os.path.basename(data_frame)):
            tasks.append(os.path.join(dataset_path, data_frame))
    
    # 使用多线程处理每个任务
    with Pool(processes=max_threads) as pool:
        results = pool.starmap(execute_commands, [(task, processed_frames, './6dof_errors_frame.txt') for task in tasks])

    # 检查哪些帧处理失败
    for idx, result in enumerate(results):
        if not result:
            print(f"There was an error during execution for {tasks[idx]}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='task 3 and 4')
    
    # 添加参数及其默认值
    parser.add_argument('--grasp_one_frame', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/70_frame', help='dataset for one frame')
    parser.add_argument('--dataset_path', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341')
    parser.add_argument('--BigDataset_path', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02')
    parser.add_argument('--max_threads', type=int, default=4, help='Maximum number of threads for parallel processing')

    args = parser.parse_args()
    grasp_one_frame = args.grasp_one_frame
    dataset_path = args.dataset_path
    BigDataset_path = args.BigDataset_path
    max_threads = args.max_threads

    # 处理数据帧
    process_dataset(dataset_path, max_threads)
