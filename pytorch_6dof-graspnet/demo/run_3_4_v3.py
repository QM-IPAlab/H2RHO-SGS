## multi thread process data like /home/e/eez095/dexycb_data/20200813-subject-02
## 5 frames for each /home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341
import subprocess
import os
import re
import argparse
from tqdm import tqdm
import concurrent.futures
from threading import Lock

# 锁，用于文件写入操作的线程安全
error_lock = Lock()
completed_lock = Lock()


def execute_commands(safe_grasp_folder, error_file):
    """执行指定命令并记录错误"""
    commands = [
        f"python -m demo.main --safe_grasp_folder {safe_grasp_folder}",
        f"python -m demo.sample_v4 --base_dir {safe_grasp_folder}"
    ]

    processes = []  # 存储所有子进程
    for cmd in commands:
        try:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            processes.append((cmd, process))
        except Exception as e:
            with error_lock, open(error_file, 'a') as ef:
                ef.write(f" {cmd}\n")
                ef.write(f" {e}\n\n")
            return False  # 失败则不执行下一个命令

    # **等待所有进程完成，同时收集错误**
    all_success = True
    for cmd, process in processes:
        try:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                all_success = False
                with error_lock, open(error_file, 'a') as ef:
                    ef.write(f"Error executing command: {cmd}\n")
                    ef.write(f"Error message: {stderr.decode()}\n\n")
        except subprocess.TimeoutExpired:
            process.kill()  #强制终止进程
            stdout, stderr = process.communicate()  # 取出缓冲区里的数据，防止阻塞
            all_success = False
            with error_lock, open(error_file, 'a') as ef:
                ef.write(f"Timeout executing {cmd} in {safe_grasp_folder}\n")
                ef.write(f"Partial output before timeout:\n{stdout}\n\n") 
    return all_success


def custom_sort(filename):
    if '_target' in filename:
        return (float('inf'),)
    match = re.findall(r'(\d+)', filename)
    return tuple(int(num) for num in match)

def process_dataset_path(dataset_path, completed_file, error_file):
    """处理单个 dataset_path 的所有帧目录"""
    data_frames = [
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f)) and re.match(r'\d+_frame$', os.path.basename(f))
    ]
    data_frames = sorted(data_frames,key=custom_sort)

    if os.path.exists(completed_file):
        with open(completed_file, 'r') as cf:
            completed_datasets = {line.strip() for line in cf.readlines()}
    else:
        completed_datasets = set()
    
    all_success = True
    step = len(data_frames) // 6  # 计算平均间隔
    selected_paths = [data_frames[i * step] for i in range(1,6)]
    print(f'selected_paths {selected_paths}')
    for data_frame in selected_paths:
        safe_grasp_folder = os.path.join(dataset_path, data_frame)
        if safe_grasp_folder in completed_datasets:
                print(f"Skipping already processed: {safe_grasp_folder}")
                continue


        success = execute_commands(safe_grasp_folder, error_file)
        if not success:
            all_success = False
        if success:
            with completed_lock, open(completed_file, 'a') as cf:
                cf.write(f"{safe_grasp_folder}\n")


    return all_success

"""处理大数据集路径中的所有子数据集"""
def process_big_dataset(BigDataset_path, completed_file, error_file, max_workers,step1_complete):


    with open(step1_complete, 'r') as file:
        valid_dirs = {line.strip() for line in file.readlines()}
  
    

    dataset_paths = [
        os.path.join(BigDataset_path, d)
        for d in os.listdir(BigDataset_path)
        if os.path.isdir(os.path.join(BigDataset_path, d)) and os.path.join(BigDataset_path, d) in valid_dirs
    ]

    # 多线程处理数据集
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dataset = {}

  
        for dataset_path in dataset_paths:
            future = executor.submit(process_dataset_path, dataset_path, completed_file, error_file)
            future_to_dataset[future] = dataset_path

        # 监控任务完成情况
        for future in tqdm(concurrent.futures.as_completed(future_to_dataset), desc="Processing datasets"):
            dataset_path = future_to_dataset[future]
            try:
                if future.result():
                    print(f"Successfully processed dataset: {dataset_path}")
                else:
                    print(f"Failed to process dataset: {dataset_path}")
            except Exception as e:
                print(f"Error processing dataset {dataset_path}: {e}")
    
    print(f'finish!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process big dataset with error handling and tracking.')
    parser.add_argument('--BigDataset_path', type=str, required=True, help='Path to the big dataset.')
    parser.add_argument('--completed_file', type=str, default='./log/completed_datasets.txt', help='File to record completed datasets.')
    parser.add_argument('--error_file', type=str, default='./log/error.txt', help='File to record error messages.')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of concurrent workers.')
    parser.add_argument('--step1_complete', type=str, default='/home/e/eez095/project/dex-ycb-toolkit/log/completed_datasets.txt', help='step1-2 complete file')
    

    args = parser.parse_args()

    process_big_dataset(
        BigDataset_path=args.BigDataset_path,
        completed_file=args.completed_file,
        error_file=args.error_file,
        max_workers=args.max_workers,
        step1_complete = args.step1_complete
    )
