# multiprocess for data like '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341'
import subprocess
import argparse
import os
import re
from multiprocessing import Pool


def execute_commands(args):
    source_path, n_views = args
    output_path = os.path.join(source_path, 'FSGS_output')
    commands = [
        # f"python train.py  --source_path {source_path} --model_path {output_path} --n_views {n_views} --depth_pseudo_weight 0.03 --kk ",
        f"python render.py --source_path {source_path} --model_path {output_path} --iteration 10000 --kk --video"
    ]

    for cmd in commands:
        print(f"Executing: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            error_message = f"Error executing command: {cmd}\nError message: {e}\n"
            print(error_message)
            return False  # If there's an error, return False
        print(f"Command completed: {cmd}\n")

    print(f"All commands executed for {source_path}.")
    return True  # Return True if all commands are successfully executed


def process_dataset(dataset_path, n_views, max_threads):
    data_frames = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    tasks = [
        (os.path.join(dataset_path, data_frame), n_views)
        for data_frame in data_frames if re.match(r'\d+_frame$', os.path.basename(data_frame))
    ]

    # Use multiprocessing Pool with user-defined number of threads (max_threads)
    with Pool(processes=max_threads) as pool:
        results = pool.map(execute_commands, tasks)

    # Check for any failures
    for idx, result in enumerate(results):
        if not result:
            print(f"There was an error during execution: {tasks[idx][0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='task 5')

    parser.add_argument('--source_path', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/0_frame', help='dataset for one frame')
    parser.add_argument('--n_views', type=int, default=6)
    parser.add_argument('--dataset_path', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341')
    parser.add_argument('--BigDataset_path', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02')
    parser.add_argument('--max_threads', type=int, default=8, help='Maximum number of threads for multiprocessing')

    args = parser.parse_args()
    source_path = args.source_path
    n_views = args.n_views
    dataset_path = args.dataset_path
    max_threads = args.max_threads

    # Process the dataset in parallel with a customizable number of threads
    process_dataset(dataset_path, n_views, max_threads)
