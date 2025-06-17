import os

def get_subfolders(root_dir):
    """获取 root_dir 下的所有孙子文件夹路径"""
    subfolders = []
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path):
                    subfolders.append(subfolder_path)
    return subfolders

def get_recorded_folders(txt_file):
    """从 txt 文件中读取已记录的文件夹路径"""
    recorded_folders = set()
    with open(txt_file, 'r') as f:
        for line in f:
            recorded_folders.add(line.strip())
    return recorded_folders

def find_unrecorded_folders(root_dir, txt_file):
    """找出未被记录的孙子文件夹"""
    subfolders = get_subfolders(root_dir)
    recorded_folders = get_recorded_folders(txt_file)
    
    unrecorded_folders = [folder for folder in subfolders if folder not in recorded_folders]
    return unrecorded_folders

import re
if __name__ == "__main__":
    root_directory = "/home/e/eez095/dexycb_data/20200813-subject-02"
    txt_file_path = "/home/e/eez095/project/dex-ycb-toolkit/log/completed_datasets.txt"  
    unprocess_file = "/home/e/eez095/project/dex-ycb-toolkit/log/unprocess_data.txt"  
    unrecorded = find_unrecorded_folders(root_directory, txt_file_path)
    with open(unprocess_file, 'w') as ef:
        for folder in unrecorded:
            if re.match(r'\d+_frame$', os.path.basename(folder)):
                 ef.write(f"{folder}\n")
