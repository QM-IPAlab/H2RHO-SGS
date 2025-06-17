# fn_weight = 1 不加强对fn的监督
# inference:输入图像  输出应该移动的欧拉角+位姿
# 加入hand object掩码。hand mask:1,h,w object mask:1 h w. 更改
# loss归一化 并分开计算旋转与平移
# 输出中加入是否是pre grasp的分类标签
# fn_weight：现在设置的pre grasp class中，fn的loss的weight是fn_weight
# Ltrans Lrot Lclass: score:score:1 保持数值大概一致

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from PIL import Image
import re
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
import transforms3d
import matplotlib.pyplot as plt
from generate_mask import get_mask
from generate_mask import mv_background
import argparse
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
global_threshold = 0.9
score = 2 #ltrans:Lclass
fn_weight = 1
num_traj = 20

class FiveChannelTransform:
    def __init__(self):
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整尺寸
            transforms.ToTensor(),         # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 保持尺寸一致
            transforms.ToTensor(),          # 只做 ToTensor()，保持 0/1 分布
            transforms.Lambda(lambda x:( x > 0).float())
        ])
       
    def __call__(self, img, hand_mask, obj_mask):
        
        img_rgb = self.rgb_transform(img)  # 变成 (3, 224, 224)
        
        # 处理手部 & 物体掩码（2 通道）
        mask_hand = self.mask_transform(hand_mask)  # (1, 224, 224)
        mask_obj = self.mask_transform(obj_mask)    # (1, 224, 224)
        
        # indices = np.nonzero(np.array(mask_hand))
        # values = np.array(mask_hand)[indices]
        # print(f'2-----------------------values:{values}    img: {np.array(img_rgb)}\n hand_mask {np.array(mask_hand)} \n obj_mask {np.array(mask_obj)}\n ')
        
        # 拼接所有通道：RGB (3,224,224) + 掩码 (1,224,224) + 掩码 (1,224,224) = (5, 224, 224)
        # print(f'img_rgb.shape  {img_rgb.shape} mask_hand.shape {mask_hand.shape} mask_obj.shape{mask_obj.shape}')
        img_5ch = torch.cat([img_rgb, mask_hand, mask_obj], dim=0)
        
        # img_mask = torch.cat([ mask_hand, mask_obj], dim=0)
        # print(f'3----------img: {np.array(img_5ch)}\n')
        return img_5ch


def eular_R(euler_angles,translation):
    rotation_matrix = transforms3d.euler.euler2mat(euler_angles[0],euler_angles[1],euler_angles[2],'sxyz')

    # 构造 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation

    return T

   

def process_pose(pose_matrix):
    # 从4x4矩阵提取欧拉角和位移
    rotation_matrix = pose_matrix[:3, :3]
    translation = pose_matrix[:3, 3]
    
    # 将旋转矩阵转换为欧拉角 (假设使用xyz顺序)
    euler_angles = transforms3d.euler.mat2euler(rotation_matrix, 'sxyz')
    
    return np.concatenate([euler_angles, translation])
# Custom sort function to order filenames in a desired way
def custom_sort(filename):
    if '_target' in filename:
        return (float('inf'),)
    match = re.findall(r'(\d+)', filename)
    return tuple(int(num) for num in match)

# Save dataset paths to a text file
def save_to_txt(data, save_txt_path):
    with open(save_txt_path, 'w') as f:
        for data_item in data:
            line = f"{data_item['img_path']} {data_item['current_traj']} {data_item['next_traj']}\n"
            f.write(line)

# Dataset class to load images and trajectories
class CustomDataset(Dataset):
    def __init__(self, txt_path, transform=None, save_txt_path=None):
        self.transform = transform
        self.save_txt_path = save_txt_path
        self.data = []
        self.poses = []
        with open(txt_path, 'r') as file:
            for line in tqdm(file, desc="Loading file paths", unit=" lines"):
                try:
                    data_path = line.strip()
                    image_dir = os.path.join(data_path, "FSGS_output", "video", "sample_v610000")
                    trajectory_dir = os.path.join(data_path, "trajectory_v6")
                    hand_ply_file = os.path.join(data_path, 'handover_3D','hand.ply')
                    obj_ply_file = os.path.join(data_path, 'handover_3D','object.ply')
                    mask_dir = os.path.join(data_path, 'mask')
                    subfolders = sorted([f for f in os.listdir(image_dir) if not f.endswith(".mp4")])
                    i_ = 0
                    for subfolder in tqdm(subfolders,desc="for one scene"):
                        i_ = i_ +1
                        if i_ > num_traj:
                            break
                        try:
                            img_folder = os.path.join(image_dir, subfolder)
                            traj_folder = os.path.join(trajectory_dir, subfolder)
                            
                            img_files = sorted([f for f in os.listdir(img_folder) if not f.endswith(".mp4")], key=custom_sort)
                            traj_files = sorted([f for f in os.listdir(traj_folder) if not f.endswith(".mp4")], key=custom_sort)

                            for i in range(len(img_files) - 1):
                                if img_files[i].endswith("_target.png"):
                                    continue
                                elif img_files[i + 1].endswith("_target.png"):
                                    next_traj_path = os.path.join(traj_folder, traj_files[i])
                                    pre_grasp_class = 1.0
                                else:
                                    next_traj_path = os.path.join(traj_folder, traj_files[i + 1])
                                    # pre_grasp_class = 0.0
                                    # print(f'{traj_folder} {traj_files[i + 1]} {img_files[i + 1]}')
                                    pre_grasp_class = (float(i/len(img_files))) #before pregrasp， class score for pose <0.9
                                                        
                                hand_img, object_img = get_mask(os.path.join(traj_folder, traj_files[i]), hand_ply_file, obj_ply_file,mask_dir)
                                

                                self.data.append({
                                    "img_path": os.path.join(img_folder, img_files[i]),
                                    "current_traj": os.path.join(traj_folder, traj_files[i]),
                                    "next_traj": next_traj_path,
                                    "hand_mask":hand_img,
                                    "object_mask": object_img,
                                    "pre_grasp_class" : pre_grasp_class
                                })
                                # 计算 pose 用于归一化
                                current_traj = np.load(os.path.join(traj_folder, traj_files[i]))
                                next_traj = np.load(next_traj_path)
                                T_A_inv = np.linalg.inv(current_traj)
                                T_AB = np.dot(T_A_inv, next_traj)
                                pose = process_pose(T_AB)
                                self.poses.append(pose)
                        except Exception as e:
                            print(f"[ERROR] Failed to process frame {img_files[i]} in {subfolder}: {e}")
                            continue  # 继续处理下一个 frame
                except Exception as e:
                    print(f"[ERROR] Failed to process subfolder {line}: {e}")
                    continue  # 继续处理下一个 subfolder

        if self.save_txt_path:
            save_to_txt(self.data, self.save_txt_path)
        # 计算 pose 归一化参数
        self.poses = np.array(self.poses)
        self.pose_mean = np.mean(self.poses, axis=0)
        self.pose_std = np.std(self.poses, axis=0) + 1e-8  # 避免除零
        print(f'mean {self.pose_mean} std:{self.pose_std}')
    def __len__(self):
        print(len(self.data))
        return len(self.data)

    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        hand_mask = data_item["hand_mask"] # (1, H, W)
        object_mask = data_item["object_mask"]  # (1, H, W)
        img = Image.open(data_item["img_path"]).convert("RGB") 

        no_bg_img_path= '/home/e/eez095/project/motion/data/12'
        # pose = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145838/36_frame/trajectory_v6/0/0_3.npy'
        # hand_ply_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145838/36_frame/handover_3D/hand.ply'
        # object_pcl_file = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145838/36_frame/handover_3D/object.ply'
        # output_dir= '/home/e/eez095/project/motion/data/12'
        # img_path = '/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145838/36_frame/FSGS_output/video/sample_v610000/0/0_3.png'
       
        # hand_mask, object_mask = get_mask(pose, hand_ply_file, object_pcl_file, no_bg_img_path)
        # img =  Image.open(img_path).convert("RGB")
        img_no_back_ground = mv_background(img, hand_mask, object_mask, no_bg_img_path)
        
      
        if self.transform:
            img = self.transform(img_no_back_ground,hand_mask,object_mask)

        
        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])
        pre_grasp_class = data_item["pre_grasp_class"]
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot( T_A_inv,next_traj)
        pose = process_pose(T_AB)
       
        pose = (pose - self.pose_mean) / self.pose_std  # 对每个维度单独归一化

        
        return img, torch.tensor(pose, dtype=torch.float32),pre_grasp_class

# Dataset class for training and testing
class PoseDataset(Dataset):
    def __init__(self, data, transform=None,pose_mean=None,pose_std=None):
        self.data = data
        self.transform = transform
        self.pose_mean = pose_mean
        self.pose_std = pose_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
       
        hand_mask = data_item["hand_mask"] # (1, H, W)
        object_mask = data_item["object_mask"]  # (1, H, W)
        img_path = data_item["img_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except OSError:
            print(f"Warning: Skipping corrupted image {img_path}")
            return None  # 或者返回默认值
        
        no_bg_img_path= '/home/e/eez095/project/motion/data/12'
        img_no_back_ground = mv_background(img, hand_mask, object_mask, no_bg_img_path)
        
        
        if self.transform:
            img = self.transform(img_no_back_ground,hand_mask,object_mask)

        
        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])
        pre_grasp_class = data_item["pre_grasp_class"]
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot( T_A_inv,next_traj)
        pose = process_pose(T_AB)
        pose = (pose - self.pose_mean) / self.pose_std
        return img, torch.tensor(pose, dtype=torch.float32),pre_grasp_class

# Model definition: Pose Estimation using ResNet18
class PoseEstimationModel(nn.Module):
    def __init__(self ):
        super(PoseEstimationModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        # 在 ResNet 之前添加一个 1x1 卷积，将 5 通道压缩成 3 通道
        self.channel_reduction = nn.Conv2d(
            in_channels=5,  # 输入 2 通道 ( hand_mask + object_mask)
            out_channels=3,  # 到 3 通道，匹配 ResNet
            kernel_size=1,   # 1x1 卷积，不改变空间分辨率
            stride=1,
            padding=0,
            bias=False
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 7)  # 3个欧拉角 + 3个位移 + pregrasp分类
        self.sigmoid = nn.Sigmoid()  # 用于 `pre-grasp` 任务

    def forward(self, x):
        x = self.channel_reduction(x)
        output = self.backbone(x)
        pose = output[:, :6]  # 前6维是 欧拉角 + 位移
        pre_grasp = output[:, 6] # 最后一维是分类概率（是否是 pre-grasp）
        # pre_grasp =torch.sigmoid(pre_grasp)
        # print(pose.dtype, pre_grasp.dtype)
        return pose, pre_grasp

def custom_bce_loss(pred, target):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    fn_mask = (target == 1) & (pred < global_threshold)  # 找到 FN（目标=1 但预测<0.5）
    loss[fn_mask] *= fn_weight  # 对 FN 处的 loss 进行放大
    return loss.mean()

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss_custom = nn.SmoothL1Loss(beta=0.5)  # 设置 δ = 0.5
        # l1_loss = self.smooth_l1_loss_custom(pred, target)
        # pos_weight_value = 100 
        # # class 0 / class 1
        # pos_weight = torch.tensor([pos_weight_value], device="cuda") 
        # self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.bce_loss = torch.nn.BCELoss()
    
    def forward(self, pred_pose, pred_class, target_pose, target_class):
        # 预测值和真实值的分离（3个旋转 + 3个平移）
        
        pred_rot, pred_trans = pred_pose[:, :3], pred_pose[:, 3:]
        target_rot, target_trans = target_pose[:, :3], target_pose[:, 3:]
        
       

        # 计算归一化后的 MSE 损失
        rot_loss = self.mse_loss(pred_rot, target_rot)
        trans_loss = self.mse_loss(pred_trans, target_trans)

        # print(pred_class.dtype,target_class.dtype)
        class_loss = custom_bce_loss(pred_class, target_class)
        
        # 总损失 = 旋转损失 + 位移损失
        total_loss = score *rot_loss + score*trans_loss + class_loss
        return total_loss, rot_loss, trans_loss, class_loss

# Training and evaluation function with mixed precision
def train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device='cuda',model_path='./best_pose_estimation_model_v10_0209.pth'):
    model.to(device)
    best_test_loss = float('inf')
    scaler = GradScaler()

    # Lists to store loss values for plotting
    train_losses_rot, train_losses_trans, train_losses_class, train_losses = [], [], [], []  
    test_losses_rot, test_losses_trans, test_losses_class, test_losses = [], [], [], []

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    log_file = f"./{model_name}_training_log.txt"  # 设定日志文件名

    for epoch in range(epochs):
        model.train()
        running_total_loss, running_total_rot_loss, running_total_trans_loss, running_class_loss = 0.0, 0.0, 0.0, 0.0
        for imgs, poses, labels in train_dataloader:
            imgs, poses, labels = imgs.to(device), poses.to(device), labels.to(device)
           

            optimizer.zero_grad()
            with autocast():
                pred_pose, pred_class = model(imgs)
                loss, rot_loss, trans_loss, class_loss = criterion(pred_pose, pred_class, poses, labels)
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total_loss += loss.item()
            running_total_rot_loss += rot_loss.item()
            running_total_trans_loss += trans_loss.item()
            running_class_loss += class_loss.item()
        
        avg_total_train_loss = running_total_loss / len(train_dataloader)
        avg_rot_train_loss = running_total_rot_loss / len(train_dataloader)
        avg_trans_train_loss = running_total_trans_loss / len(train_dataloader)
        avg_class_loss = running_class_loss / len(train_dataloader)
        train_losses.append(avg_total_train_loss)  # Store train loss for plotting
        train_losses_rot.append(avg_rot_train_loss)
        train_losses_trans.append(avg_trans_train_loss)
        train_losses_class.append(avg_class_loss)


        with open(log_file, "a") as f:  # 以追加模式 ("a") 打开文件，防止覆盖已有内容
            f.write(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_total_train_loss:.4f}, "
                    f"Rot loss: {avg_rot_train_loss:.4f}, Trans Loss: {avg_trans_train_loss:.4f}, "
                    f"Avg Class Loss: {avg_class_loss:.4f}\n")

        # print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_total_train_loss:.4f}, Rot loss:{avg_rot_train_loss:.4f}, Trans Loss: {avg_trans_train_loss:.4f}, avg_class_loss:{avg_class_loss:.4f}")

        avg_total_test_loss,avg_rot_test_loss,avg_test_trans_loss, avg_class_loss = evaluate(model, test_dataloader, criterion, device,'train')
        test_losses.append(avg_total_test_loss)  # Store test loss for plotting
        test_losses_rot.append(avg_rot_test_loss)
        test_losses_trans.append(avg_test_trans_loss)
        test_losses_class.append(avg_class_loss)

        if avg_total_test_loss < best_test_loss:
            best_test_loss = avg_total_test_loss
            torch.save(model.state_dict(), model_path)
            with open(log_file, "a") as f:  # 以追加模式 ("a") 打开文件，防止覆盖已有内容
                f.write(f"save best model ")
    
    # Plot loss curves
    plt.ylim(0, 1.5) 
    plt.yticks(np.arange(0, 1.5, 0.1))  # 0.1 间隔
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Total Loss', marker='o', color='blue')
    plt.plot(range(1, epochs + 1), train_losses_rot, label='Train Rotation Loss', marker='s', color='deepskyblue')
    plt.plot(range(1, epochs + 1), train_losses_trans, label='Train Translation Loss', marker='^', color='purple')
    plt.plot(range(1, epochs + 1), train_losses_class, label='Train Grasp Class Loss', marker='p', color='navy')

    plt.plot(range(1, epochs + 1), test_losses, label='Test Total Loss', marker='x', color='red')
    plt.plot(range(1, epochs + 1), test_losses_rot, label='Test Rotation Loss', marker='d', color='orangered')
    plt.plot(range(1, epochs + 1), test_losses_trans, label='Test Translation Loss', marker='v', color='darkorange')
    plt.plot(range(1, epochs + 1), test_losses_class, label='Test Grasp Class Loss', marker='*', color='gold')

   

    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'./log/{model_name}_loss_convergence.png')  # Save the figure
    plt.show()

# Evaluate function
def evaluate(model, test_dataloader, criterion, device='cuda',mode='train'):
    model.eval()
    test_total_loss, test_total_rot_loss, test_total_trans_loss, test_total_class_loss  = 0.0, 0.0, 0.0, 0.0
    gt_rot_poses, gt_trans_poses ,pred_rot_poses ,pred_trans_poses, gt_label, predict_label = [], [], [], [], [], []  # Predicted translation
    
    with torch.no_grad():
        for imgs, poses, labels in test_dataloader:
            imgs, poses, labels = imgs.to(device), poses.to(device), labels.to(device)
            pred_pose, pred_class = model(imgs)
            # Append ground truth and predicted values
            # print(pred_class.dtype,labels)
            loss, rot_loss, trans_loss, class_loss = criterion(pred_pose, pred_class, poses, labels)
            log_file = "./training_log.txt"  # 设定日志文件名
            with open(log_file, "a") as f:  
                f.write(f'pred_pose:{pred_pose}\n poses {poses} \n')  

            test_total_loss += loss.item()
            test_total_rot_loss += rot_loss.item()
            test_total_trans_loss += trans_loss.item()
            test_total_class_loss += class_loss.item()

            gt_rot_poses.append(poses[:, :3].cpu().numpy())
            gt_trans_poses.append(poses[:, 3:].cpu().numpy())
            pred_rot_poses.append(pred_pose[:, :3].cpu().numpy())
            pred_trans_poses.append(pred_pose[:, 3:].cpu().numpy())
            gt_label.append(labels.cpu().numpy())
            pred_class_ = torch.sigmoid(pred_class)
        
            with open(log_file, "a") as f:  
                f.write(f'pred_class_: {pred_class_}\n gt class:{labels}\n')  

            pred_class_ = (pred_class_ > global_threshold).float()
            predict_label.append(pred_class_.cpu().numpy())
        
    avg_total_test_loss = test_total_loss / len(test_dataloader)
    avg_rot_test_loss = test_total_rot_loss / len(test_dataloader)
    avg_trans_test_loss = test_total_trans_loss / len(test_dataloader)
    avg_test_class_loss = test_total_class_loss / len(test_dataloader)

    gt_rot_poses = np.concatenate(gt_rot_poses, axis=0)
    gt_trans_poses = np.concatenate(gt_trans_poses, axis=0)
    pred_rot_poses = np.concatenate(pred_rot_poses, axis=0)
    pred_trans_poses = np.concatenate(pred_trans_poses, axis=0)
    gt_label = np.concatenate(gt_label, axis=0)
    predict_label = np.concatenate(predict_label, axis=0)

    print(f" testTotal Loss: {avg_total_test_loss:.4f}, Rot loss:{avg_rot_test_loss:.4f}, Trans Loss: {avg_trans_test_loss:.4f}, class loss: {avg_test_class_loss:.4f}")
    if mode == 'train':
        return avg_total_test_loss,avg_rot_test_loss,avg_trans_test_loss,avg_test_class_loss
    if mode == 'evaluate':
        print('there')
        return gt_rot_poses, gt_trans_poses, pred_rot_poses, pred_trans_poses,gt_label,predict_label
   

def test_pose(image_path, traj_path, model, transform, next_pose_path,hand_ply_file=None,obj_ply_file=None,mask_dir=None,pose_std=None, pose_mean=None):
    device='cuda'
    model.to(device)
    if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
    model = DataParallel(model)  # Enable multi-GPU support
    model.eval()

    img = Image.open(image_path).convert("RGB")

    data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(image_path)))))

    if hand_ply_file == None:
        trajectory_dir = os.path.join(data_path, "trajectory_v6")
        hand_ply_file = os.path.join(data_path, 'handover_3D','hand.ply')
        obj_ply_file = os.path.join(data_path, 'handover_3D','object.ply')
        mask_dir = os.path.join(data_path, 'mask')

    hand_mask, object_mask = get_mask( traj_path, hand_ply_file, obj_ply_file,mask_dir)
    
    no_bg_img_path= '/home/e/eez095/project/motion/data/12'
    img_no_back_ground = mv_background(img, hand_mask, object_mask, no_bg_img_path)


    img_transformed = transform(img_no_back_ground, hand_mask, object_mask)

    img_transformed = img_transformed.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_pose, pred_class = model(img_transformed)  # 预测的 6 维姿态 (3 旋转 + 3 平移)
    pose_std = torch.tensor(pose_std,device=device)
    pose_mean = torch.tensor(pose_mean,device=device)
    pred_pose = pred_pose * pose_std + pose_mean
    
    pred_class = torch.sigmoid(pred_class)
    print(f'pred_class: {pred_class}')
    pred_class = (pred_class > global_threshold).float()

    predicted_euler_angles = pred_pose[0, :3].cpu().numpy()  # 旋转部分
    predicted_translation = pred_pose[0, 3:].cpu().numpy()   # 平移部分
    predict_class = pred_class.cpu().numpy()

    print(f"Predicted Euler Angles: {predicted_euler_angles}")
    print(f"Predicted Translation: {predicted_translation}")
    print(f"predict_class:{predict_class}")

    T = eular_R(predicted_euler_angles,predicted_translation)
    

    T_ori = np.load(traj_path)
    next_pose = np.dot(T_ori,T)
    np.save(next_pose_path,next_pose)
    
    return next_pose


# Function to load the trained model
def load_model(model_path, device='cuda'):
    model = PoseEstimationModel()
    state_dict = torch.load(model_path)

    # 处理 DataParallel 兼容性问题
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # 去掉 "module." 前缀
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)  # 先加载参数
    model.to(device)

    # 如果是多 GPU 运行，则包装成 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)
  
    
    return model

def evaluate_only(model, txt_path, transform, criterion, device='cuda'):
    # Load the dataset
    dataset = CustomDataset(txt_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()  # Set the model to evaluation mode

    running_loss = 0.0
    running_rot_loss = 0.0
    running_trans_loss = 0.0

    gt_rot_poses = []  # Ground truth rotation
    gt_trans_poses = []  # Ground truth translation
    pred_rot_poses = []  # Predicted rotation
    pred_trans_poses = []  # Predicted translation

    # Disable gradient calculation for inference
    with torch.no_grad():
        for imgs, poses in dataloader:
            imgs, poses = imgs.to(device), poses.to(device)
            
            # Get the model's predictions
            predicted_pose = model(imgs)

            # Compute separate rotation and translation losses
            rot_loss = criterion(predicted_pose[:, :3], poses[:, :3])  # Rotation loss
            trans_loss = criterion(predicted_pose[:, 3:], poses[:, 3:])  # Translation loss
            loss = rot_loss + trans_loss  # Total loss

            running_loss += loss.item()
            running_rot_loss += rot_loss.item()
            running_trans_loss += trans_loss.item()

            # Append ground truth and predicted values
            gt_rot_poses.append(poses[:, :3].cpu().numpy())
            gt_trans_poses.append(poses[:, 3:].cpu().numpy())
            pred_rot_poses.append(predicted_pose[:, :3].cpu().numpy())
            pred_trans_poses.append(predicted_pose[:, 3:].cpu().numpy())

    # Compute average losses
    avg_loss = running_loss / len(dataloader)
    avg_rot_loss = running_rot_loss / len(dataloader)
    avg_trans_loss = running_trans_loss / len(dataloader)

    print(f"Average Test Loss: {avg_loss:.4f}, Rotation Loss: {avg_rot_loss:.4f}, Translation Loss: {avg_trans_loss:.4f}")

    # Flatten the lists for easier comparison
    gt_rot_poses = np.concatenate(gt_rot_poses, axis=0)
    gt_trans_poses = np.concatenate(gt_trans_poses, axis=0)
    pred_rot_poses = np.concatenate(pred_rot_poses, axis=0)
    pred_trans_poses = np.concatenate(pred_trans_poses, axis=0)

    return avg_loss, avg_rot_loss, avg_trans_loss, gt_rot_poses, gt_trans_poses, pred_rot_poses, pred_trans_poses

# Main function to load data, create dataloaders, and start training
def main(mode, train_txt_path, test_txt_path, save_txt_path, test_image_path, evaluate_txt_path, model_path, test_traj_path, next_pose_path,hand_ply_file,obj_ply_file,mask_dir):
    

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 实例化 transform
    five_channel_transform = FiveChannelTransform()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    if mode == 'train':
        train_dataset = CustomDataset(train_txt_path, transform=five_channel_transform, save_txt_path=save_txt_path)
        test_dataset = CustomDataset(test_txt_path, transform=five_channel_transform, save_txt_path=save_txt_path)


        pose_mean = train_dataset.pose_mean
        pose_std =  train_dataset.pose_std
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        np.save(f"./{model_name}pose_mean.npy", pose_mean)
        np.save(f"./{model_name}pose_std.npy", pose_std)
        # train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)
        # train_dataset = PoseDataset(train_data, transform=five_channel_transform,pose_mean = pose_mean,pose_std= pose_std)
        # test_dataset = PoseDataset(test_data, transform=five_channel_transform,pose_mean = pose_mean,pose_std= pose_std)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, pin_memory=True)
        model = PoseEstimationModel()
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)  # Enable multi-GPU support
        criterion  = PoseLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=50, device=device,model_path = model_path)

    elif mode == 'test':
        model = load_model(model_path, device='cuda')
        criterion  = PoseLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        model_name =  os.path.splitext(os.path.basename(model_path))[0]
        pose_mean = np.load(f"./{model_name}pose_mean.npy")
        pose_std = np.load(f"./{model_name}pose_std.npy")
        predicted_pose = test_pose(test_image_path, test_traj_path, model, five_channel_transform,next_pose_path,hand_ply_file,obj_ply_file,mask_dir,pose_std ,pose_mean)
        print(f'next pose: {predicted_pose}')

    elif mode == 'evaluate':
        model = load_model(model_path, device='cuda')
        evaluate_dataset = CustomDataset(evaluate_txt_path, transform=five_channel_transform, save_txt_path=save_txt_path)
        # evaluate_dataset = PoseDataset(evaluate_dataset, transform=five_channel_transform)
        evaluate_dataloader = DataLoader(evaluate_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        gt_rot_poses, gt_trans_poses, pred_rot_poses, pred_trans_poses,gt_label,predict_label= evaluate(model, evaluate_dataloader, criterion, device='cuda',mode = 'evaluate')
       
        # print(f"Pose differences for first :\n gt_rot_poses {gt_rot_poses[:10]}\n, gt_trans_poses {gt_trans_poses[:10]}\n, pred_rot_poses {pred_rot_poses[:10]}\n, pred_trans_poses {pred_trans_poses[:10]}\n,gt_label {gt_label[:10]}\n,predict_label {predict_label[:10]}")  # 打印前10个样本的差异
        # print(f"Pose differences for first :\n gt_rot_poses {gt_rot_poses}\n, gt_trans_poses {gt_trans_poses}\n, pred_rot_poses {pred_rot_poses}\n, pred_trans_poses {pred_trans_poses}\n,gt_label {gt_label}\n,predict_label {predict_label}")  # 打印前10个样本的差异

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Estimation")
    parser.add_argument('--mode', choices=['train', 'test', 'evaluate'], required=True, help="Mode to run: 'train', 'test' or 'evaluate'")
    parser.add_argument('--train_txt_path', type=str, default="/home/e/eez095/project/policy_learning/dataset.txt", help="Path to the input dataset text file")
    parser.add_argument('--test_txt_path', type=str, default="/home/e/eez095/project/policy_learning/dataset.txt", help="Path to the input dataset text file")
    parser.add_argument('--save_txt_path', type=str, default="/home/e/eez095/project/policy_learning/getitem_data.txt", help="to check what is inside the dataset")
    test_image_path = "/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/59_frame/FSGS_output/video/ours_10000/12/12_4.png"
    parser.add_argument('--model_path', type=str, default="./model/best_pose_estimation_model_v10_0209.pth", help="Path to the pre-trained model file")
    parser.add_argument('--test_image_path', type=str, help="Path to the test image")
    parser.add_argument('--evaluate_txt_path', type=str, default="/home/e/eez095/project/policy_learning/dataset_evaluate.txt", help="Path to the evaluation dataset text file")
    parser.add_argument('--test_traj_path',type=str,help='test image trajectory')

    parser.add_argument('--next_pose_path',type=str,help='next_pose_path')
    parser.add_argument('--hand_ply_file',type=str,help='hand_ply_file')
    parser.add_argument('--obj_ply_file',type=str,help='obj_ply_file')
    parser.add_argument('--mask_dir',type=str,help='mask_dir')

    args = parser.parse_args()

    main(args.mode,  args.train_txt_path, args.test_txt_path, args.save_txt_path, args.test_image_path, args.evaluate_txt_path, args.model_path, args.test_traj_path,args.next_pose_path,args.hand_ply_file,args.obj_ply_file,args.mask_dir)
