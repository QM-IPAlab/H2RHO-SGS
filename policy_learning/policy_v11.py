#inference:输入图像  输出应该移动的欧拉角+位姿
# 加入hand object掩码。hand mask:1,h,w object mask:1 h w. 更改
# loss归一化 并分开计算旋转与平移
# 输出中加入是否是pre grasp的分类标签
# 增强class=1的学习
# 
# fn_weight=200：现在设置的pre grasp class中，fn的loss的weight是fn_weight=200
# Ltrans Lrot Lclass: 50:50:1 保持数值大概一致
# class是线性增长
# when test, input is mask image
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
import argparse
import torch
import torchvision.transforms as transforms
global_threshold = 0.9
score = 200 #ltrans:Lclass
fn_weight = 50

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
        # indices = np.nonzero(np.array(hand_mask))
        # values = np.array(hand_mask)[indices]
        # print(f'----------------values:{values}')
        # print(f'img: {np.array(img)} hand_mask {np.array(hand_mask)} obj_mask {np.array(obj_mask)}')
        # 处理 RGB 图像（前 3 通道）
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
        with open(txt_path, 'r') as file:
            for line in file:
                data_path = line.strip()
                image_dir = os.path.join(data_path, "FSGS_output", "video", "sample_v410000")
                trajectory_dir = os.path.join(data_path, "trajectory_v4")
                hand_ply_file = os.path.join(data_path, 'handover_3D','hand.ply')
                obj_ply_file = os.path.join(data_path, 'handover_3D','object.ply')
                mask_dir = os.path.join(data_path, 'mask')
                subfolders = sorted([f for f in os.listdir(image_dir) if not f.endswith(".mp4")])

                for subfolder in subfolders:
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
                            pre_grasp_class = (float(i/len(img_files)))*global_threshold #before pregrasp， class score for pose <0.9
                                                 
                        hand_img, object_img = get_mask(os.path.join(traj_folder, traj_files[i]), hand_ply_file, obj_ply_file,mask_dir)
                        

                        self.data.append({
                            "img_path": os.path.join(img_folder, img_files[i]),
                            "current_traj": os.path.join(traj_folder, traj_files[i]),
                            "next_traj": next_traj_path,
                            "hand_mask":hand_img,
                            "object_mask": object_img,
                            "pre_grasp_class" : pre_grasp_class
                        })

        if self.save_txt_path:
            save_to_txt(self.data, self.save_txt_path)

    def __len__(self):
        return len(self.data)

    
    # def __getitem__(self, idx):
    #     data_item = self.data[idx]
    #     hand_mask = torch.tensor(data_item["hand_mask"], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
    #     object_mask = torch.tensor(data_item["object_mask"], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
    #     img = Image.open(data_item["img_path"]).convert("RGB")
    #     img = torch.cat([img, hand_mask, object_mask], dim=0)
   
    #     if self.transform:
    #         img = self.transform(img)
    #     current_traj = np.load(data_item["current_traj"])
    #     next_traj = np.load(data_item["next_traj"])
    #     T_A_inv = np.linalg.inv(current_traj)
    #     T_AB = np.dot( next_traj,T_A_inv)
    #     pose = process_pose(T_AB)
    #     pre_grasp_class = data_item["pre_grasp_class"]
    #     return img, torch.tensor(pose, dtype=torch.float32),pre_grasp_class
    def __getitem__(self, idx):
        data_item = self.data[idx]
        hand_mask = data_item["hand_mask"] # (1, H, W)
        object_mask = data_item["object_mask"]  # (1, H, W)
        img = Image.open(data_item["img_path"]).convert("RGB") 
      
        if self.transform:
            img = self.transform(img,hand_mask,object_mask)

        
        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])
        pre_grasp_class = data_item["pre_grasp_class"]
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot( next_traj,T_A_inv)
        pose = process_pose(T_AB)
        
        return img, torch.tensor(pose, dtype=torch.float32),pre_grasp_class

# Dataset class for training and testing
class PoseDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
       
        hand_mask = data_item["hand_mask"] # (1, H, W)
        object_mask = data_item["object_mask"]  # (1, H, W)
        img = Image.open(data_item["img_path"]).convert("RGB") 
      
        if self.transform:
            img = self.transform(img,hand_mask,object_mask)

        
        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])
        pre_grasp_class = data_item["pre_grasp_class"]
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot( next_traj,T_A_inv)
        pose = process_pose(T_AB)
        return img, torch.tensor(pose, dtype=torch.float32),pre_grasp_class

# Model definition: Pose Estimation using ResNet18
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        # 在 ResNet 之前添加一个 1x1 卷积，将 5 通道压缩成 3 通道
        self.channel_reduction = nn.Conv2d(
            in_channels=5,  # 输入 5 通道 (RGB + hand_mask + object_mask)
            out_channels=3,  # 降维到 3 通道，匹配 ResNet
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
        # pos_weight_value = 100 
        # # class 0 / class 1
        # pos_weight = torch.tensor([pos_weight_value], device="cuda") 
        # self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.bce_loss = torch.nn.BCELoss()
    
    def forward(self, pred_pose, pred_class, target_pose, target_class):
        # 预测值和真实值的分离（3个旋转 + 3个平移）
        
        pred_rot, pred_trans = pred_pose[:, :3], pred_pose[:, 3:]
        target_rot, target_trans = target_pose[:, :3], target_pose[:, 3:]
        
        # # 计算旋转和位移的均值和标准差，进行归一化
        # rot_mean, rot_std = target_rot.mean(dim=0), target_rot.std(dim=0) + 1e-6
        # trans_mean, trans_std = target_trans.mean(dim=0), target_trans.std(dim=0) + 1e-6
        
        # target_rot = (target_rot - rot_mean) / rot_std
        # target_trans = (target_trans - trans_mean) / trans_std
        # pred_rot = (pred_rot - rot_mean) / rot_std
        # pred_trans = (pred_trans - trans_mean) / trans_std
        
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

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_total_train_loss:.4f}, Rot loss:{avg_rot_train_loss:.4f}, Trans Loss: {avg_trans_train_loss:.4f}, avg_class_loss:{avg_class_loss:.4f}")

        avg_total_test_loss,avg_rot_test_loss,avg_test_trans_loss, avg_class_loss = evaluate(model, test_dataloader, criterion, device,'train')
        test_losses.append(avg_total_test_loss)  # Store test loss for plotting
        test_losses_rot.append(avg_rot_test_loss)
        test_losses_trans.append(avg_test_trans_loss)
        test_losses_class.append(avg_class_loss)

        if avg_total_test_loss < best_test_loss:
            best_test_loss = avg_total_test_loss
            torch.save(model.state_dict(), model_path)
            print("Saved best model!")
    
    # Plot loss curves
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
    plt.savefig('loss_convergence.png')  # Save the figure
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
            print(f'pred_class_: {pred_class_}\n labels:{labels}\n')
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

    print(f" Total Loss: {avg_total_test_loss:.4f}, Rot loss:{avg_rot_test_loss:.4f}, Trans Loss: {avg_trans_test_loss:.4f}, class loss: {avg_test_class_loss:.4f}")
    if mode == 'train':
        return avg_total_test_loss,avg_rot_test_loss,avg_trans_test_loss,avg_test_class_loss
    if mode == 'evaluate':
        print('there')
        return gt_rot_poses, gt_trans_poses, pred_rot_poses, pred_trans_poses,gt_label,predict_label
   

def test_pose(image_path, traj_path, model, transform, next_pose_path,object_mask_img_file,hand_mask_file):
    
    device='cuda'
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")

    data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(image_path)))))

    # hand_mask, object_mask = get_mask( traj_path, hand_ply_file, obj_ply_file,mask_dir)
    hand_mask =  Image.open(hand_mask_file).convert('L')  
    object_mask = Image.open(object_mask_img_file).convert('L')

    img_transformed = transform(img, hand_mask, object_mask)

    img_transformed = img_transformed.unsqueeze(0).to(device)
    # print(f'when test {img_transformed}')

    with torch.no_grad():
        pred_pose, pred_class = model(img_transformed)  # 预测的 6 维姿态 (3 旋转 + 3 平移)
    
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
    next_pose = np.dot(T,T_ori)
    np.save(next_pose_path,next_pose)
    
    return next_pose


# Function to load the trained model
def load_model(model_path, device='cuda'):
    model = PoseEstimationModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
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
def main(mode, txt_path, save_txt_path, test_image_path, evaluate_txt_path, model_path, test_traj_path, next_pose_path,hand_ply_file, obj_ply_file, mask_dir, object_mask_img_file, hand_mask_file):
    model = PoseEstimationModel()
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)  # Enable multi-GPU support

    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 实例化 transform
    five_channel_transform = FiveChannelTransform()

    

    
    criterion  = PoseLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

   

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if mode == 'train':
        dataset = CustomDataset(txt_path, transform=five_channel_transform, save_txt_path=save_txt_path)
        train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)

        
        train_dataset = PoseDataset(train_data, transform=five_channel_transform)
        test_dataset = PoseDataset(test_data, transform=five_channel_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=20, device=device,model_path = model_path)

    elif mode == 'test':
        model = load_model(model_path, device='cuda')
       
        predicted_pose = test_pose(test_image_path, test_traj_path, model, five_channel_transform,next_pose_path,object_mask_img_file, hand_mask_file)
        print(f'next pose: {predicted_pose}')

    elif mode == 'evaluate':
        model = load_model(model_path, device='cuda')
        evaluate_dataset = CustomDataset(evaluate_txt_path, transform=five_channel_transform, save_txt_path=save_txt_path)
        # evaluate_dataset = PoseDataset(evaluate_dataset, transform=five_channel_transform)
        evaluate_dataloader = DataLoader(evaluate_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        gt_rot_poses, gt_trans_poses, pred_rot_poses, pred_trans_poses,gt_label,predict_label= evaluate(model, evaluate_dataloader, criterion, device='cuda',mode = 'evaluate')
       
        print(f"Pose differences for first :\n gt_rot_poses {gt_rot_poses[:1]}\n, gt_trans_poses {gt_trans_poses[:1]}\n, pred_rot_poses {pred_rot_poses[:1]}\n, pred_trans_poses {pred_trans_poses[:1]}\n,gt_label {gt_label[:1]}\n,predict_label {predict_label[:1]}")  # 打印前10个样本的差异
        print(f"Pose differences for first :\n gt_rot_poses {gt_rot_poses}\n, gt_trans_poses {gt_trans_poses}\n, pred_rot_poses {pred_rot_poses}\n, pred_trans_poses {pred_trans_poses}\n,gt_label {gt_label}\n,predict_label {predict_label}")  # 打印前10个样本的差异

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose Estimation")
    parser.add_argument('--mode', choices=['train', 'test', 'evaluate'], required=True, help="Mode to run: 'train', 'test' or 'evaluate'")
    parser.add_argument('--txt_path', type=str, default="/home/e/eez095/project/policy_learning/dataset.txt", help="Path to the input dataset text file")
    parser.add_argument('--save_txt_path', type=str, default="/home/e/eez095/project/policy_learning/getitem_data.txt", help="to check what is inside the dataset")
    test_image_path = "/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/59_frame/FSGS_output/video/ours_10000/12/12_4.png"
    parser.add_argument('--model_path', type=str, default="./best_pose_estimation_model_v10_0209.pth", help="Path to the pre-trained model file")
    parser.add_argument('--test_image_path', type=str, help="Path to the test image")
    parser.add_argument('--evaluate_txt_path', type=str, default="/home/e/eez095/project/policy_learning/dataset_evaluate.txt", help="Path to the evaluation dataset text file")
    parser.add_argument('--test_traj_path',type=str,help='test image trajectory')
    parser.add_argument('--next_pose_path',type=str,help='next_pose_path')
    parser.add_argument('--hand_ply_file',type=str,default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_152926/36_frame/handover_3D/hand.ply',help='hand_ply_file')
    parser.add_argument('--obj_ply_file',type=str,default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_152926/36_frame//handover_3D/object.ply',help='obj_ply_file')
    parser.add_argument('--mask_dir',type=str,default='/home/e/eez095/project/motion/data/8',help='mask_dir')

    parser.add_argument('--object_mask_file',type=str,help='object_mask_img_file')
    parser.add_argument('--hand_mask_file', type=str, help="hand_mask_file")


    args = parser.parse_args()

    main(args.mode, args.txt_path, args.save_txt_path, args.test_image_path, args.evaluate_txt_path, args.model_path, args.test_traj_path,args.next_pose_path,args.hand_ply_file,args.obj_ply_file,args.mask_dir,args.object_mask_file,args.hand_mask_file)
