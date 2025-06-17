import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        """
        Args:
            txt_path (str): Path to the txt file containing data directories.
            transform (callable, optional): Image transformations to apply.
        """
        self.transform = transform

        # Parse txt file to gather data paths
        self.data = []
        with open(txt_path, 'r') as file:
            for line in file:
                data_path = line.strip()

                # Define paths for images and trajectories
                image_dir = os.path.join(data_path, "FSGS_output", "video", "ours_10000")
                trajectory_dir = os.path.join(data_path, "trajectory")

                # List all subfolders
                subfolders = sorted(os.listdir(image_dir))
                for subfolder in subfolders:
                    if subfolder.endswith(".mp4"):
                        continue  # Skip .mp4 files

                    img_folder = os.path.join(image_dir, subfolder)
                    traj_folder = os.path.join(trajectory_dir, subfolder)

                    # Get all image and trajectory files
                    img_files = sorted(os.listdir(img_folder))
                    traj_files = sorted(os.listdir(traj_folder))

                    # Pair images and their corresponding trajectory changes
                    for i in range(len(img_files) - 2):
                        if img_files[i].endswith("_target.png") or img_files[i].endswith(".mp4"):
                            continue  # Skip _target.png and .mp4 files

                        # Handle the case where the next trajectory is the same as the current
                        next_traj_path = os.path.join(traj_folder, traj_files[i + 1])
                        if img_files[i + 1].endswith("_target.png"):
                            next_traj_path = os.path.join(traj_folder, traj_files[i])

                        self.data.append({
                            "img_path": os.path.join(img_folder, img_files[i]),
                            "current_traj": os.path.join(traj_folder, traj_files[i]),
                            "next_traj": next_traj_path,
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch data entry
        data_item = self.data[idx]

        # Load image
        img = Image.open(data_item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Load trajectories and compute pose change
        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])
        # 计算 T_A 的逆矩阵
        T_A_inv = np.linalg.inv(current_traj)
        # 计算从A到B的变换矩阵 T_AB
        T_AB = np.dot(T_A_inv, next_traj)

         

        return img, torch.tensor(T_AB, dtype=torch.float32)


# Path to the txt file containing dataset paths
txt_path = "/home/e/eez095/project/policy_learning/dataset.txt"

# Define transformations
# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize dataset
dataset = CustomDataset(txt_path, transform=transform)

# Data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        # 使用预训练的 ResNet18
        self.backbone = models.resnet18(pretrained=True)
        # 修改最后一层以输出 16 个值（4x4 的位姿）
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 16)

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 4, 4)  # 将输出调整为 [batch_size, 4, 4]


model = PoseEstimationModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
def train(model, dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, poses in dataloader:
            imgs, poses = imgs.to(device), poses.to(device)

            # 前向传播
            outputs = model(imgs)
            # import pdb 
            # pdb.set_trace()

            # 计算损失
            loss = criterion(outputs, poses)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 开始训练
train(model, dataloader, criterion, optimizer, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu')

