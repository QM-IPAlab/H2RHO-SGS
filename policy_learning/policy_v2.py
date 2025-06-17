import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import re

def custom_sort(filename):
    # 如果文件名包含 '_target'，则返回一个非常大的数字（确保它在最后）
    if '_target' in filename:
        return (float('inf'),)  # 将 '_target' 文件排到最后
    
    # 否则，提取数字部分并按其顺序排序
    match = re.findall(r'(\d+)', filename)
    # 返回数字的元组（保证按顺序排序）
    return tuple(int(num) for num in match)

# 定义CustomDataset类
class CustomDataset(Dataset):
    def __init__(self, txt_path, transform=None, save_txt_path=None):
        """
        Args:
            txt_path (str): Path to the txt file containing data directories.
            transform (callable, optional): Image transformations to apply.
        """
        self.transform = transform
        self.save_txt_path = save_txt_path

        # Parse txt file to gather data paths
        self.data = []
        with open(txt_path, 'r') as file:
            for line in file:
                data_path = line.strip()

                # Define paths for images and trajectories
                image_dir = os.path.join(data_path, "FSGS_output", "video", "ours_10000")
                trajectory_dir = os.path.join(data_path, "trajectory")

                # List all subfolders
                # subfolders = sorted(os.listdir(image_dir))
                subfolders = sorted([f for f in os.listdir(image_dir) if not f.endswith(".mp4")])

                for subfolder in subfolders:
                    if subfolder.endswith(".mp4"):
                        continue  # Skip .mp4 files

                    img_folder = os.path.join(image_dir, subfolder)
                    traj_folder = os.path.join(trajectory_dir, subfolder)

                    # Get all image and trajectory files
                    img_files = sorted([f for f in os.listdir(img_folder) if not f.endswith(".mp4")], key=custom_sort)
                    traj_files = sorted([f for f in os.listdir(traj_folder) if not f.endswith(".mp4")], key=custom_sort)

                    # Pair images and their corresponding trajectory changes
                    for i in range(len(img_files) - 1):
                        if img_files[i].endswith("_target.png") or img_files[i].endswith(".mp4"):
                            continue  # Skip _target.png and .mp4 files
                        print(os.path.join(img_folder, img_files[i]))
                        print( os.path.join(traj_folder, traj_files[i]))
                        # print(img_files[i+1])
                        # print(traj_files[i+1])
                        # Handle the case where the next trajectory is the same as the current
                        next_traj_path = os.path.join(traj_folder, traj_files[i + 1])
                        if img_files[i + 1].endswith("_target.png"):
                            next_traj_path = os.path.join(traj_folder, traj_files[i])


                        self.data.append({
                            "img_path": os.path.join(img_folder, img_files[i]),
                            "current_traj": os.path.join(traj_folder, traj_files[i]),
                            "next_traj": next_traj_path,
                        })
        if self.save_txt_path:
            self._save_to_txt()

    def _save_to_txt(self):
        with open(self.save_txt_path, 'w') as f:
            for data_item in self.data:
                line = f"{data_item['img_path']} {data_item['current_traj']} {data_item['next_traj']}\n"
                f.write(line)
    
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


# 数据路径
txt_path = "/home/e/eez095/project/policy_learning/dataset.txt"

# 图像变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
save_txt_path = '/home/e/eez095/project/policy_learning/getitem_data'
dataset = CustomDataset(txt_path, transform=transform, save_txt_path = save_txt_path)

# 获取数据
all_data = dataset.data

# 拆分数据集
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# 创建训练集和测试集的Dataset类
class TrainDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        img = Image.open(data_item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])

        # 计算 T_A 的逆矩阵
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)

        return img, torch.tensor(T_AB, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        img = Image.open(data_item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])

        # 计算 T_A 的逆矩阵
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)

        return img, torch.tensor(T_AB, dtype=torch.float32)


# 创建训练集和测试集的DataLoader
train_dataset = TrainDataset(train_data, transform=transform)
test_dataset = TestDataset(test_data, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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


# 初始化模型
model = PoseEstimationModel()
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# 评估函数
def evaluate(model, dataloader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, poses in dataloader:
            imgs, poses = imgs.to(device), poses.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, poses)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss


# 训练和评估函数
def train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model = model.to(device)
    best_test_loss = float('inf')  # 用于保存最好的模型
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, poses in train_dataloader:
            imgs, poses = imgs.to(device), poses.to(device)

            # 前向传播
            outputs = model(imgs)

            # 计算损失
            loss = criterion(outputs, poses)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(train_dataloader):.4f}")

        # 在每个epoch后评估模型
        test_loss = evaluate(model, test_dataloader, criterion, device)

        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_pose_estimation_model.pth")
            print("Saved best model!")



# 开始训练和评估
train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu')
