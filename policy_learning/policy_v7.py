#inference:输入图像  输出应该移动的欧拉角+位姿
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
                image_dir = os.path.join(data_path, "FSGS_output", "video", "ours_10000")
                trajectory_dir = os.path.join(data_path, "trajectory")
                subfolders = sorted([f for f in os.listdir(image_dir) if not f.endswith(".mp4")])

                for subfolder in subfolders:
                    img_folder = os.path.join(image_dir, subfolder)
                    traj_folder = os.path.join(trajectory_dir, subfolder)
                    img_files = sorted([f for f in os.listdir(img_folder) if not f.endswith(".mp4")], key=custom_sort)
                    traj_files = sorted([f for f in os.listdir(traj_folder) if not f.endswith(".mp4")], key=custom_sort)

                    for i in range(len(img_files) - 1):
                        if img_files[i].endswith("_target.png"):
                            continue
                        next_traj_path = os.path.join(traj_folder, traj_files[i + 1])
                        if img_files[i + 1].endswith("_target.png"):
                            next_traj_path = os.path.join(traj_folder, traj_files[i])

                        self.data.append({
                            "img_path": os.path.join(img_folder, img_files[i]),
                            "current_traj": os.path.join(traj_folder, traj_files[i]),
                            "next_traj": next_traj_path,
                        })

        if self.save_txt_path:
            save_to_txt(self.data, self.save_txt_path)

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        data_item = self.data[idx]
        img = Image.open(data_item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)
        pose = process_pose(T_AB)
        return img, torch.tensor(pose, dtype=torch.float32)

# Dataset class for training and testing
class PoseDataset(Dataset):
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
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)
        pose = process_pose(T_AB)
        return img, torch.tensor(pose, dtype=torch.float32)

# Model definition: Pose Estimation using ResNet18
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 6)  # 3个欧拉角 + 3个位移

    def forward(self, x):
        return self.backbone(x)


# Training and evaluation function with mixed precision
def train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    best_test_loss = float('inf')
    scaler = GradScaler()

    # Lists to store loss values for plotting
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, poses in train_dataloader:
            imgs, poses = imgs.to(device), poses.to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, poses)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)  # Store train loss for plotting
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        test_loss = evaluate(model, test_dataloader, criterion, device)
        test_losses.append(test_loss)  # Store test loss for plotting

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_pose_estimation_model.pth")
            print("Saved best model!")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='x')
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_convergence.png')  # Save the figure
    plt.show()

# Evaluate function
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

# Function to test the model with a single image
def test_pose(image_path, model, transform, device='cuda'):
    # Load the image and apply transformations
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device (GPU or CPU)
    img = img.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for inference
    with torch.no_grad():
        # Get the model's prediction
        predicted_pose = model(img)

    # Return the predicted pose (4x4 transformation matrix)
    return predicted_pose.squeeze().cpu().numpy()

# Function to load the trained model
def load_model(model_path, device='cuda'):
    model = PoseEstimationModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model


# Evaluate function
def evaluate_only(model, txt_path, transform, criterion, device='cuda'):
    # Load the dataset
    dataset = CustomDataset(txt_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model.eval()  # Set the model to evaluation mode

    running_loss = 0.0
    gt_poses = []  # Ground truth poses
    predicted_poses = []  # Predicted poses

    # Disable gradient calculation for inference
    with torch.no_grad():
        for imgs, poses in dataloader:
            imgs, poses = imgs.to(device), poses.to(device)
            
            # Get the model's predictions
            predicted_pose = model(imgs)

            # Calculate the loss (MSE between predicted pose and ground truth pose)
            loss = criterion(predicted_pose, poses)
            running_loss += loss.item()

            # Append ground truth and predicted poses for each sample
            gt_poses.append(poses.cpu().numpy())
            predicted_poses.append(predicted_pose.cpu().numpy())

    # Calculate the average loss
    avg_loss = running_loss / len(dataloader)
    print(f"Average Test Loss: {avg_loss:.4f}")

    # Flatten the lists for easier comparison between GT and predicted poses
    gt_poses = np.concatenate(gt_poses, axis=0)
    predicted_poses = np.concatenate(predicted_poses, axis=0)

    
    

    return avg_loss, gt_poses, predicted_poses


# Main function to load data, create dataloaders, and start training
def main():
    txt_path = "/home/e/eez095/project/policy_learning/dataset.txt"
    save_txt_path = '/home/e/eez095/project/policy_learning/getitem_data.txt'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomDataset(txt_path, transform=transform, save_txt_path=save_txt_path)
    train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)

    train_dataset = PoseDataset(train_data, transform=transform)
    test_dataset = PoseDataset(test_data, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = PoseEstimationModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)  # Enable multi-GPU support

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device=device)

    #test 
    model = load_model("./best_pose_estimation_model.pth", device='cuda')
    image_path = "/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/59_frame/FSGS_output/video/ours_10000/12/12_4.png"
    predicted_pose = test_pose(image_path, model, transform, device='cuda')
    print(predicted_pose)

    #evaluate
    txt_path = "/home/e/eez095/project/policy_learning/dataset_evaluate.txt"
    criterion = nn.MSELoss()  # 使用 MSE 损失函数
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    avg_loss, gt_poses, predicted_poses = evaluate_only(model, txt_path, transform, criterion, device='cuda')
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Pose differences: {gt_poses[:1], predicted_poses[:1]}")  # 打印前10个样本的差异


if __name__ == "__main__":
    main()
