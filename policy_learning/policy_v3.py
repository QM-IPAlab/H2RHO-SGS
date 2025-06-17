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

# Custom sort function to order filenames in a desired way
def custom_sort(filename):
    # If the filename contains '_target', return a large number to place it at the end
    if '_target' in filename:
        return (float('inf'),)

    # Otherwise, extract numeric parts and sort based on them
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
        """
        Args:
            txt_path (str): Path to the txt file containing data directories.
            transform (callable, optional): Image transformations to apply.
            save_txt_path (str, optional): Path to save the processed data as a txt file.
        """
        self.transform = transform
        self.save_txt_path = save_txt_path

        # Parse the txt file to gather data paths
        self.data = []
        with open(txt_path, 'r') as file:
            for line in file:
                data_path = line.strip()

                # Define paths for images and trajectories
                image_dir = os.path.join(data_path, "FSGS_output", "video", "ours_10000")
                trajectory_dir = os.path.join(data_path, "trajectory")

                # Get all subfolders (excluding .mp4 files)
                subfolders = sorted([f for f in os.listdir(image_dir) if not f.endswith(".mp4")])

                for subfolder in subfolders:
                    img_folder = os.path.join(image_dir, subfolder)
                    traj_folder = os.path.join(trajectory_dir, subfolder)

                    # Get all image and trajectory files (excluding .mp4 files)
                    img_files = sorted([f for f in os.listdir(img_folder) if not f.endswith(".mp4")], key=custom_sort)
                    traj_files = sorted([f for f in os.listdir(traj_folder) if not f.endswith(".mp4")], key=custom_sort)

                    # Pair images and their corresponding trajectory changes
                    for i in range(len(img_files) - 1):
                        # Skip target and .mp4 files
                        if img_files[i].endswith("_target.png"):
                            continue

                        # Handle the case where the next trajectory is the same as the current one
                        next_traj_path = os.path.join(traj_folder, traj_files[i + 1])
                        if img_files[i + 1].endswith("_target.png"):
                            next_traj_path = os.path.join(traj_folder, traj_files[i])

                        # Append the paired data to the list
                        self.data.append({
                            "img_path": os.path.join(img_folder, img_files[i]),
                            "current_traj": os.path.join(traj_folder, traj_files[i]),
                            "next_traj": next_traj_path,
                        })

        # Optionally save data to txt
        if self.save_txt_path:
            save_to_txt(self.data, self.save_txt_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]

        # Load image
        img = Image.open(data_item["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Load trajectories and compute pose change
        current_traj = np.load(data_item["current_traj"])
        next_traj = np.load(data_item["next_traj"])

        # Compute the inverse of current_traj
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)

        return img, torch.tensor(T_AB, dtype=torch.float32)

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

        # Compute the inverse of current_traj and pose transformation
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)

        return img, torch.tensor(T_AB, dtype=torch.float32)

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

# Training and evaluation function
def train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    best_test_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, poses in train_dataloader:
            imgs, poses = imgs.to(device), poses.to(device)

            # Forward pass
            outputs = model(imgs)

            # Compute loss
            loss = criterion(outputs, poses)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(train_dataloader):.4f}")

        # Evaluate after each epoch
        test_loss = evaluate(model, test_dataloader, criterion, device)

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_pose_estimation_model.pth")
            print("Saved best model!")

# Model definition: Pose Estimation using ResNet18
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 16)

    def forward(self, x):
        x = self.backbone(x)
        return x.view(-1, 4, 4)  # Reshape to [batch_size, 4, 4]

# Main function to load data, create dataloaders, and start training
def main():
    # Dataset paths and transformations
    txt_path = "/home/e/eez095/project/policy_learning/dataset.txt"
    save_txt_path = '/home/e/eez095/project/policy_learning/getitem_data.txt'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and split into training and testing sets
    dataset = CustomDataset(txt_path, transform=transform, save_txt_path=save_txt_path)
    train_data, test_data = train_test_split(dataset.data, test_size=0.2, random_state=42)

    # DataLoader setup
    train_dataset = PoseDataset(train_data, transform=transform)
    test_dataset = PoseDataset(test_data, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = PoseEstimationModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)  # Enable multi-GPU support
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device=device)


# Entry point of the program
if __name__ == "__main__":
    main()
