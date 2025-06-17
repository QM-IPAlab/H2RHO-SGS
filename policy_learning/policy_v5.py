#使用四元数和位移作为模型的输入与输出
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
# Loss function
def quaternion_loss(pred_quat, gt_quat):
    dot_product = np.dot(pred_quat, gt_quat)
    return 1 - dot_product**2  # Euclidean distance between quaternions

def loss_fn(predicted_pose, ground_truth_pose):
    predicted_quat = predicted_pose[:4]
    predicted_translation = predicted_pose[4:]
    
    gt_quat = ground_truth_pose[:4]
    gt_translation = ground_truth_pose[4:]
    
    # Quaternion loss
    quat_loss = quaternion_loss(predicted_quat, gt_quat)
    
    # Translation loss (MSE)
    translation_loss = nn.MSELoss()(predicted_translation, gt_translation)
    
    return quat_loss + translation_loss


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
        
        # Decompose the transformation matrix into rotation (quaternion) and translation (vector)
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)
        
        # Convert 4x4 transformation matrix to quaternion and translation vector
        rotation_matrix = T_AB[:3, :3]
        translation_vector = T_AB[:3, 3]
        
        # Convert the rotation matrix to quaternion
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        # Concatenate quaternion and translation vector as the pose output
        pose = np.concatenate([quat, translation_vector], axis=0)
        import pdb 
        pdb.set_trace()
        return img, torch.tensor(pose, dtype=torch.float32)

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                t = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / t
                x = 0.25 * t
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / t
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / t
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                t = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / t
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / t
                y = 0.25 * t
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / t
            else:
                t = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / t
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / t
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / t
                z = 0.25 * t

        return np.array([w, x, y, z])



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
        
        # Decompose the transformation matrix into rotation (quaternion) and translation (vector)
        T_A_inv = np.linalg.inv(current_traj)
        T_AB = np.dot(T_A_inv, next_traj)
        
        # Convert 4x4 transformation matrix to quaternion and translation vector
        rotation_matrix = T_AB[:3, :3]
        translation_vector = T_AB[:3, 3]
        
        # Convert the rotation matrix to quaternion
        quat = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        # Concatenate quaternion and translation vector as the pose output
        pose = np.concatenate([quat, translation_vector], axis=0)
    
        return img, torch.tensor(pose, dtype=torch.float32)
    
    def rotation_matrix_to_quaternion(self, rotation_matrix):
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                t = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / t
                x = 0.25 * t
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / t
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / t
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                t = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / t
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / t
                y = 0.25 * t
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / t
            else:
                t = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / t
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / t
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / t
                z = 0.25 * t

        return np.array([w, x, y, z])
    
    

# Model definition: Pose Estimation using ResNet18
class PoseEstimationModel(nn.Module):
    def __init__(self):
        super(PoseEstimationModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 7)  # 4 for quaternion and 3 for translation

    def forward(self, x):
        x = self.backbone(x)
        return x  # Output a vector of length 7 (4 quaternion components + 3 translation components)


def train_and_evaluate(model, train_dataloader, test_dataloader, criterion, optimizer, epochs=10, device='cuda'):
    model.to(device)
    best_test_loss = float('inf')
    scaler = GradScaler()

    # Prepare the dataset for evaluation (instead of passing DataLoader)
    test_dataset = CustomDataset(txt_path, transform=transform)

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

        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {running_loss / len(train_dataloader):.4f}")
        test_loss = evaluate(model, test_dataset, transform, criterion, device)  # Pass the test dataset directly

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), "best_pose_estimation_model.pth")
            print("Saved best model!")


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


def evaluate(model, dataset, transform, criterion, device='cuda'):
    # Use the dataset object directly to create a DataLoader
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
    avg_loss, gt_poses, predicted_poses = evaluate(model, txt_path, transform, criterion, device='cuda')
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Pose differences: {gt_poses[:1], predicted_poses[:1]}")  # 打印前10个样本的差异







if __name__ == "__main__":
    main()
