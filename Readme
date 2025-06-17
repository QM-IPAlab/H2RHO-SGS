# Learning Human-to-Robot Handovers through 3D Scene Reconstruction

This is the official code release for the paper *"Learning Human-to-Robot Handovers through 3D Scene Reconstruction."*

---

## Pipeline Overview

### Step 1: Prepare the DexYCB Dataset for FSGS Input

**Download the data:**

```bash
python -m gdown 14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA
```

**Environment setup:**

```bash
module load Miniconda3/4.12.0  
module load CUDA/11.7.0
module load GCC/9.3.0
module load GCCcore/9.3.0
module load Python/3.8.2

export DEX_YCB_DIR='/home/e/eez095/dexycb_data'
cd project/dex-ycb-toolkit/
source dexycb/bin/activate
```

**Generate dataset:**
Edit `dex_ycb.py` if you don't have the full dataset:

```python
_SUBJECTS = [
  '20200813-subject-02',
]
```

Then run:

```bash
python examples/create_dataset.py
```

Choose camera views (6/7/8), delete the corresponding series from the `meta` folder in the dataset. Primary camera cannot be removed.

Convert to COLMAP format:

```bash
python examples/get_pointcloud.py --name 20200813-subject-02/20200813_145341
```

### Step 2: Segment Hand and Object

```bash
python examples/visualize_pose.py --src /home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/0_frame
```

This produces segmentation results under the COLMAP directory (e.g., `handover_3D`).

### Step 1+2 Combined

```bash
module purge
module load Miniconda3/4.12.0
module load CUDA/11.7.0
module load GCC/9.3.0
module load GCCcore/9.3.0
module load Python/3.8.2

cd project/dex-ycb-toolkit/
source dexycb/bin/activate
```

Edit `dex_ycb.py` if needed and then:

```bash
python run_1_2.py --dataSet 20200813-subject-02/20200813_145341 --dataSetBig /home/e/eez095/dexycb_data/20200813-subject-02
```

**Multi-threaded version (only works at subject level):**

```bash
python run_1_2_v3.py --dataSetBig /home/e/eez095/dexycb_data/20200813-subject-02 --completed_file xxx --error_file xxx
```

---

### Step 3: 6-DOF Grasp Estimation

**On sulis:**

install:
```bash
module purge
module load CUDA/11.3.1
module load GCCcore/9.3.0
module load Python/3.8.2
cd project/pytorch_6dof-graspnet
virtualenv pytorch_6dof_grspnet
source pytorch_6dof_grspnet/bin/activate

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

cd ../Pointnet2_PyTorch
export TORCH_CUDA_ARCH_LIST="7.0 8.0 9.0"
pip3 install -r requirements.txt

cd ../pytorch_6dof-graspnet/
pip install numpy==1.21.1
pip3 install -r requirements.txt

pip install configobj plyfile
```

run the code:
```bash
module purge
module load CUDA/11.3.1
module load GCCcore/9.3.0
module load Python/3.8.2
cd project/pytorch_6dof-graspnet
source pytorch_6dof_grspnet/bin/activate

python -m demo.main --safe_grasp_folder /path/to/70_frame
```

Parameters:

* Modify `visualiztion_utils.py`: `min_grasps = 100`
* Adjust settings in `main.py` lines 54-111
* Results saved to `gpw.npy`

---

### Step 4: Trajectory Sampling

**On sulis:**

```bash
module purge
module load CUDA/11.3.1
module load GCCcore/9.3.0
module load Python/3.8.2
cd project/pytorch_6dof-graspnet
source pytorch_6dof_grspnet/bin/activate
python -m demo.sample_v4 --base_dir /path/to/70_frame
```

Results are saved in `trajectory/` folders, each representing a trajectory with poses like `i_0.npy`, `i_target.npy`, etc.

**On PC:**

```bash
conda activate /home/robot_tutorial/anaconda3/6dofgraspnet_pt
python -m demo.sample --base_dir /path/to/54_frame_handover_3D
```

Transfer results back to sulis to continue processing.

### Step 3+4 Combined

**Single-frame or directory-level processing:**

```bash
python -m demo.run_3_4 --grasp_one_frame /path/to/frame --dataset_path /path/to/scene --BigDataset_path /path/to/subject
```

**Multi-threaded (subject level):**

```bash
python -m demo.run_3_4_v2 --BigDataset_path /path/to/subject --completed_file ./completed_datasets.txt --error_file ./error.txt --max_workers 8
```

**Multi-threaded (scene level):**

```bash
python -m demo.run_3_4_v1 --dataset_path /path/to/scene --max_threads 4
```

**Selective frame processing:**

```bash
python -m demo.run_3_4_v3 --BigDataset_path /path/to/subject --completed_file ./log/completed_datasets.txt --error_file ./log/error.txt --max_workers 8
```

Check for valid trajectories:

```bash
python demo/check_complete_file.py
```

---

### Step 5: FSGS Reconstruction and Rendering

**Environment setup:**

```bash
module purge
module load CUDA/11.3.1
module load GCCcore/9.3.0
module load Python/3.8.2
cd project/FSGS
source FSGS/bin/activate
```

**Train and render:**

```bash
python train.py --source_path /path/to/70_frame --model_path /path/to/FSGS_output/ --iteration 10000 --kk
python render_v4.py --source_path /path/to/70_frame --model_path /path/to/FSGS_output --iteration 10000 --kk --video
```

**Batch mode:**

```bash
python run_5_v2.py --BigDataset_path /path/to/subject --step3_complete /path/to/non_empty_paths.txt
```

---

### Step 6: Supervised Learning

**Environment:**

```bash
module purge
module load CUDA/11.3.1
module load GCCcore/11.2.0 Python/3.9.6
cd project/policy_learning/
source policy/bin/activate
```

**Training:**

```bash
python ./script/policy_v15.py --mode train --model_path ./model/v15.pth --train_txt_path ./dataset_file/data_0310_no_obj23_train.txt --text_txt_path ./dataset_file/data_0310_no_obj23_test.txt
```

**Install:**

```bash
virtualenv policy
source policy/bin/activate
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install open3d
```

# 💗 Batched Pipeline for Processing and Training Larger Models

This repository contains a multi-step pipeline for processing the DexYCB dataset, rendering scenes, filtering data, and training policies on selected samples.

---

## 📁 Directory Overview

- Project root: `/home/e/eez095/project`
- DexYCB data: `/home/e/eez095/dexycb_data/20200813-subject-02`
- Final dataset list: `/home/e/eez095/project/policy_learning/dataset.txt`
- Evaluation log: `/Users/yuekun/study/experiment/dexycb_category.xsl`

---

## ✅ Step 1-2: Pose Visualization and Initial Logging

Run the modified `examples/visualize_pose.py` script to process all the dataset files.

### Output Logs:
- Completed samples:  
  `/home/e/eez095/project/dex-ycb-toolkit/log/completed_datasets.txt`
- Errors:  
  `/home/e/eez095/project/dex-ycb-toolkit/log/error_log.txt`

---

## 🔧 Step 3-4: Sample & Render with 6-DoF Model

Use `sample_v4` and `render_v4` with the 6-DoF grasp model.

### Command:
```bash
python -m demo.run_3_4_v3 \
  --BigDataset_path /home/e/eez095/dexycb_data/20200813-subject-02 \
  --completed_file ./log/completed_datasets.txt \
  --error_file ./log/error.txt \
  --max_workers 8 \
  --step1_complete /home/e/eez095/project/dex-ycb-toolkit/log/completed_datasets.txt
```
Logs:
Output logs:
/home/e/eez095/project/pytorch_6dof-graspnet/log/
Post-check:
Edit paths manually in demo/check_complete_file.py, then run:
python demo/check_complete_file.py
This script will create:
/home/e/eez095/project/pytorch_6dof-graspnet/log/non_empty_paths.txt
Use this .txt file as input for FSGS (next step).

## 🧠 Step 5: Run FSGS Optimization

Environment Setup:
```bash
module purge
module load CUDA/11.3.1
module load GCCcore/9.3.0
module load Python/3.8.2
cd project/FSGS
source FSGS/bin/activate
```
Command:
```bash
python run_5_v2.py \
  --step3_complete /home/e/eez095/project/pytorch_6dof-graspnet/log/completed_datasets.txt \
  --completed_file ./log/completed_datasets.txt
```

## 📝 Step 6: Manual Filtering and Logging

### Input:
Use the output file:
/home/e/eez095/project/FSGS/log/completed_datasets.txt
### Sort:
```bash
python sort_log.py \
  --input_file /home/e/eez095/project/FSGS/log/completed_datasets.txt \
  --output_file /home/e/eez095/project/FSGS/log/completed_datasets_log_v4.txt
```
Manual Curation:
Use Excel sheet:
/Users/yuekun/study/experiment/dexycb_category.xsl
Table 2: For recording object & image quality
Table 3: Exclusions
Final Output:
Write selected high-quality data into:
/home/e/eez095/project/policy_learning/dataset.txt

## 🎯 Step 7: Train Policy Model

Use the filtered dataset.txt to train the policy.
Command:
```bash
python policy_v12_T.py \
  --mode train \
  --model_path <path_to_save_model> \
  --txt_path /home/e/eez095/project/policy_learning/dataset.txt
```
🗺️ Notes

All steps depend on accurate logs; ensure paths are consistent and error logs are reviewed.
You may adjust --max_workers in step 3-4 to match your system’s performance.

