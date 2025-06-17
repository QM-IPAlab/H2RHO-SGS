import os
import subprocess

def run_colmap(rgb_image_path, depth_image_path, output_path):
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # COLMAP 数据库路径
    database_path = os.path.join(output_path, "database.db")

    # 特征提取命令
    feature_extractor_cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", rgb_image_path,
        "--ImageReader.single_camera", "1"
    ]

    # 执行特征提取
    subprocess.run(feature_extractor_cmd)

    # 匹配特征命令
    matcher_cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", database_path
    ]

    # 执行特征匹配
    subprocess.run(matcher_cmd)

    # 创建稀疏模型的输出路径
    sparse_model_path = os.path.join(output_path, "sparse")
    
    # 运行映射器命令
    mapper_cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", rgb_image_path,
        "--output_path", sparse_model_path
    ]

    # 执行映射器
    subprocess.run(mapper_cmd)

    print("COLMAP processing completed. Output saved to:", sparse_model_path)

# 示例调用
run_colmap("path/to/rgb/images", "path/to/depth/images", "output/path")