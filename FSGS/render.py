# render image name is same with the .npz name

import shutil
import copy
import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import time
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path
from utils.general_utils import vis_depth


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering["render"], os.path.join(render_path, view.image_name + '.png'))
                                            #'{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if args.render_depth:
            depth_map = vis_depth(rendering['depth'][0].detach().cpu().numpy())
            np.save(os.path.join(render_path, view.image_name + '_depth.npy'), rendering['depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)

def render_video(source_path, model_path, iteration, views, gaussians, pipeline, background, fps=30):
    render_path = os.path.join(model_path, 'video', "ours_{}".format(iteration))
    if os.path.exists(render_path):
        shutil.rmtree(render_path)
    makedirs(render_path, exist_ok=True)
    view = copy.deepcopy(views[0]) 

    # 因为没有poses_bounds.npy 所以使用360
    # if source_path.find('llff') != -1:
    # render_poses = generate_spiral_path(np.load(source_path + '/poses_bounds.npy'))
    # elif source_path.find('360') != -1:
    render_poses_example = generate_ellipse_path(views)


    render_poses = []
    folder_path = os.path.join(source_path,'trajectory')

    size = (view.original_image.shape[2], view.original_image.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    final_video = cv2.VideoWriter(os.path.join(render_path, 'final_video.mp4'), fourcc, fps, size)

    # # 遍历指定目录下的所有文件
    print(f"folder_path:{folder_path} \n")
    for idi, item in enumerate(os.listdir(folder_path)):
        render_path_j = os.path.join(render_path,item)
        os.makedirs(render_path_j,exist_ok = True)
        item_path = os.path.join(folder_path,item)
        print(f'dealing route number:{item}')
        final_video_j = cv2.VideoWriter(os.path.join(render_path_j, 'video.mp4'), fourcc, fps, size)
        
        for file_name in sorted(os.listdir(item_path)):
            if file_name.endswith('.npy'):  # 只处理 .npy 文件
                file_path = os.path.join(item_path, file_name)
                data = np.load(file_path)
                pose_c2w = copy.deepcopy(data)
                R_s = pose_c2w[:3, :3]
                t_s = pose_c2w[:3, 3]
                R_s_inv = np.linalg.inv(R_s) #torch.inverse(R_s)
                t_s_inv = -R_s_inv @ t_s  #torch.mv(R_s_inv, -t_s)
            
                view.world_view_transform = torch.tensor(getWorld2View2(R_s, t_s_inv, view.trans, view.scale)).transpose(0, 1).cuda()
                view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
                view.camera_center = view.world_view_transform.inverse()[3, :3]


                # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

                rendering = render(view, gaussians, pipeline, background)

                img = torch.clamp(rendering["render"], min=0., max=1.)
                img_name =  os.path.splitext(file_name)[0] + '.png'
                torchvision.utils.save_image(img, os.path.join(render_path_j, img_name))
                video_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255.).astype(np.uint8)[..., ::-1]
                final_video_j.write(video_img)
                final_video.write(video_img)
        final_video_j.release()

    final_video.release()

def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):
    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.video:
            render_video(dataset.source_path, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(),
                         gaussians, pipeline, background, args.fps)

        if not args.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
        if not args.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--kk", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)
