from __future__ import print_function

import numpy as np
import argparse
import grasp_estimator
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
import mayavi.mlab as mlab
from utils import utils
from data import DataLoader
import trimesh
from plyfile import PlyData


def make_parser():
    parser = argparse.ArgumentParser(
        description='6-DoF GraspNet Demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grasp_sampler_folder',
                        type=str,
                        default='checkpoints/gan_pretrained/')
    parser.add_argument('--grasp_evaluator_folder',
                        type=str,
                        default='checkpoints/evaluator_pretrained/')
    parser.add_argument('--refinement_method',
                        choices={"gradient", "sampling"},
                        default='sampling')
    parser.add_argument('--refine_steps', type=int, default=25)

    parser.add_argument('--npy_folder', type=str, default='demo/data/')
    parser.add_argument('--ply_folder', type=str, default='demo/data_ply/')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.8,
        help=
        "When choose_fn is something else than all, all grasps with a score given by the evaluator notwork less than the threshold are removed"
    )
    parser.add_argument(
        '--choose_fn',
        choices={
            "all", "better_than_threshold", "better_than_threshold_in_sequence"
        },
        default='better_than_threshold',
        help=
        "If all, no grasps are removed. If better than threshold, only the last refined grasps are considered while better_than_threshold_in_sequence consideres all refined grasps"
    )

    parser.add_argument('--target_pc_size', type=int, default=1024) #指定目标点云的大小
    parser.add_argument('--num_grasp_samples', type=int, default=200)
    parser.add_argument(
        '--generate_dense_grasps',
        action='store_true',
        help=
        "If enabled, it will create a [num_grasp_samples x num_grasp_samples] dense grid of latent space values and generate grasps from these."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=30,
        help=
        "Set the batch size of the number of grasps we want to process and can fit into the GPU memory at each forward pass. The batch_size can be increased for a GPU with more memory."
    )
    parser.add_argument('--train_data', action='store_true')
    parser.add_argument('--scale_factor', type=float, default= 1)
    
    opts, _ = parser.parse_known_args()
    if opts.train_data:
        parser.add_argument('--dataset_root_folder',
                            required=True,
                            type=str,
                            help='path to root directory of the dataset.')
    return parser


def get_color_for_pc(pc, K, color_image):
    proj = pc.dot(K.T)
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]

    pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
    for i, p in enumerate(proj):
        x = int(p[0])
        y = int(p[1])
        pc_colors[i, :] = color_image[y, x, :]

    return pc_colors


def backproject(depth_cv,
                intrinsic_matrix,
                return_finite_depth=True,
                return_selection=False):

    depth = depth_cv.astype(np.float32, copy=True)

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection

    return X


def main(args):
    parser = make_parser()
    args = parser.parse_args()
    grasp_sampler_args = utils.read_checkpoint_args(args.grasp_sampler_folder)
    grasp_sampler_args.is_train = False
    grasp_evaluator_args = utils.read_checkpoint_args(
        args.grasp_evaluator_folder)
    grasp_evaluator_args.continue_train = True
    estimator = grasp_estimator.GraspEstimator(grasp_sampler_args,
                                               grasp_evaluator_args, args)
    if args.train_data:
        grasp_sampler_args.dataset_root_folder = args.dataset_root_folder
        grasp_sampler_args.num_grasps_per_object = 1
        grasp_sampler_args.num_objects_per_batch = 1
        dataset = DataLoader(grasp_sampler_args)
        for i, data in enumerate(dataset):
            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                data["pc"].squeeze())
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(data["pc"][0],
                       grasps=generated_grasps,
                       grasp_scores=generated_scores)
            print('close the window to continue to next object . . .')
            mlab.show()
    else:
        for ply_file in glob.glob(os.path.join(args.ply_folder, '*.ply')):   

            plydata = PlyData.read(ply_file)
            pc = plydata['vertex'].data
            points = pc[['x', 'y', 'z']]
            points_array = np.array(points.tolist())
            pc = points_array.astype(np.float32)
            pc = points_array - np.mean(points_array, axis=0)
            pc = pc * args.scale_factor

            pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
            pc_colors[:, :] = [0, 0, 255]
            print("pc shape: ", pc.shape)
            print("pc first 10 points: ", pc[:10])

            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                pc)
            print(f'generated_grasps:{generated_grasps} \n generated_scores:{generated_scores}\n')
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(
                pc,
                pc_color=pc_colors,
                grasps=generated_grasps,
                grasp_scores=generated_scores,
            )
            print('close the window to continue to next object . . .')
            mlab.show()

           

            '''
            # example of Yik's object file
            test_obj = '/home/robot_tutorial/vgn_ws/src/autosdf_j2/demo/0000.obj'
            mesh = trimesh.load_mesh(test_obj)
            # get the point cloud of the mesh
            mesh_pc = mesh.sample(2048)
            print("mesh_pc shape: ", mesh_pc.shape)
            mesh_pc = mesh_pc[:, :3]
            mesh_pc = np.array(mesh_pc)
            mesh_pc = mesh_pc.astype(np.float32)
            mesh_pc = mesh_pc - np.mean(mesh_pc, axis=0)
            mesh_pc_colors = np.zeros((mesh_pc.shape[0], 3), dtype=np.uint8)
            mesh_pc_colors[:, :] = [255, 0, 0]
            print("mesh_pc shape: ", mesh_pc.shape)
            print("mesh_pc first 10 points: ", mesh_pc[:10])
            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                mesh_pc)
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(
                mesh_pc,
                pc_color=mesh_pc_colors,
                grasps=generated_grasps,
                grasp_scores=generated_scores,
            )
            print('close the window to continue to next object . . .')
            mlab.show()
            '''
            
        '''
        for npy_file in glob.glob(os.path.join(args.npy_folder, '*.npy')): # original code using npy file
        # Depending on your numpy version you may need to change allow_pickle
        # from True to False.

            data = np.load(npy_file, allow_pickle=True,
                            encoding="latin1").item()

            depth = data['depth']
            image = data['image']
            K = data['intrinsics_matrix']
            # Removing points that are farther than 1 meter or missing depth
            # values.
            #depth[depth == 0 or depth > 1] = np.nan

            np.nan_to_num(depth, copy=False)
            mask = np.where(np.logical_or(depth == 0, depth > 1))
            depth[mask] = np.nan
            pc, selection = backproject(depth,
                                        K,
                                        return_finite_depth=True,
                                        return_selection=True)
            pc_colors = image.copy()
            pc_colors = np.reshape(pc_colors, [-1, 3])
            pc_colors = pc_colors[selection, :]

            # Smoothed pc comes from averaging the depth for 10 frames and removing
            # the pixels with jittery depth between those 10 frames.
            object_pc = data['smoothed_object_pc']
            
            generated_grasps, generated_scores = estimator.generate_and_refine_grasps(
                object_pc)
            mlab.figure(bgcolor=(1, 1, 1))
            draw_scene(
                pc,
                pc_color=pc_colors,
                grasps=generated_grasps,
                grasp_scores=generated_scores,
            )
            print('close the window to continue to next object . . .')
            mlab.show()
        '''

if __name__ == '__main__':
    main(sys.argv[1:])
