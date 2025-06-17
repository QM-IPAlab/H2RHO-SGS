from __future__ import print_function
# from turtle import color

import numpy as np
import argparse
import grasp_estimator_yik
import sys
import os
import glob
import mayavi.mlab as mlab
from utils.visualization_utils import *
from utils import utils
from data import DataLoader
import trimesh
from plyfile import PlyData

def getSafeGrasps(grasps, scores, pc_h):
    filtered_grasps = []
    filtered_scores = []
    y_min, y_max = -0.01, 0.01
    x_min, x_max = -0.035, 0.035
    z_min, z_max = -0.15, 0.20
    bbox_coords = np.array([[x_min, y_min, z_min],
                            [x_max, y_min, z_min],
                            [x_max, y_max, z_min],
                            [x_min, y_max, z_min],
                            [x_min, y_min, z_max],
                            [x_max, y_min, z_max],
                            [x_max, y_max, z_max],
                            [x_min, y_max, z_max]])
    # bbox_coords = np.concatenate([bbox_coords, np.ones((bbox_coords.shape[0], 1))], axis=1)
    pc_h = np.concatenate([pc_h, np.ones((pc_h.shape[0], 1))], axis=1)
    for i in range(grasps.shape[0]):
        grasp = grasps[i]
        score = scores[i]

        pc_h_grasp = np.matmul(np.linalg.inv(grasp), pc_h.T).T[:,:3]
        # Check if pointcloud of hand pc_h is in bounding box of the grasp bbox_grasp
        in_collision = False
        for j in range(pc_h_grasp.shape[0]):
            if pc_h_grasp[j][0] > x_min and pc_h_grasp[j][0] < x_max and pc_h_grasp[j][1] > y_min and pc_h_grasp[j][1] < y_max and pc_h_grasp[j][2] > z_min and pc_h_grasp[j][2] < z_max:
                in_collision = True
                break
        if not in_collision:
            filtered_grasps.append(grasp)
            filtered_scores.append(score)
    filtered_grasps = np.array(filtered_grasps)
    filtered_scores = np.array(filtered_scores)

    # print("Removed {} grasps out of {}".format(grasps.shape[0] - filtered_grasps.shape[0], grasps.shape[0]))
    return filtered_grasps, filtered_scores

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

    parser.add_argument('--safe_grasp_folder', type=str, default='/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/70_frame/')
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
    estimator = grasp_estimator_yik.GraspEstimator(grasp_sampler_args,
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
            # print('close the window to continue to next object . . .')
            mlab.show()
    else:
        # for folder in os.listdir(args.safe_grasp_folder): 
        #---safe grasp--------------------------------------:   
        ply_file = os.path.join(args.safe_grasp_folder,'handover_3D', 'handover.ply') 
        plydata = PlyData.read(ply_file)
        pc = plydata['vertex'].data
        points = pc[['x', 'y', 'z']]
        points_array = np.array(points.tolist())
        pc = points_array.astype(np.float32)
        pc_colors = np.zeros((pc.shape[0], 3), dtype=np.uint8)
        pc_colors[:, :] = [0, 0, 255]

        plydata_hand = os.path.join(args.safe_grasp_folder,'handover_3D', 'hand.ply')
        plydata_hand = PlyData.read(plydata_hand)
        pc_h = plydata_hand['vertex'].data
        points_h = pc_h[['x', 'y', 'z']]
        points_array_h = np.array(points_h.tolist())
        pc_h = points_array_h.astype(np.float32)
        pc_h_colors = np.zeros((pc_h.shape[0], 3), dtype=np.uint8)
        pc_h_colors[:, :] = [0, 0, 255]

        plydata_o = os.path.join(args.safe_grasp_folder,'handover_3D', 'object.ply')
        plydata_o = PlyData.read(plydata_o)
        pc_o = plydata_o['vertex'].data
        points_o = pc_o[['x', 'y', 'z']]
        points_array_o = np.array(points_o.tolist())
        pc_o = points_array_o.astype(np.float32)
        pc_o_colors = np.zeros((pc_o.shape[0], 3), dtype=np.uint8)
        pc_o_colors[:, :] = [0, 0, 255]

        ply_whole = os.path.join(args.safe_grasp_folder, 'points3D.ply')
        plydata_whole = PlyData.read(ply_whole)
        pc_whole_data = plydata_whole['vertex'].data
        pc_whole_colors = np.array(pc_whole_data[['red', 'green', 'blue']])
        pc_whole_colors = np.array(pc_whole_colors.tolist())
        pc_whole_colors = pc_whole_colors.astype(np.uint8)
        points_whole = pc_whole_data[['x', 'y', 'z']]
        points_array_whole = np.array(points_whole.tolist())
        pc_whole = points_array_whole.astype(np.float32)
    
        
        mean = np.mean(pc_o, axis=0)
        pc_o = pc_o - mean
        pc_h = pc_h -mean
        pc = pc 
        pc_whole= pc_whole 

        generated_grasps, generated_scores = estimator.generate_and_refine_grasps_safe(
                pc_o, pc_h) 
        
        generated_grasps = np.array(generated_grasps)
        generated_scores = np.array(generated_scores)

        # print("Generated grasps: ", generated_grasps.shape)
        
        if generated_grasps.shape[0] == 0:
            print("No grasps generated")
            
        
        # Check if grasp is in collision with hand
        
        filtered_grasps, filtered_scores = getSafeGrasps(generated_grasps, generated_scores, pc_h)
        
        for i in range(filtered_grasps.shape[0]):
            filtered_grasps[i][0:3, 3] += mean

        sorted_scores = np.sort(filtered_scores)[::-1]  # Sort in descending order
        # Step 2: Check if there are at least three unique scores
        # if len(sorted_scores) < 3:
        #     print("Not enough unique scores to find the third largest.")
        # else:
        #     third_largest_score = sorted_scores[2]  # Get the third largest score
        # third_largese_index = np.where(filtered_scores == third_largest_score)[0][0]
        # third_largese_grasp = filtered_grasps[third_largese_index]
        # third_largese_grasp_w = third_largese_grasp.copy()
        
        max_score_index = np.argmax(filtered_scores)
        max_score = filtered_scores[max_score_index]
        best_grasp = filtered_grasps[max_score_index]
        # print("pointcloud mean:\n", mean )
        # print("best grasp in world coordinate:\n", best_grasp)
        print("best score:", max_score)
        np.save(f'{os.path.join(args.safe_grasp_folder)}/gpw.npy', best_grasp)
    
        
        # mlab.figure(bgcolor=(1, 1, 1))
        
        # mlab.axes(color=(0, 0, 0),x_axis_visibility=True, y_axis_visibility=True, z_axis_visibility=True, xlabel='x', ylabel='y', zlabel='z')
        # draw_scene(
        #     pc,
        #     pc_color=pc_colors,
        #     grasps=filtered_grasps,
        #     grasp_scores=filtered_scores,
        # )
        # mlab.savefig('grasp.png')
        
        # print('close the window to continue to next object . . .')
        # mlab.show()


if __name__ == '__main__':
    main(sys.argv[1:])
