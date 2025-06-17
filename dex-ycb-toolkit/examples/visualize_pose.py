# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]
# use master camera as label

"""Example of visualizing object and hand pose of one image sample."""
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import pyrender
import trimesh
import torch
import cv2
import matplotlib.pyplot as plt
import argparse
import yaml

from manopth.manolayer import ManoLayer


from dex_ycb_toolkit.factory import get_dataset


def create_scene(sample, obj_file):
  """Creates the pyrender scene of an image sample.

  Args:
    sample: A dictionary holding an image sample.
    obj_file: A dictionary holding the paths to YCB OBJ files.

  Returns:
    A pyrender scene object.
  """
  # Create pyrender scene.
  scene = pyrender.Scene(bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
                         ambient_light=np.array([1.0, 1.0, 1.0]))

  # Add camera.
  fx = sample['intrinsics']['fx']
  fy = sample['intrinsics']['fy']
  cx = sample['intrinsics']['ppx']
  cy = sample['intrinsics']['ppy']
  cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
  scene.add(cam, pose=np.eye(4))

  # Load poses.
  label = np.load(sample['label_file'])
  pose_y = label['pose_y']
  pose_m = label['pose_m']

  # Load YCB meshes.
  mesh_y = []
  for i in sample['ycb_ids']:
    mesh = trimesh.load(obj_file[i])
    mesh = pyrender.Mesh.from_trimesh(mesh)
    mesh_y.append(mesh)

  #grasp object

  ycb_class = sample['ycb_ids'][sample['ycb_grasp_ind']]

  # Add YCB meshes.
  # for o in range(len(pose_y)):
  o=sample['ycb_grasp_ind']
  if np.all(pose_y[o] != 0.0):
    pose = np.vstack((pose_y[o], np.array([[0, 0, 0, 1]], dtype=np.float32)))
    # print(f'{pose} \n')
    # pose[1] *= -1
    # pose[2] *= -1
    node = scene.add(mesh_y[o], name = 'object', pose=pose)
  
 
  # Load MANO layer.
  mano_layer = ManoLayer(flat_hand_mean=False,
                         ncomps=45,
                         side=sample['mano_side'],
                         mano_root='manopth/mano/models',
                         use_pca=True)
  faces = mano_layer.th_faces.numpy()
  betas = torch.tensor(sample['mano_betas'], dtype=torch.float32).unsqueeze(0)

  # Add MANO meshes.
  if not np.all(pose_m == 0.0):
    pose = torch.from_numpy(pose_m)
    vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
    vert /= 1000
    vert = vert.view(778, 3)
    vert = vert.numpy()
    # vert[:, 1] *= -1
    # vert[:, 2] *= -1
    mesh = trimesh.Trimesh(vertices=vert, faces=faces)
    mesh1 = pyrender.Mesh.from_trimesh(mesh)
    mesh1.primitives[0].material.baseColorFactor = [0.7, 0.7, 0.7, 1.0]
    mesh2 = pyrender.Mesh.from_trimesh(mesh, wireframe=True)
    mesh2.primitives[0].material.baseColorFactor = [0.0, 0.0, 0.0, 1.0]
    node1 = scene.add(mesh1,name = 'hand1')
    node2 = scene.add(mesh2, name = 'hand2')

  return scene

def kk_load_data(src):
  name = 's0_train' #已见过的物品 train的划分
  dataset = get_dataset(name)
  idx = 70
  sample = dataset[idx]

  _name_frame = src #'/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341/0_frame'
  _name = os.path.dirname(_name_frame) #'/home/e/eez095/dexycb_data/20200813-subject-02/20200813_145341'

  assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
  _data_dir = os.environ['DEX_YCB_DIR']

  # Load meta.
  meta_file = _name + "/meta.yml"
  with open(meta_file, 'r') as f:
    meta = yaml.load(f, Loader=yaml.FullLoader)
  _serials = meta['serials'] #每个相机是一个文件夹 
  extrinsic= meta['extrinsics']
  _h = 480
  _w = 640
  _ycb_ids = meta['ycb_ids']
  _ycb_grasp_ind = meta['ycb_grasp_ind']
  _mano_sides = meta['mano_sides']
  _color_format = "color_{:06d}.jpg"
  _depth_format = "aligned_depth_to_color_{:06d}.png"
  _label_format = "labels_{:06d}.npz"

  # extrinsic
  extr_file = _data_dir + "/calibration/" + "extrinsics_" + extrinsic +"/extrinsics.yml"
  with open(extr_file, 'r') as f_ex:
    extr = yaml.load(f_ex, Loader=yaml.FullLoader)
  master_seris = extr['master']
  
  # intrincs
  # _serials[seris_num] = '840412060917'
  # seris_num = 0
  # intr_file = _data_dir + "/calibration/intrinsics/" + _serials[seris_num] + '_' + str(
  intr_file = _data_dir + "/calibration/intrinsics/" + master_seris + '_' + str(
              _w) + 'x' + str(_h) + ".yml"
  with open(intr_file, 'r') as f_c:
    intr = yaml.load(f_c, Loader=yaml.FullLoader)
  fx = intr['color']['fx']
  fy = intr['color']['fy']
  cx = intr['color']['ppx']
  cy = intr['color']['ppy']
  #mano
  mano_calib_file = os.path.join(_data_dir, "calibration",
                                       "mano_{}".format(meta['mano_calib'][0]),
                                       "mano.yml")
  with open(mano_calib_file, 'r') as f:
    mano_calib = yaml.load(f, Loader=yaml.FullLoader)
  #-----
  frame = int(os.path.basename(_name_frame).split('_')[0])
  d_path = os.path.join(_name, master_seris)
  sample['color_file'] =  os.path.join(d_path, _color_format.format(frame))
  sample['label_file'] = os.path.join(d_path, _label_format.format(frame)) #得选主相机！
  sample['intrinsics'] =  {'fx': fx, 'fy': fy, 'ppx': cx, 'ppy': cy}
  sample['ycb_ids'] = _ycb_ids
  sample['mano_side'] = _mano_sides[0]
  sample['mano_betas'] = mano_calib['betas']
  sample['ycb_grasp_ind'] = _ycb_grasp_ind
  # import pdb 
  # pdb.set_trace()



  return sample,dataset
  
  #-----------create own data finish--------------------------------

def main():
  parser = argparse.ArgumentParser(description="Arguments for the script")
  parser.add_argument('--src')
  args = parser.parse_args()
  sample,dataset = kk_load_data(args.src)
 
  scene_r = create_scene(sample, dataset.obj_file)
  scene_v = create_scene(sample, dataset.obj_file)


  #-----------download Scene to .ply-------
  #scene_r pyrender.scene.Scene object
  scene = scene_r
  meshes = []
  all_vertices = []
  all_faces = []
  current_vertex_count = 0
  # Iterate over all nodes in the Pyrender scene
  num = 0
  for node in scene.mesh_nodes:
      # Get the mesh associated with the node
      mesh = node.mesh
      # Convert Pyrender mesh to Trimesh
      trimesh_mesh = trimesh.Trimesh(vertices=mesh.primitives[0].positions, faces=mesh.primitives[0].indices)
     
      pose = node.matrix
      transformed_vertices = trimesh_mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
     
      # Append transformed vertices and faces to lists
      all_vertices.append(transformed_vertices)
      all_faces.append(trimesh_mesh.faces + current_vertex_count)
      
      # Update the vertex count for face indexing
      current_vertex_count += len(trimesh_mesh.vertices)
      if node.name == 'hand1': #hand
        hand_vertices = np.vstack(transformed_vertices)
        hand_faces = np.vstack(trimesh_mesh.faces)
      if node.name == 'object': #object
        object_vertices = np.vstack(transformed_vertices)
        object_faces = np.vstack(trimesh_mesh.faces)
      num += 1

  # Combine all vertices and faces into single arrays
  combined_vertices = np.vstack(all_vertices)
  combined_faces = np.vstack(all_faces)

  # Create a single Trimesh object with combined data
  combined_trimesh = trimesh.Trimesh(vertices=combined_vertices,
                                      faces=combined_faces)
  hand_trimesh = trimesh.Trimesh(vertices=hand_vertices,
                                      faces=hand_faces)
  object_trimesh = trimesh.Trimesh(vertices=object_vertices,
                                      faces=object_faces)
  # Export to PLY format
  os.makedirs(os.path.join(args.src,'handover_3D'), exist_ok = True)
  combined_trimesh.export(os.path.join(args.src,'handover_3D','handover.ply'))
  hand_trimesh.export(os.path.join(args.src,'handover_3D','hand.ply'))
  object_trimesh.export(os.path.join(args.src,'handover_3D','object.ply'))
  # print('transform scene object to .ply object successfully')

  # print('Visualizing pose in camera view using pyrender renderer')

  r = pyrender.OffscreenRenderer(viewport_width=dataset.w,
                                 viewport_height=dataset.h)
  im_render, _ = r.render(scene_r)
  # render 方法的主要功能是将传入的三维场景（在这里是 scene_r）转换为二维图像

  im_real = cv2.imread(sample['color_file'])
  im_real = im_real[:, :, ::-1]

  im = 0.33 * im_real.astype(np.float32) + 0.67 * im_render.astype(np.float32)
  im = im.astype(np.uint8)

  # print('Close the window to continue.')

  plt.imshow(im)
  plt.tight_layout()
  plt.savefig('./visualize_pose.png', bbox_inches='tight')
  # plt.show()
  # print('Visualizing pose using pyrender 3D viewer')

  # pyrender.Viewer(scene_v)


if __name__ == '__main__':
  main()
