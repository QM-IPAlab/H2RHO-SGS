# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB sequence loader."""

import torch
import os
import yaml
import numpy as np
import cv2
import shutil

from scipy.spatial.transform import Rotation as Rot

from .layers.ycb_group_layer import YCBGroupLayer
from .layers.mano_group_layer import MANOGroupLayer
from .layers.ycb_layer import dcm2rv, rv2dcm


class SequenceLoader1():
  """DexYCB sequence loader."""

  def __init__(
      self,
      name,
      device='cuda:0',
      preload=True,
      app='viewer',
  ):
    """Constructor.

    Args:
      name: Sequence name.
      device: A torch.device string argument. The specified device is used only
        for certain data loading computations, but not storing the loaded data.
        Currently the loaded data is always stored as numpy arrays on CPU.
      preload: Whether to preload the point cloud or load it online.
      app: 'viewer' or 'renderer'.
    """
    assert device in ('cuda', 'cpu') or device.split(':')[0] == 'cuda'
    assert app in ('viewer', 'renderer')
    self._name = name
    self._device = torch.device(device)
    self._preload = preload
    self._app = app

    assert 'DEX_YCB_DIR' in os.environ, "environment variable 'DEX_YCB_DIR' is not set"
    self._dex_ycb_dir = os.environ['DEX_YCB_DIR']

    # Load meta.
    meta_file = self._dex_ycb_dir + '/' + self._name + "/meta.yml"
    with open(meta_file, 'r') as f:
      meta = yaml.load(f, Loader=yaml.FullLoader)

    self._serials = meta['serials'] #每个相机是一个文件夹 seris = ['836212060125', '839512060362'.......]
    # print(f'self._serials----------------{self._serials}')
    self._h = 480
    self._w = 640
    self._num_cameras = len(self._serials)
    self._data_dir = [
        self._dex_ycb_dir + '/' + self._name + '/' + s for s in self._serials
    ]
    self._color_prefix = "color_"
    self._depth_prefix = "aligned_depth_to_color_"
    self._label_prefix = "labels_"
    self._num_frames = meta['num_frames']
    self._ycb_ids = meta['ycb_ids']
    self._mano_sides = meta['mano_sides']

    # Load intrinsics.
    def intr_to_K(x):
      return torch.tensor(
          [[x['fx'], 0.0, x['ppx']], [0.0, x['fy'], x['ppy']], [0.0, 0.0, 1.0]],
          dtype=torch.float32,
          device=self._device)

    self._K = [] # 八个相机的内参
    for s in self._serials:
      intr_file = self._dex_ycb_dir + "/calibration/intrinsics/" + s + '_' + str(
          self._w) + 'x' + str(self._h) + ".yml"
      with open(intr_file, 'r') as f:
        intr = yaml.load(f, Loader=yaml.FullLoader)
      K = intr_to_K(intr['color'])
      self._K.append(K)
    self._K_inv = [torch.inverse(k) for k in self._K]

    # Load extrinsics.
    extr_file = self._dex_ycb_dir + "/calibration/extrinsics_" + meta[
        'extrinsics'] + "/extrinsics.yml"
    with open(extr_file, 'r') as f:
      extr = yaml.load(f, Loader=yaml.FullLoader)
    T = extr['extrinsics']
    T = {
        s: torch.tensor(T[s], dtype=torch.float32,
                        device=self._device).view(3, 4) for s in T
    }
    self._R = [T[s][:, :3] for s in self._serials]
    self._t = [T[s][:, 3] for s in self._serials]
    self._R_inv = [torch.inverse(r) for r in self._R]
    self._t_inv = [torch.mv(r, -t) for r, t in zip(self._R_inv, self._t)]
    # import pdb 
    # pdb.set_trace()
    self._master_intrinsics = self._K[[
        i for i, s in enumerate(self._serials) if s == extr['master']
    ][0]].cpu().numpy()
    #'840412060917'去不掉 因为是mater extrincs

    self._tag_R = T['apriltag'][:, :3]
    self._tag_t = T['apriltag'][:, 3]
    self._tag_R_inv = torch.inverse(self._tag_R)
    self._tag_t_inv = torch.mv(self._tag_R_inv, -self._tag_t) # R-1 * -T
    self._tag_lim = [-0.00, +1.20, -0.10, +0.70, -0.10, +0.70]

    # Compute texture coordinates.
    y, x = torch.meshgrid(torch.arange(self._h), torch.arange(self._w))
    x = x.float()
    y = y.float()
    s = torch.stack((x / (self._w - 1), y / (self._h - 1)), dim=2)
    self._pcd_tex_coord = [s.numpy()] * self._num_cameras

    # Compute rays.
    self._p = [] #用于存储每个相机的光线方向
    ones = torch.ones((self._h, self._w), dtype=torch.float32)
    xy1s = torch.stack((x, y, ones), dim=2).view(self._w * self._h, 3).t() #生成齐次坐标 每个像素 (x, y)，齐次坐标表示为 (x, y, 1)
    xy1s = xy1s.to(self._device)
    for c in range(self._num_cameras):
      p = torch.mm(self._K_inv[c], xy1s)
      self._p.append(p)

    #----------------------------------Load point cloud---------------------
    from tqdm import tqdm
    if self._preload:
      print('-------------------Preloading point cloud------------------')
      for i in tqdm(range(self._num_frames),desc='generate pointcloud'): # 某一时刻
        self._color_frame = []
        self._depth_frame = []
        for c in range(self._num_cameras): #遍历多个相机
          rgb, d = self._load_frame_rgbd(c, i)
          self._color_frame.append(rgb)
          self._depth_frame.append(d)
        
        self._color_frame = np.array(self._color_frame, dtype=np.uint8)
        self._depth_frame = np.array(self._depth_frame, dtype=np.uint16)
        self._pcd_rgb = [x for x in self._color_frame]
        self._pcd_vert_frame = []
        self._pcd_mask = []
        for c in range(self._num_cameras):
          p, m = self._deproject_depth_and_filter_points(self._depth_frame[c], c) # 一个相机，1帧的点云     
          self._pcd_vert_frame.append(p)
          self._pcd_mask.append(m)
        pcl = np.array(self._pcd_vert_frame)
        #  p ( 480, 640, 3)
        # self._pcd_vert list (8,480,640,3)
        import trimesh
        pcl = pcl.reshape(-1, 3)
        pcl_color_whole = np.array(self._pcd_rgb).reshape(-1,3)
        point_cloud = trimesh.PointCloud(vertices = pcl, colors = pcl_color_whole)
        # 保存点云为 PLY 文件
        ply_output_path = os.path.join(os.path.join(self._dex_ycb_dir, self._name), str(i)+ 'points3D.ply')
        point_cloud.export(ply_output_path)

      print(f'------------complete generate pointcloud-----------------')
      
      #original code for _color,depth,pcl
      self._color = []
      self._depth = []
      for c in range(self._num_cameras):
        color = []
        depth = []
        for i in range(self._num_frames):
          rgb, d = self._load_frame_rgbd(c, i)
          color.append(rgb)
          depth.append(d)
          # break #只添加第一帧
          
        self._color.append(color)
        self._depth.append(depth)
      self._color = np.array(self._color, dtype=np.uint8)
      self._depth = np.array(self._depth, dtype=np.uint16)
      self._pcd_rgb = [x for x in self._color]
      self._pcd_vert = []
      self._pcd_mask = []
      for c in range(self._num_cameras):
        p, m = self._deproject_depth_and_filter_points(self._depth[c], c) # 一个相机，74帧的点云    p:(72, 480, 640, 3) 
        self._pcd_vert.append(p)  
        self._pcd_mask.append(m)

    else:
      print('Loading point cloud online') #the whole scene pointcloud is empty
      self._pcd_rgb = [
          np.zeros((self._h, self._w, 3), dtype=np.uint8)
          for _ in range(self._num_cameras)
      ]
      self._pcd_vert = [
          np.zeros((self._h, self._w, 3), dtype=np.float32)
          for _ in range(self._num_cameras)
      ]

      self._pcd_mask = [
          np.zeros((self._h, self._w), dtype=np.bool_)
          for _ in range(self._num_cameras)
      ]
    
    #compute colmap format file
    print('-----------------generating colmap format--------------')
    
    for i in tqdm(range(self._num_frames),desc='generating colmap format'): # 某一时刻 i_frme文件夹里
        dist_folder = os.path.join(os.path.join(self._dex_ycb_dir, self._name) , f'{i}_frame' )
        if os.path.exists(dist_folder):
          shutil.rmtree(dist_folder) 
        os.makedirs(dist_folder, exist_ok=True)
        dist_img_folder = os.path.join(dist_folder,'images_8')
        os.makedirs(dist_img_folder, exist_ok=True)
       
        dist_sparse_folder = os.path.join(os.path.join(dist_folder,'sparse'),'0')
        os.makedirs(dist_sparse_folder, exist_ok=True)
        camara_file = os.path.join(dist_sparse_folder,'cameras.txt')
        images_file = os.path.join(dist_sparse_folder,'images.txt')
        point3D_file = os.path.join(dist_sparse_folder,'point3D.txt')
        ply_path = os.path.join(os.path.join(self._dex_ycb_dir, self._name), f'{i}points3D.ply')
        des_ply_path = os.path.join(dist_folder,'points3D.ply')
        shutil.move(ply_path,des_ply_path)

        for c in range(self._num_cameras):
          color_file = self._data_dir[
              c] + '/' + self._color_prefix + "{:06d}.jpg".format(i)
          destination_file = os.path.join( dist_img_folder, os.path.basename(os.path.dirname(color_file))) +'.jpg'
          shutil.copy(color_file,destination_file)
        
        num = 1
        for s in self._serials: #某个相机
          #---write cameras.txt
          intr_file = self._dex_ycb_dir + "/calibration/intrinsics/" + s + '_' + str(
              self._w) + 'x' + str(self._h) + ".yml"
          with open(intr_file, 'r') as f_c:
            intr = yaml.load(f_c, Loader=yaml.FullLoader)
          
          _color_file = os.path.join(os.path.join(os.path.join( self._dex_ycb_dir, self._name),s),'color_000000.jpg')
   
          img = cv2.imread(_color_file)
          height, width = img.shape[:2]
          
          fx = intr['color']['fx']
          fy = intr['color']['fy']
          cx = intr['color']['ppx']
          cy = intr['color']['ppy']
          with open(camara_file, 'a') as f_camera:
            f_camera.write(f'{s} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n')

          #---write images.txt
          # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
          R_s = T[s][:, :3]
          t_s = T[s][:, 3]
          R_s_inv = torch.inverse(R_s)
          t_s_inv = torch.mv(R_s_inv, -t_s)
         
          from scipy.spatial.transform import Rotation
          r = Rotation.from_matrix(R_s_inv.cpu().numpy())
          q = r.as_quat()
          IMAGE_ID = num
          QW = q[3]
          QX = q[0]
          QY = q[1]
          QZ = q[2]
          TX = t_s_inv[0]
          TY = t_s_inv[1] 
          TZ = t_s_inv[2]
          CAMERA_ID = s
          num += 1
          NAME = s+'.jpg'

          with open(images_file, 'a') as f_img:
            f_img.write(f'{IMAGE_ID} {QW} {QX} {QY} {QZ} {TX} {TY} {TZ} {CAMERA_ID} {NAME}\n \n')
    print('---------------complete generating colmap format--------------')

    # print('---------------mask out hand and object-----------------------')
    # for i in tqdm(range(self._num_frames),desc='generating mask pointcloud'): # 某一时刻 i_frme文件夹里
    #     self._color_frame = []
    #     self._depth_frame = []
    #     for c in range(self._num_cameras): #遍历多个相机
    #       rgb, d = self._load_frame_rgbd(c, i)
    #       self._color_frame.append(rgb)
    #       self._depth_frame.append(d)
        
    #     self._color_frame = np.array(self._color_frame, dtype=np.uint8)
    #     self._depth_frame = np.array(self._depth_frame, dtype=np.uint16)
    #     self._pcd_rgb = [x for x in self._color_frame]
        
    #     self._pcd_mask_hand_object = [] 
    #     self._color_pcd_mask_hand_object = []
    #     self.valid_handover_mask = []
    #     self._valid_mask_3d = []
    #     for c in range(self._num_cameras):
    #       label_file = self._data_dir[
    #           c] + '/' + self._label_prefix + "{:06d}.npz".format(i)

    #       p_mask, valid_mask, valid_mask_3d = self._deproject_depth_and_mask_hand_objct(self._depth_frame[c],self._color_frame[c], c,label_file)
    #       #p_mask [h,w,3] valid_mask [h,w] 

    #       # self.valid_handover_mask.append(valid_mask)
    #       self._pcd_mask_hand_object.append(p_mask)
    #       self._color_pcd_mask_hand_object.append(self._color_frame[c])
    #       self._valid_mask_3d.append(valid_mask_3d)
       
    #     pcl_mask_object_hand = np.array(self._pcd_mask_hand_object)
    #     pcl_color = np.array(self._color_pcd_mask_hand_object)
    #     pcl_mask_3d = np.array(self._valid_mask_3d)
        
    #     pcl_mask_object_hand = pcl_mask_object_hand.reshape(-1, 3)
    #     pcl_color = pcl_color.reshape(-1,3)
    #     pcl_mask_3d = pcl_mask_3d.reshape(-1,3)

    #     import pdb 
    #     pdb.set_trace()
    #     m = pcl_mask_3d[:, 2] <= 0
    #     pcl_mask_object_hand = pcl_mask_object_hand[m]
    #     pcl_color = pcl_color[m]

        
 
    #     # 保存点云为 PLY 文件
    #     import trimesh
    #     point_cloud_hand_object = trimesh.PointCloud(vertices = pcl_mask_object_hand, colors = pcl_color)   
    #     handobject_ply_output_path = os.path.join(os.path.join(os.path.join(self._dex_ycb_dir, self._name), f'{i}_frame'),'handover_points3D.ply')
    #     point_cloud_hand_object.export(handobject_ply_output_path)



    # Create YCB group layer.
    self._ycb_group_layer = YCBGroupLayer(self._ycb_ids).to(self._device)

    self._ycb_model_dir = self._dex_ycb_dir + "/models"
    self._ycb_count = self._ycb_group_layer.count
    self._ycb_material = self._ycb_group_layer.material
    self._ycb_tex_coords = self._ycb_group_layer.tex_coords

    # Create MANO group layer.
    mano_betas = []
    for m in meta['mano_calib']:
      mano_calib_file = self._dex_ycb_dir + "/calibration/mano_" + m + "/mano.yml"
      with open(mano_calib_file, 'r') as f:
        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
      betas = np.array(mano_calib['betas'], dtype=np.float32)
      mano_betas.append(betas)

    self._mano_group_layer = MANOGroupLayer(self._mano_sides,
                                            mano_betas).to(self._device)
    
    # Prepare data for viewer.
    if app == 'viewer':
      s = np.cumsum([0] + self._ycb_group_layer.count[:-1])
      e = np.cumsum(self._ycb_group_layer.count)
      self._ycb_seg = list(zip(s, e))

      ycb_file = self._dex_ycb_dir + '/' + self._name + "/pose.npz"
      data = np.load(ycb_file)
      ycb_pose = data['pose_y']
      i = np.any(ycb_pose != [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=2)
      pose = ycb_pose.reshape(-1, 7)
      v, n = self.transform_ycb(pose)
      self._ycb_vert = [
          np.zeros((self._num_frames, n, 3), dtype=np.float32)
          for n in self._ycb_count
      ]
      self._ycb_norm = [
          np.zeros((self._num_frames, n, 3), dtype=np.float32)
          for n in self._ycb_count
      ]
      for o in range(self._ycb_group_layer.num_obj):
        io = i[:, o]
        self._ycb_vert[o][io] = v[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]
        self._ycb_norm[o][io] = n[io, self._ycb_seg[o][0]:self._ycb_seg[o][1]]

      mano_file = self._dex_ycb_dir + '/' + self._name + "/pose.npz"
      data = np.load(mano_file)
      mano_pose = data['pose_m']
      i = np.any(mano_pose != 0.0, axis=2)
      pose = torch.from_numpy(mano_pose).to(self._device)
      pose = pose.view(-1, self._mano_group_layer.num_obj * 51)
      verts, _ = self._mano_group_layer(pose)
      # Numpy array is faster than PyTorch Tensor here.
      verts = verts.cpu().numpy()
      f = self._mano_group_layer.f.cpu().numpy()
      v = verts[:, f.ravel()]
      n = np.cross(v[:, 1::3, :] - v[:, 0::3, :], v[:, 2::3, :] - v[:, 1::3, :])
      n = np.repeat(n, 3, axis=1)
      l = verts[:, f[:, [0, 1, 1, 2, 2, 0]].ravel(), :]
      self._mano_vert = [
          np.zeros((self._num_frames, 4614, 3), dtype=np.float32)
          for _ in range(self._mano_group_layer.num_obj)
      ]
      self._mano_norm = [
          np.zeros((self._num_frames, 4614, 3), dtype=np.float32)
          for _ in range(self._mano_group_layer.num_obj)
      ]
      self._mano_line = [
          np.zeros((self._num_frames, 9228, 3), dtype=np.float32)
          for _ in range(self._mano_group_layer.num_obj)
      ]
      for o in range(self._mano_group_layer.num_obj):
        io = i[:, o]
        self._mano_vert[o][io] = v[io, 4614 * o:4614 * (o + 1), :]
        self._mano_norm[o][io] = n[io, 4614 * o:4614 * (o + 1), :]
        self._mano_line[o][io] = l[io, 9228 * o:9228 * (o + 1), :]

    # Prepare data for renderer.
    if app == 'renderer':
      self._ycb_pose = []
      self._mano_vert = []
      self._mano_joint_3d = []

      for c in range(self._num_cameras):
        ycb_pose = []
        mano_pose = []
        mano_joint_3d = []
        for i in range(self._num_frames):
          label_file = self._data_dir[
              c] + '/' + self._label_prefix + "{:06d}.npz".format(i)
          label = np.load(label_file)
          pose_y = np.hstack((label['pose_y'],
                              np.array([[[0, 0, 0, 1]]] * len(label['pose_y']),
                                       dtype=np.float32)))
          pose_m = label['pose_m']
          joint_3d = label['joint_3d']
          ycb_pose.append(pose_y)
          mano_pose.append(pose_m)
          mano_joint_3d.append(joint_3d)
        ycb_pose = np.array(ycb_pose, dtype=np.float32)
        mano_pose = np.array(mano_pose, dtype=np.float32)
        mano_joint_3d = np.array(mano_joint_3d, dtype=np.float32)
        self._ycb_pose.append(ycb_pose)
        self._mano_joint_3d.append(mano_joint_3d)

        i = np.any(mano_pose != 0.0, axis=2)
        pose = torch.from_numpy(mano_pose).to(self._device)
        pose = pose.view(-1, self._mano_group_layer.num_obj * 51)
        verts, _ = self._mano_group_layer(pose)
        verts = verts.cpu().numpy()
        mano_vert = [
            np.zeros((self._num_frames, 778, 3), dtype=np.float32)
            for _ in range(self._mano_group_layer.num_obj)
        ]
        for o in range(self._mano_group_layer.num_obj):
          io = i[:, o]
          mano_vert[o][io] = verts[io, 778 * o:778 * (o + 1), :]
        self._mano_vert.append(mano_vert)

    self._frame = -1


  def _load_frame_rgbd(self, c, i):
    """Loads an RGB-D frame.

    Args:
      c: Camera index.
      i: Frame index.

    Returns:
      color: A unit8 numpy array of shape [H, W, 3] containing the color image.
      depth: A uint16 numpy array of shape [H, W] containing the depth image.
    """
    color_file = self._data_dir[
        c] + '/' + self._color_prefix + "{:06d}.jpg".format(i)
    color = cv2.imread(color_file)
    color = color[:, :, ::-1]
    depth_file = self._data_dir[
        c] + '/' + self._depth_prefix + "{:06d}.png".format(i)
    depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
    return color, depth
  
  def _generate_colmap_image(self, c, i):
    """generate colmap image file.

    Args:
      c: Camera index.
      i: Frame index.

    """
    color_file = self._data_dir[
        c] + '/' + self._color_prefix + "{:06d}.jpg".format(i)
    
    return color, depth

  def _deproject_depth_and_filter_points(self, d, c):
    """Deprojects a depth image to point cloud and filters points.

    Args:
      d: A uint16 numpy array of shape [F, H, W] or [H, W] containing the depth
        image in millimeters. 形状可以是 [F, H, W]（多帧）或 [H, W]（单帧），单位为毫米。
      c: Camera index. 

    Returns:
      p: A float32 numpy array of shape [F, H, W, 3] or [H, W, 3] containing the
        point cloud.
      m: A bool numpy array of shape [F, H, W] or [H, W] containing the mask for
        points within the tag cooridnate limit.
    """
    nd = d.ndim
    d = d.astype(np.float32) / 1000 #单位转换为米
    d = torch.from_numpy(d).to(self._device)
    p = torch.mul(
        d.view(1, -1, self._w * self._h).expand(3, -1, -1),
        self._p[c].unsqueeze(1)) #k_inverse
        # d:[74, 480, 640] -> [1, 74, 307200] -> [3, 74, 307200]  
        # self._p[c]:[3, 307200]-> self._p[c].unsqueeze(1) [3, 1, 307200]
        # p:[3, 74, 307200] from pixel fram to camera frame
    p = torch.addmm(self._t[c].unsqueeze(1), self._R[c], p.view(3, -1)) # world coordinate
    p_tag = torch.addmm(self._tag_t_inv.unsqueeze(1), self._tag_R_inv, p) #tage coordinate
    mx1 = p_tag[0, :] > self._tag_lim[0]
    mx2 = p_tag[0, :] < self._tag_lim[1]
    my1 = p_tag[1, :] > self._tag_lim[2]
    my2 = p_tag[1, :] < self._tag_lim[3]
    mz1 = p_tag[2, :] > self._tag_lim[4]
    mz2 = p_tag[2, :] < self._tag_lim[5]
    m = mx1 & mx2 & my1 & my2 & mz1 & mz2 #指示哪些点在有效范围内
    p = p.t().view(-1, self._h, self._w, 3)
    m = m.view(-1, self._h, self._w)
    if nd == 2:
      p = p.squeeze(0)
      m = m.squeeze(0)
    p = p.cpu().numpy()
    m = m.cpu().numpy()
    return p, m

  def _deproject_depth_and_mask_hand_objct(self, d,color, c, l):
    """Deprojects a depth image to point cloud and filters points.

    Args:
      d: A uint16 numpy array of shape [F, H, W] or [H, W] containing the depth
        image in millimeters. 形状可以是 [F, H, W]（多帧）或 [H, W]（单帧），单位为毫米。
      color:A uint16 numpy array of shape [F, H, W] or [H, W] containing the
        image in millimeters. 形状可以是 [F, H, W]（多帧）或 [H, W]（单帧），单位为毫米
      c: Camera index. 
      l:label file

    Returns:
      p: A float32 numpy array of shape [F, H, W, 3] or [H, W, 3] containing the
        point cloud.
     
    """
    nd = d.ndim
    d_mask = d
    label = np.load(l)
    seg = label['seg']
    #1-21 (YCB object), or 255 (hand).
    self.object_catgry = 5
    _hand =255  
    count = 0
    d_mask = d_mask.astype(np.float32) / 1000 #单位转换为米
    valid_mask = np.full((self._h, self._w), 10)

    
    for v in range(self._h):
      for u in range(self._w):
        label_uv = seg[v][u]
        if(label_uv == 255):
          count +=1
        elif label_uv != 255:
          valid_mask[v][u] = -10
          
    
   
    valid_mask_3d = torch.from_numpy(valid_mask).to(self._device)
    valid_mask_3d = torch.mul(
        valid_mask_3d.view(1, -1, self._w * self._h).expand(3, -1, -1),
        self._p[c].unsqueeze(1))
    valid_mask_3d = torch.addmm(self._t[c].unsqueeze(1), self._R[c], valid_mask_3d.view(3, -1)) # world coordinate
    valid_mask_3d = valid_mask_3d.t().view(-1, self._h, self._w, 3)
    if nd == 2:
      valid_mask_3d = valid_mask_3d.squeeze(0)
    valid_mask_3d = valid_mask_3d.cpu().numpy()


    d_mask = torch.from_numpy(d_mask).to(self._device)
    p_mask = torch.mul(
        d_mask.view(1, -1, self._w * self._h).expand(3, -1, -1),
        self._p[c].unsqueeze(1)) 
    p_mask = torch.addmm(self._t[c].unsqueeze(1), self._R[c], p_mask.view(3, -1)) # world coordinate
  
    p_mask = p_mask.t().view(-1, self._h, self._w, 3)
   
    if nd == 2:
      p_mask = p_mask.squeeze(0)
     
    p_mask = p_mask.cpu().numpy()
   
   
    return p_mask,valid_mask,valid_mask_3d


  def transform_ycb(self,
                    pose,
                    c=None,
                    camera_to_world=True,
                    run_ycb_group_layer=True,
                    return_trans_mat=False):
    """Transforms poses in SE3 between world and camera frames.

    Args:
      pose: A float32 numpy array of shape [N, 7] or [N, 6] containing the
        poses. Each row contains one pose represented by rotation in quaternion
        (x, y, z, w) or rotation vector and translation.
      c: Camera index.
      camera_to_world: Whether from camera to world or from world to camera.
      run_ycb_group_layer: Whether to return vertices and normals by running the
        YCB group layer or to return poses.
      return_trans_mat: Whether to return poses in transformation matrices.

    Returns:
      If run_ycb_group_layer is True:
        v: A float32 numpy array of shape [F, V, 3] containing the vertices.
        n: A float32 numpy array of shape [F, V, 3] containing the normals.
      else:
        A float32 numpy array of shape [N, 6] containing the transformed poses.
    """
    if pose.shape[1] == 7:
      q = pose[:, :4]
      t = pose[:, 4:]
      R = Rot.from_quat(q).as_dcm().astype(np.float32)
      R = torch.from_numpy(R).to(self._device)
      t = torch.from_numpy(t).to(self._device)
    if pose.shape[1] == 6:
      r = pose[:, :3]
      t = pose[:, 3:]
      r = torch.from_numpy(r).to(self._device)
      t = torch.from_numpy(t).to(self._device)
      R = rv2dcm(r)
    if c is not None:
      if camera_to_world:
        R_c = self._R[c]
        t_c = self._t[c]
      else:
        R_c = self._R_inv[c]
        t_c = self._t_inv[c]
      R = torch.bmm(R_c.expand(R.size(0), -1, -1), R)
      t = torch.addmm(t_c, t, R_c.t())
    if run_ycb_group_layer or not return_trans_mat:
      r = dcm2rv(R)
      p = torch.cat([r, t], dim=1)
    else:
      p = torch.cat([R, t.unsqueeze(2)], dim=2)
      p = torch.cat([
          p,
          torch.tensor([[[0, 0, 0, 1]]] * R.size(0),
                       dtype=torch.float32,
                       device=self._device)
      ],
                    dim=1)
    if run_ycb_group_layer:
      p = p.view(-1, self._ycb_group_layer.num_obj * 6)
      v, n = self._ycb_group_layer(p)
      v = v[:, self._ycb_group_layer.f.view(-1)]
      n = n[:, self._ycb_group_layer.f.view(-1)]
      v = v.cpu().numpy()
      n = n.cpu().numpy()
      return v, n
    else:
      p = p.cpu().numpy()
      return p


  @property
  def serials(self):
    return self._serials

  @property
  def num_cameras(self):
    return self._num_cameras

  @property
  def num_frames(self):
    return self._num_frames

  @property
  def dimensions(self):
    return self._w, self._h

  @property
  def ycb_ids(self):
    return self._ycb_ids

  @property
  def K(self):
    return self._K

  @property
  def master_intrinsics(self):
    return self._master_intrinsics

  def step(self):
    """Steps the frame."""
    self._frame = (self._frame + 1) % self._num_frames
    if not self._preload:
      self._update_pcd()

  def _update_pcd(self):
    """Updates the point cloud."""
    for c in range(self._num_cameras):
      rgb, d = self._load_frame_rgbd(c, self._frame)
      p, m = self._deproject_depth_and_filter_points(d, c)
      self._pcd_rgb[c][:] = rgb
      self._pcd_vert[c][:] = p
      self._pcd_mask[c][:] = m

  @property
  def pcd_rgb(self):
    if self._preload:
      return [x[self._frame] for x in self._pcd_rgb]
    else:
      return self._pcd_rgb

  @property
  def pcd_vert(self):
    if self._preload:
      return [x[self._frame] for x in self._pcd_vert]
    else:
      return self._pcd_vert

  @property
  def pcd_tex_coord(self):
    return self._pcd_tex_coord

  @property
  def pcd_mask(self):
    if self._preload:
      return [x[self._frame] for x in self._pcd_mask]
    else:
      return self._pcd_mask

  @property
  def ycb_group_layer(self):
    return self._ycb_group_layer

  @property
  def num_ycb(self):
    return self._ycb_group_layer.num_obj

  @property
  def ycb_model_dir(self):
    return self._ycb_model_dir

  @property
  def ycb_count(self):
    return self._ycb_count

  @property
  def ycb_material(self):
    return self._ycb_material

  @property
  def ycb_pose(self):
    if self._app == 'viewer':
      return None
    if self._app == 'renderer':
      return [x[self._frame] for x in self._ycb_pose]

  @property
  def ycb_vert(self):
    if self._app == 'viewer':
      return [x[self._frame] for x in self._ycb_vert]
    if self._app == 'renderer':
      return None

  @property
  def ycb_norm(self):
    if self._app == 'viewer':
      return [x[self._frame] for x in self._ycb_norm]
    if self._app == 'renderer':
      return None

  @property
  def ycb_tex_coords(self):
    return self._ycb_tex_coords

  @property
  def mano_group_layer(self):
    return self._mano_group_layer

  @property
  def num_mano(self):
    return self._mano_group_layer.num_obj

  @property
  def mano_vert(self):
    if self._app == 'viewer':
      return [x[self._frame] for x in self._mano_vert]
    if self._app == 'renderer':
      return [[y[self._frame] for y in x] for x in self._mano_vert]

  @property
  def mano_norm(self):
    if self._app == 'viewer':
      return [x[self._frame] for x in self._mano_norm]
    if self._app == 'renderer':
      return None

  @property
  def mano_line(self):
    if self._app == 'viewer':
      return [x[self._frame] for x in self._mano_line]
    if self._app == 'renderer':
      return None

  @property
  def mano_joint_3d(self):
    if self._app == 'viewer':
      return None
    if self._app == 'renderer':
      return [x[self._frame] for x in self._mano_joint_3d]
