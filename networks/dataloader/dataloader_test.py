# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Zerong Zheng (zzr18@mails.tsinghua.edu.cn)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the <organization> nor the
#  names of its contributors may be used to endorse or promote products
#  derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Data loader"""

from __future__ import division, print_function

import os
import glob
import math

import cv2
import numpy as np
import scipy.spatial
import scipy.io as sio
import pickle as pkl
import json
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import networks.constant as constant
from networks.dataloader.utils import load_data_list, generate_cam_Rt,rotate_vertex

from networks.util import util
from networks.util import obj_io
import trimesh
from .util import load_obj,projection
from networks.graph_cmr.models.get_joints_sample import gaussian_sampling_around_keypoints
# import pykdtree.kdtree as kd
from  networks.models.train_options import TrainOptions
from networks.graph_cmr.models import GraphCNN,  SMPL
from PIL import Image


class TrainingImgDataset(Dataset):
    def __init__(self, dataset_dir,
                 img_h, img_w, training, testing_res,
                 view_num_per_item, point_num, load_pts2smpl_idx_wgt,
                 smpl_data_folder='/media/gpu/dataset_SSD/code/PaMIR-main/networks/data'):
        super(TrainingImgDataset, self).__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.training = training
        self.testing_res = testing_res
        self.dataset_dir = dataset_dir
        self.view_num_per_item = view_num_per_item
        self.point_num = point_num
        self.load_pts2smpl_idx_wgt = load_pts2smpl_idx_wgt
        self.data_aug = self.training

        self.data_list = load_data_list("/media/gpu/备份/GYQ/thuman2_CHON/thuman2_360views","list.txt")
        # self.data_list =("/media/gpu/dataset_SSD/dataset/TH2Render")
        # self.len = len(self.data_list) * self.view_num_per_item
        self.len = len(self.data_list)
        # load smpl model data for usage
        jmdata = np.load(os.path.join(smpl_data_folder, 'joint_model.npz'))
        self.J_dirs = jmdata['J_dirs']
        self.J_template= jmdata['J_template']
        # SMPL model
        # self.device = torch.device('cuda')
        # self.smpl = SMPL().to(self.device)
        # some default parameters for testing
        self.default_testing_cam_R = constant.cam_R
        self.default_testing_cam_t = constant.cam_t
        self.default_testing_cam_f = constant.cam_f
        self.smpl_vertex_code, self.smpl_face_code, self.smpl_faces, self.smpl_tetras = \
            util.read_smpl_constants('/media/gpu/dataset_SSD/code/PaMIR-main/networks/data')
        self.opt = TrainOptions().parse()

        # self.smpl = SMPL(
        #     '/media/gpu/dataset_SSD/code/Cloth-shift/PaMIR-main/networks/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')


    def __len__(self):
        return self.len

    def __getitem__(self, item):

        p = np.random.rand()
        data_list = self.data_list

        model_id = item // self.view_num_per_item
        view_id = (item % self.view_num_per_item)*(360/self.view_num_per_item)
        # view_id=120

        data_item = data_list[model_id]

        cam_f = self.default_testing_cam_f
        point_num = self.point_num

        img,_,_ = self.load_image(data_item, view_id)
        # scan2smpl_gt = self.load_image(data_item, view_id)
        # cam_R, cam_t = self.load_cams(data_item, view_id)
        # pts_ids, pts, pts_ov = self.load_points(data_item, point_num)
        # pts_r = self.rotate_points(pts, view_id)
        # pts_proj = self.project_points(pts, cam_R, cam_t, cam_f)
        # pts_clr = pts_clr * alpha + beta
        # label_path_vs_smplwithot_calib,label_path_vs_smpl,meshcnn_scale,meshcnn_trans,meshcnn_calib_mat = self.load_smpl_parameters2(data_item,model_id,view_id)

        pose, betas, trans, scale = self.load_smpl_parameters(data_item)
        pose, betas, trans, scale= self.update_smpl_params(pose, betas, trans, scale, view_id)
        calib_mat=self.load_gt_mesh_calib( data_item, view_id)
        # label_path_vs,label_path_norm,label_path_faces,calib_mat=self.load_gt_mesh( data_item, view_id)
        feature_fine, feature_global=self.load_feature(data_item,view_id)
        # hrnet_keypoints = self.load_hrnet_keypoints(data_item,view_id)
        # self.draw_hrnet_keypoints(data_item,view_id)
        # p2ps=self.load_p2m( data_item, view_id)


        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        calib_former = torch.Tensor(projection_matrix).float()
        calib = calib_former


        return_dict = {
            'model_id': int(data_item),
            'view_id': int(view_id),
            'data_item': data_item,
            'img': torch.from_numpy(img.transpose((2, 0, 1))),
            # 'diff_S_gt':torch.from_numpy(scan2smpl_gt),
            # 'img_gray': torch.from_numpy(img_gray),
            # 'mask':torch.from_numpy(msk),
            'betas': torch.from_numpy(betas),
            'pose': torch.from_numpy(pose),
            # "gt_vertices_cam":torch.from_numpy(label_path_vs_smpl).float(),
            'scale': torch.from_numpy(scale),
            'trans': torch.from_numpy(trans),
            'img_dir':os.path.join(self.dataset_dir, data_item, 'render/%04d.png' % view_id),
            "feature_global":torch.Tensor(feature_global).float(),
            "feature_local": torch.Tensor(feature_fine).float(),
            "calib":calib,
            # 'label_path_vs': torch.Tensor(label_path_vs).float(),
            # 'label_path_faces': torch.Tensor(label_path_faces).float(),
            # 'label_path_norm': torch.tensor(label_path_norm).float(),
            "calib_data":torch.tensor(calib_mat).float(),
            # "gt_dis":torch.from_numpy(p2ps)
            # 'hrnet_keypoint': torch.from_numpy(hrnet_keypoints),
            # "meshcnn_scale":torch.tensor(meshcnn_scale).float(),
            #  "meshcnn_trans":torch.tensor(meshcnn_trans).float(),
            # "label_path_vs_smplwithot_calib":torch.tensor(label_path_vs_smplwithot_calib),
            # 'near_keypoints_3d':torch.from_numpy(nearby_points_3d),
            # "no_near_keypoints_3d":torch.from_numpy(no_nearby_points_3d),
            # 'keypoints2d_nearby_points':torch.from_numpy(nearby_points_2d)
            #     cam_R, cam_t, cam_f
            #     'cam_R':torch.from_numpy(cam_R.transpose()),
            #     'cam_t':torch.from_numpy(cam_t),

            #     'keypoints_test': torch.from_numpy(keypoints_test),

        }

        return return_dict

    def load_p2m(self,data_item,view_id):
        # jisuan gt_smpl meigediandao gt_scan de zuiduanjuli
        distance_path=os.path.join(self.dataset_dir, data_item, "p2s", '%04d.txt' % (view_id))
        distance_data = np.loadtxt(distance_path, dtype=float)
        return distance_data

    def load_gt_mesh_calib(self,data_item,view_id):
        # label_path = os.path.join(self.dataset_dir, data_item,'%04s.obj' % (data_item))
        # label_path_vs, label_path_faces = load_obj(label_path)
        calib_path= os.path.join(self.dataset_dir, data_item,"calib", '%04d.txt' % (view_id))
        # calib_path = '/media/star/备份/GYQ/thuman2_CHON/thuman2_360views/%04s/calib/%04d.txt' % (
        # data_item, view_id )
        calib_data = np.loadtxt(calib_path, dtype=float)

        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        # calib_verts = projection(label_path_vs * 100, calib_mat)
        # calib_verts[:, 1] *= -1
        # label_path_vs =calib_verts
        #
        # tmesh = trimesh.Trimesh(vertices=label_path_vs, faces=label_path_faces)
        #
        # label_path_vs = tmesh.vertices
        # label_path_norm = tmesh.vertex_normals

        return calib_mat
    def load_gt_mesh(self,data_item,view_id):
        label_path = os.path.join(self.dataset_dir, data_item,'%04s.obj' % (data_item))
        label_path_vs, label_path_faces = trimesh.load(label_path).vertices, trimesh.load(label_path).faces
        calib_path= os.path.join(self.dataset_dir, data_item,"calib", '%04d.txt' % (view_id))
        # calib_path = '/media/star/备份/GYQ/thuman2_CHON/thuman2_360views/%04s/calib/%04d.txt' % (
        # data_item, view_id )
        calib_data = np.loadtxt(calib_path, dtype=float)

        extrinsic = calib_data[:4, :4]
        intrinsic = calib_data[4:8, :4]
        calib_mat = np.matmul(intrinsic, extrinsic)
        calib_verts = projection(label_path_vs * 100, calib_mat)
        calib_verts[:, 1] *= -1


        tmesh = trimesh.Trimesh(vertices=calib_verts, faces=label_path_faces)

        label_path_vs = tmesh.vertices
        label_path_norm = tmesh.vertex_normals

        return label_path_vs,label_path_norm,tmesh.faces,calib_mat
    def load_feature(self,data_item, view_id):
        if self.opt.phase == 'train':
            # normals_f = Image.open(r'/media/star/备份/systhesizeDataset/%04d/normal_F/%03d.png' % (model_id, view_id * 20)).convert('RGB')
            # normals_b = Image.open(r'/media/star/备份/systhesizeDataset/%04d/normal_B/%03d.png' % (model_id, view_id * 20)).convert('RGB')
            feature_fine_path = os.path.join("/media/gpu/备份/GYQ/thuman2_CHON/pifuhd_feature", data_item, "feature_hd_local", '%04d.npy' % (view_id))
            feature_global_path = os.path.join("/media/gpu/备份/GYQ/thuman2_CHON/pifuhd_feature", data_item, "feature_hd", '%04d.npy' % (view_id))
            # feature_fine_path = r'/media/star/备份/GYQ/thuman2_CHON/feature_hd_fine/%04s/%03d.npy' % (data_item, view_id )
            # feature_global_path = r'/media/star/备份/GYQ/thuman2_CHON/feature_hd/%04s/%03d.npy' % (data_item, view_id )

        else:
            feature_fine_path = r'/media/star/备份/GYQ/thuman2_CHON/feature_hd_fine/%04s/%03d.npy' % (data_item, view_id)
            feature_global_path = r'/media/star/备份/GYQ/thuman2_CHON/feature_hd/%04s/%03d.npy' % (data_item, view_id)

            # normals_f = Image.open(
            #     r'/media/amax/4C76448F76447C28/LYH/realworld/extreme/input/pymafx/econ_normal/%04d_normal_F.png' % (
            #         view_id)).convert('RGB')
            # normals_b = Image.open(
            #     r'/media/amax/4C76448F76447C28/LYH/realworld/extreme/input/pymafx/econ_normal/%04d_normal_B.png' % (
            #         view_id)).convert('RGB')
            # feature_fine_path = r'/media/amax/4C76448F76447C28/LYH/realworld/extreme/feature_hd_local/%04d.npy' % (
            #     view_id)
            # feature_global_path = r'/media/amax/4C76448F76447C28/LYH/realworld/extreme/feature_hd/%04d.npy' % (
            #     view_id)
        try:
            feature_fine = np.load(feature_fine_path)
            # feature_fine = feature_fine[None, :, :, :]
            # feature_fine = torch.Tensor(feature_fine).to(self.device).float()

            feature_global = np.load(feature_global_path)
            # feature_global = feature_global[None, :, :, :]
        except:
            print(feature_fine_path, feature_global_path)
        # print(feature_fine_path, feature_global_path)
        return feature_fine,feature_global




    def load_keypoints(self, data_item, view_id):
        # print(os.path.join(self.dataset_dir, data_item, 'keypoints/%04d_keypoints.json' % view_id))
        with open(os.path.join(self.dataset_dir, data_item, 'keypoints/%04d_keypoints.json' % view_id)) as fp:
            data = json.load(fp)
        keypoints = []
        if 'people' in data:
            for idx, person_data in enumerate(data['people']):
                kp_data = np.array(person_data['pose_keypoints_2d'], dtype=np.float64)
                kp_data = kp_data.reshape([-1, 3])
                kp_data = kp_data[constant.body25_to_joint]      # rearrange keypoints
                kp_data[constant.body25_to_joint < 0] =None     # remove undefined keypoints
                kp_data=kp_data[constant.smplbody24_to_joint >= 0]
                # kp_data[:, 0] = kp_data[:, 0]*2 / self.img_w - 1.0
                # kp_data[:, 1] = kp_data[:, 1]*2 / self.img_h - 1.0
                keypoints.append(kp_data)
        # mask=np.array(keypoints[0])!=np.array([-1.,-1. ,0.])
        # print(mask.shape)
        # print(keypoints)
        # keypoints[:, 0] = keypoints[:, 0]*2 / self.img_w - 1.0
        # keypoints[:, 1] = keypoints[:, 1]*2 / self.img_h - 1.0

        if len(keypoints) == 0:
            keypoints.append(np.zeros([24, 3]))
        # print("np.array(keypoints[0], dtype=np.float32):",np.array(keypoints[0], dtype=np.float32).shape)
        return np.array(keypoints[0], dtype=np.float32)

    def load_keypoints_test(self, data_item, view_id):
        # print(os.path.join(self.dataset_dir, data_item, 'keypoints/%04d_keypoints.json' % view_id))
        with open(os.path.join(self.dataset_dir, data_item, 'keypoints/%04d_keypoints.json' % view_id)) as fp:
            data = json.load(fp)
        keypoints = []
        if 'people' in data:
            for idx, person_data in enumerate(data['people']):
                kp_data = np.array(person_data['pose_keypoints_2d'], dtype=np.float64)
                kp_data = kp_data.reshape([-1, 3])
                kp_data = kp_data[constant.body25_to_joint]      # rearrange keypoints
                kp_data[constant.body25_to_joint < 0] *=0.0    # remove undefined keypoints
                kp_data[:, 0] = kp_data[:, 0]*2 / self.img_w - 1.0
                kp_data[:, 1] = kp_data[:, 1]*2 / self.img_h - 1.0
                keypoints.append(kp_data)
        # mask=np.array(keypoints[0])!=np.array([-1.,-1. ,0.])
        # print(mask.shape)
        # print(keypoints)
        if len(keypoints) == 0:
            keypoints.append(np.zeros([24, 3]))
        # print("np.array(keypoints[0], dtype=np.float32):",np.array(keypoints[0], dtype=np.float32).shape)
        return np.array(keypoints[0], dtype=np.float32)

    def load_image(self, data_item, view_id):
        img_fpath = os.path.join(
            self.dataset_dir, data_item, 'render/%04d.png' % view_id)
        # msk_fpath = os.path.join(
        #     self.dataset_dir,  data_item, 'mask/%04d.png' % view_id)
        # print(img_fpath)
        try:
            img = cv.imread(img_fpath).astype(np.uint8)
            # img = cv.resize(img, (224, 224))
            # msk = cv.imread(msk_fpath).astype(np.uint8)
            # msk = cv.resize(msk, (224, 224))
        except:
            raise RuntimeError('Failed to load image: ' + img_fpath)

        assert img.shape[0] == self.img_h and img.shape[1] == self.img_w
        img = np.float32(cv.cvtColor(img, cv.COLOR_RGB2BGR)) / 255
        return img, 1.0, 0.0
    def  load_keypoints_near(self,coords,keypoints_2d):
        mask=np.ones((17),dtype=np.int)
        for index in range(17):
            if(keypoints_2d[index][0]<10 or keypoints_2d[index][1]<10):
                mask[index]=0

        # keypoints_2d=keypoints_2d[mask!=0][..., :-1]
        keypoints_2d = keypoints_2d[..., :-1]
        keypoints_2d[:, 0] = keypoints_2d[:, 0]*2 / self.img_w - 1.0
        keypoints_2d[:, 1] = keypoints_2d[:, 1]*2 / self.img_h - 1.0
        mu = 0
        sigma = 0.022
        points = np.random.normal(loc=mu, scale=sigma, size=(keypoints_2d.shape[0], 32, 2))
        # 将采样的点加到每个点上，得到每个点周围的64个点
        neighborhoods = points + np.expand_dims(keypoints_2d, axis=1)
        # 将结果重塑为 (17*64, 2) 的数组
        nearby_points_2d = np.reshape(neighborhoods, (-1, 2))



        T_joints = np.array(obj_io.load_obj_data("/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/joints.obj")['v'])
        T_joints=T_joints[constant.smplbody24_to_joint >= 0]
        # T_joints = T_joints[mask!=0]
        nearby_points_3d,no_nearby_points_3d=self.load_keypoints_near_3d(T_joints)

        # print(nearby_points_3d.shape,no_nearby_points_3d.shape,nearby_points_2d.shape)

        return nearby_points_3d,no_nearby_points_3d,nearby_points_2d



    def load_keypoints_near_2d(self,coords,keypoints):
        from sklearn.neighbors import KDTree
        # mask = constant.smplbody24_to_joint >= 0

        T_joints = keypoints[..., :-1]

        k = 160

        # 存储所有的临近点
        nearby_points = np.zeros((keypoints.shape[0], k, 2))

        # 寻找每个关节点最近的k个点
        for i in range(keypoints.shape[0]):
            kdtree = KDTree(coords)
            joint = T_joints[i]
            dists, indices = kdtree.query(joint.reshape(1, -1), k=k)

            # 保存临近点
            nearby_points[i] = coords[indices[0]]


            coords = np.delete(coords, indices, axis=0)

        # 将临近点矩阵展平
        nearby_points = nearby_points.reshape(-1, 2)

        print("Nearby points shape:", nearby_points.shape)
        return nearby_points

    def load_keypoints_near_3d(self, T_joints):
        from sklearn.neighbors import KDTree
        mask = constant.smplbody24_to_joint >= 0
        ref_vertices = np.array(
            obj_io.load_obj_data("/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/ref_vertices.obj")['v'])
        # T_joints = np.array(
        #     obj_io.load_obj_data("/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/joints.obj")['v'])
        # T_joints = T_joints[mask]

        # 构建KD树
        # kdtree = KDTree(ref_vertices)
        k = 32

        # 存储所有的临近点
        nearby_points = np.zeros((T_joints.shape[0], k, 3))

        # 寻找每个关节点最近的k个点
        for i in range(T_joints.shape[0]):
            kdtree = KDTree(ref_vertices)
            joint = T_joints[i]
            dists, indices = kdtree.query(joint.reshape(1, -1), k=k)

            # 保存临近点
            nearby_points[i] = ref_vertices[indices[0]]


            ref_vertices = np.delete(ref_vertices, indices, axis=0)

        # 将临近点矩阵展平
        nearby_points = nearby_points.reshape(-1, 3)

        # print("Nearby points shape:", nearby_points.shape)
        # print("Remaining points shape:", ref_vertices.shape)
        return nearby_points,ref_vertices



    def get_sample_coords(self, coords, radius):
        # 获得batch size 和num_joints

        batch_size, num_joints, _ = coords.size()

        # 构建采样角度
        theta = torch.arange(0, 2 * math.pi, 2 * math.pi / 64).to(coords.device)

        # 扩展坐标张量的维度，以便计算偏移量
        coords = coords.unsqueeze(2).expand(batch_size, num_joints, 64, 2)

        # 计算偏移量并归一化到 [-1, 1]
        x_offset = radius * torch.cos(theta).view(1, 1, -1, 1).expand(batch_size, num_joints, -1, 1)
        y_offset = radius * torch.sin(theta).view(1, 1, -1, 1).expand(batch_size, num_joints, -1, 1)
        offsets = torch.cat([x_offset, y_offset], dim=3)
        sample_coords = (coords + offsets) / radius

        return sample_coords

    def load_cams(self, data_item, view_id):
        dat_fpath = os.path.join(
            self.dataset_dir,  data_item,'meta/cam_data.mat')
        try:
            cams_data = sio.loadmat(dat_fpath, verify_compressed_data_integrity=False)
        except ValueError as e:
            print('Value error occurred when loading ' + dat_fpath)
            raise ValueError(str(e))
        cams_data = cams_data['cam'][0]
        cam_param = cams_data[view_id]
        cam_R, cam_t = generate_cam_Rt(
            center=cam_param['center'][0, 0], right=cam_param['right'][0, 0],
            up=cam_param['up'][0, 0], direction=cam_param['direction'][0, 0])
        cam_R = cam_R.astype(np.float32)
        cam_t = cam_t.astype(np.float32)
        return cam_R, cam_t



    def load_smpl_parameters(self, data_item):
        dat_fpath = os.path.join(
           self.dataset_dir, data_item, data_item+'_smpl.pkl')
        if not os.path.exists(dat_fpath):
            dat_fpath = os.path.join(
                self.dataset_dir, data_item, data_item + '.pkl')
        if int(data_item)<526:
            with open(dat_fpath, 'rb') as fp:
                data = pkl.load(fp)
                pose = np.float32(data['body_pose']).reshape((-1))
                global_orient = np.float32(data['global_orient']).reshape((-1))
                pose = np.concatenate((global_orient, pose))
                betas = np.float32(data['betas']).reshape((-1,))
                trans = np.float32(data['transl']).reshape((1, -1))
                scale = np.float32(data['scale']).reshape((1, -1))
        else:
            with open(dat_fpath, 'rb') as fp:
                data = pkl.load(fp)

                pose = np.float32(data['pose']).reshape((-1))
                # global_orient = np.float32(data['root_pose']).reshape((-1))
                # pose = np.concatenate((global_orient, pose))
                betas = np.float32(data['betas']).reshape((-1,))
                trans = np.float32(data['trans']).reshape((1, -1))
                scale = np.float32(data['scale']).reshape((1, -1))
        return pose, betas, trans, scale

    def load_smpl_parameters2(self, data_item,model_id,view_id):
        dat_fpath = os.path.join(
           "/media/gpu/dataset_SSD/dataset/THuman2.0_smpl", data_item+'_smpl.pkl')

        with open(dat_fpath, 'rb') as fp:
            data = pkl.load(fp)
            pose = np.float32(data['body_pose']).reshape((-1))
            global_orient = np.float32(data['global_orient']).reshape((-1))
            pose = np.concatenate((global_orient, pose))
            betas = np.float32(data['betas']).reshape((-1,))
            trans = np.float32(data['transl']).reshape((1, -1))
            scale = np.float32(data['scale']).reshape((1, -1))
            gt_smpl_verts=self.smpl(torch.from_numpy(pose).unsqueeze(0),torch.from_numpy(betas).unsqueeze(0))
            gt_smpl_verts=gt_smpl_verts.squeeze().numpy()
            gt_smpl_verts=scale*gt_smpl_verts+trans

            calib_path = '/media/star/备份/GYQ/thuman2_CHON/thuman2_360views/%04s/calib/%04d.txt' % (
                data_item, view_id)
            # print(calib_path)
            calib_data = np.loadtxt(calib_path, dtype=float)

            extrinsic = calib_data[:4, :4]
            intrinsic = calib_data[4:8, :4]
            calib_mat = np.matmul(intrinsic, extrinsic)
            calib_verts = projection(gt_smpl_verts * 100, calib_mat)
            calib_verts[:, 1] *= -1
            label_path_vs = calib_verts
            # tmesh = trimesh.Trimesh(vertices=label_path_vs, faces=self.smpl_faces)
            # os.makedirs(r'/media/gpu/dataset_SSD/dataset/THuman2.0_Release_copy/0000/2/'  , exist_ok=True)
            # tmesh.export(r'/media/gpu/dataset_SSD/dataset/THuman2.0_Release_copy/0000/2/%04d_gtsmpl.obj' % (view_id))



        return gt_smpl_verts,label_path_vs,scale,trans,calib_mat

    def load_smpl_parameters3(self, data_item):
        dat_fpath = os.path.join(
           "/media/star/2023/dataset/THuman2.0_smpl", data_item+'_smpl.pkl')
        with open(dat_fpath, 'rb') as fp:
            data = pkl.load(fp)
            pose = np.float32(data['body_pose']).reshape(( -1))

            global_orient = np.float32(data['global_orient']).reshape((-1))

            pose = np.concatenate((global_orient, pose))

            #
            #
            # betas = np.float32(data['betas']).reshape((-1,))
            # trans = np.float32(data['transl']).reshape((1, -1))
            # scale = np.float32(data['scale']).reshape((1, -1))

            # theta_host = []
            # for r in pose:
            #     print(cv.Rodrigues(r))
            #     theta_host.append(cv.Rodrigues(r)[0])
            # theta_host = np.asarray(theta_host).reshape((1, -1)).squeeze()
            # var_dict['pose'] = theta_host


        return pose

    def rotate_points(self, pts, view_id):
        # rotate points to current view
        angle = 2 * np.pi * view_id / self.view_num_per_item
        pts_rot = np.zeros_like(pts)
        pts_rot[:, 0] = pts[:, 0] * math.cos(angle) - pts[:, 2] * math.sin(angle)
        pts_rot[:, 1] = pts[:, 1]
        pts_rot[:, 2] = pts[:, 0] * math.sin(angle) + pts[:, 2] * math.cos(angle)
        return pts_rot.astype(np.float32)

    def project_points(self, pts, cam_R, cam_t, cam_f):

        pts_proj = np.dot(pts, cam_R.transpose()) + cam_t
        # print(cam_R.transpose().shape)
        pts_proj[:, 0] = pts_proj[:, 0] * cam_f / pts_proj[:, 2] / (self.img_w / 2)
        pts_proj[:, 1] = pts_proj[:, 1] * cam_f / pts_proj[:, 2] / (self.img_h / 2)
        pts_proj = pts_proj[:, :2]
        return pts_proj.astype(np.float32)

    def update_smpl_params(self, pose, betas, trans, scale, view_id):
        # body shape and scale doesn't need to change
        betas_updated = np.copy(betas)
        scale_updated = np.copy(scale)

        # update body pose
        angle = 2 * np.pi * view_id / 360

        delta_r = cv.Rodrigues(np.array([0, angle, 0]))[0]
        root_rot = cv.Rodrigues(pose[:3])[0]
        root_rot_updated = np.matmul(delta_r, root_rot)
        pose_updated = np.copy(pose)
        pose_updated[:3] = np.squeeze(cv.Rodrigues(root_rot_updated)[0])

        # update body translation
        J = self.J_dirs.dot(betas) + self.J_template
        root = J[0]
        J_orig = np.expand_dims(root, axis=-1)
        J_new = np.dot(delta_r, np.expand_dims(root, axis=-1))
        J_orig, J_new = np.reshape(J_orig, (1, -1)), np.reshape(J_new, (1, -1))
        trans_updated = np.dot(delta_r, np.reshape(trans, (-1, 1)))
        trans_updated = np.reshape(trans_updated, (1, -1)) + (J_new - J_orig) * scale
        return np.float32(pose_updated), np.float32(betas_updated), \
               np.float32(trans_updated), np.float32(scale_updated)

    def load_hrnet_keypoints(self, data_item,view_id):
        # print(os.path.join(self.dataset_dir, data_item, 'keypoints/%04d.json' % view_id))
        with open(os.path.join(self.dataset_dir, data_item, 'keypoints/%04d.json' % view_id)) as fp:
            data = json.load(fp)
        keypoints = []
        kp_data = np.array(data['keypoints_coco'], dtype=np.float64)
        kp_data = kp_data.reshape([-1, 3])
        kp_data = kp_data[constant.coco_to_joint]  # rearrange keypoints

        kp_data[:, 0] = kp_data[:, 0] * 2 / self.img_w - 1.0
        kp_data[:, 1] = kp_data[:, 1] * 2 / self.img_h - 1.0
        kp_data[:, 2] = 1.0
        kp_data[constant.coco_to_joint < 0] *= 0.0  # remove undefined keypoints
        keypoints.append(kp_data)
        if len(keypoints) == 0:
            keypoints.append(np.zeros([24, 3]))

        return np.array(keypoints[0], dtype=np.float32)
    def draw_hrnet_keypoints(self, data_item,view_id):
        keypoints_path=os.path.join(self.dataset_dir, data_item, 'keypoints/%04d.json' % view_id)
        img_item=os.path.join(self.dataset_dir, data_item, 'render/%04d.png' % view_id)
        img=cv.imread(img_item)
        with open(keypoints_path) as fp:
            data = json.load(fp)
        keypoints = []

        kp_data = np.array(data['keypoints_coco'], dtype=np.float64)
        kp_data = kp_data.reshape([-1, 3])
        kp_data = kp_data[constant.coco_to_joint]  # rearrange keypoints
        kp_data[constant.coco_to_joint < 0] *= 0.0  # remove undefined keypoints
        for i in range(kp_data.shape[0]):
            # print(kp_data[i, 0],kp_data[i, 1])
            x = int(kp_data[i, 0])
            y = int(kp_data[i, 1])
            img = cv.putText(img, str(i), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        os.makedirs(os.path.join(self.dataset_dir, data_item,'render_keypoints'),exist_ok=True)
        # print(os.path.join(self.dataset_dir, data_item,'render_keypoints'))
        cv.imwrite(os.path.join(self.dataset_dir, data_item, 'render_keypoints/%04d.png' % view_id), img)



def worker_init_fn(worker_id):  # set numpy's random seed
    seed = torch.initial_seed()
    seed = seed % (2 ** 32)
    np.random.seed(seed + worker_id)


class TrainingImgLoader(DataLoader):
    def __init__(self, dataset_dir, img_h, img_w,
                 training=True, testing_res=512,
                 view_num_per_item=18, point_num=5000,
                 load_pts2smpl_idx_wgt=False, batch_size=4, num_workers=8):

        self.dataset = TrainingImgDataset(
            dataset_dir=dataset_dir, img_h=img_h, img_w=img_w,

            training=training, testing_res=testing_res,
            view_num_per_item=view_num_per_item, point_num=point_num,
            load_pts2smpl_idx_wgt=load_pts2smpl_idx_wgt)
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.training = training
        self.testing_res = testing_res
        self.dataset_dir = dataset_dir
        self.view_num_per_item = view_num_per_item
        self.point_num = point_num
        super(TrainingImgLoader, self).__init__(
            self.dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers,
            worker_init_fn=worker_init_fn, drop_last=True)

def get_sample_coords(coords, radius):
    # 获得batch size 和num_joints

    batch_size, num_joints, _ = coords.size()

    # 构建采样角度
    theta = torch.arange(0, 0.1*math.pi ,0.1* math.pi / 64).to(coords.device)

    # 扩展坐标张量的维度，以便计算偏移量
    coords = coords.unsqueeze(2).expand(batch_size, num_joints, 64, 2)

    # 计算偏移量并归一化到 [-1, 1]
    x_offset = radius * torch.cos(theta).view(1, 1, -1, 1).expand(batch_size, num_joints, -1, 1)
    y_offset = radius * torch.sin(theta).view(1, 1, -1, 1).expand(batch_size, num_joints, -1, 1)
    offsets = torch.cat([x_offset, y_offset], dim=3)
    sample_coords = (coords + offsets) / radius

    return sample_coords
def load_keypoints_3d(T_joints):
    from sklearn.neighbors import KDTree
    mask = constant.smplbody24_to_joint >= 0
    ref_vertices=np.array(obj_io.load_obj_data("/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/ref_vertices.obj")['v'])
    T_joints=np.array(obj_io.load_obj_data("/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/joints.obj")['v'])
    T_joints=T_joints[mask]

    # 构建KD树
    # kdtree = KDTree(ref_vertices)
    k=32

    # 存储所有的临近点
    nearby_points = np.zeros((17, k, 3))

    # 寻找每个关节点最近的k个点
    for i in range(17):
        kdtree = KDTree(ref_vertices)
        joint = T_joints[i]
        dists, indices = kdtree.query(joint.reshape(1, -1), k=k)

        # 保存临近点
        nearby_points[i] = ref_vertices[indices[0]]

        # 确保不重复
        for j in range(1, k):
            while np.isin(nearby_points[i, :j], nearby_points[i, j]).all():
                dist, index = kdtree.query(nearby_points[i, j].reshape(1, -1), k=2)
                if index[0][1] in indices:
                    indices[:,j] = index[0][1]
                    nearby_points[i, j] = ref_vertices[indices[:,j]]
                else:
                    break
        ref_vertices = np.delete(ref_vertices, indices, axis=0)


    # 将临近点矩阵展平
    nearby_points = nearby_points.reshape(-1, 3)


    print("Nearby points shape:", nearby_points.shape)
    print("Remaining points shape:", ref_vertices.shape)


    obj_io.save_obj_data({'v':nearby_points},"/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/near_ref_vertices.obj")
    obj_io.save_obj_data({'v': ref_vertices},
                         "/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/nonear_ref_vertices.obj")
    # print(result.shape)


if __name__ == '__main__':
    """tests data loader"""
    # load_keypoints_3d()

    from skimage import measure
    # import util.obj_io as obj_io

    loader = TrainingImgLoader('/media/star/2023/dataset/thuman2_CHON/thuman2_360views', 512, 512,

                               training=True, batch_size=1, num_workers=1)
    for i, items in enumerate(loader):
        print(i,items)
    #     mask = ~torch.isnan(items['keypoints'])
    #     data_2d_keypoints = items['keypoints'][mask].view(items['keypoints'].shape[0], -1, 3)
    #     data_2d_keypoints = data_2d_keypoints[..., :-1]
    #     print(data_2d_keypoints.shape)
    #
    # #     # ref_vertices_T=self.unsample_ref_vertices.expand(batch_size, -1, -1)
    # #     # features=self.resnet101(image)
    # #     # 0.1 xu yao tui qiao
    # #
    #     coords=gaussian_sampling_around_keypoints(data_2d_keypoints,0.025,32 )
    #     coords=coords.view(coords.shape[0],coords.shape[1]*coords.shape[2],coords.shape[3])
    #     coords=coords.squeeze()
    # #
    #     print(coords.shape)
    #     img_batch = items['img'].numpy()[0].transpose((1, 2, 0))
    #     keypoints2d_nearby_points=items['keypoints2d_nearby_points'].numpy()[0]
    #     # pts_batch = items['pts'].numpy()[0]
    #     # pts_proj_batch = items['pts_proj'].numpy()[0]
    #     # gt_batch = items['pts_ov'].numpy()[0]
    #     #
    #     cv.imwrite('/media/gpu/dataset_SSD/code/PaMIR-main/debug/img.jpg', np.uint8(img_batch[:, :, ::-1] * 255))
    #     img=cv.imread("/media/gpu/dataset_SSD/code/PaMIR-main/debug/img.jpg")/255
        #
        # gt_batch = gt_batch.reshape([-1])
        # inside_flag = gt_batch > 0
        # pts_proj_inside = pts_proj_batch[inside_flag]
        #
        # for p in pts_proj_inside:
        #     p = p * 256 + 256
        #     u, v = int(p[0]), int(p[1])
        #     img_batch[v, u] = np.array([0, 255, 0])
        # img=np.zeros((512,512,3),np.uint8)
        # print(img_batch.shape)shape
    #     n=0
    #     for p in keypoints2d_nearby_points:
    #         # n=n+1
    #         p = p * 256 + 256
    #         u, v = int(p[0]), int(p[1])
    #
    #         cv.circle(img,(u,v),2,(0.3,0.7,0.8),-1)
    #
    #         if u == 0:
    #             n = n + 1
    #             # break
    #
    #     print(n)
    #     cv.imwrite('/media/gpu/dataset_SSD/code/PaMIR-main/debug/img.jpg', np.uint8(img * 255))
    #
    #     # cv.imwrite('../debug/img.jpg', np.uint8(img_batch*255))
    #     # with open('../debug/pts.obj', 'w') as fp:
    #     #     for pt, pt_ov in zip(pts_batch, gt_batch):
    #     #         if pt_ov == 0:
    #     #             fp.write('v %f %f %f 1 0 0\n' % (pt[0], pt[1], pt[2]*2))
    #     #         if pt_ov > 0:
    #     #             fp.write('v %f %f %f 0 0 1\n' % (pt[0], pt[1], pt[2] * 2))
    #     # break
    #
    #     # pts_ov = np.reshape(gt_batch, (512, 512, 512))
    #     # vertices, simplices, normals, _ = measure.marching_cubes_lewiner(pts_ov, 0.5)
        # mesh = dict()
        # mesh['v'] = vertices
        # mesh['f'] = simplices
        # mesh['f'] = mesh['f'][:, (1, 0, 2)]
        # mesh['vn'] = normals
        # obj_io.save_obj_data(mesh, os.path.join('./debug', 'output_%04d.obj' % i))
        break
