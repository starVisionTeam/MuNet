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
import numpy as np
import scipy.spatial
import scipy.io as sio
import pickle as pkl
import json
import cv2 as cv
import torch
import trimesh
from torch.utils.data import Dataset, DataLoader
import networks.constant
# import open3d as o3d

def load_data_list(dataset_root, list_txt_fname):
    with open(os.path.join(dataset_root, list_txt_fname), 'r') as fp:
        lines = fp.readlines()

    lines = [l.strip(' \r\n') for l in lines]

    return lines[506:526]

def point_to_surface_distance(point_cloud, mesh):
    kd_tree = o3d.geometry.KDTreeFlann(mesh)
    distances = []
    closest_faces = []  # 保存每个点最近的面的索引
    processed_faces = set()  # 记录已经处理过的面的索引集合
    for point in point_cloud.points:
        [k, idx, dis] = kd_tree.search_knn_vector_3d(point, 1)  # 找到最近的一个点和对应的面
        print(k, idx[0], dis[0])
        distances.append(dis[0])
        # closest_faces.append(idx)
        if idx[0] not in processed_faces:
            closest_faces.append(idx[0])
            processed_faces.add(idx[0])
    return distances, closest_faces


def generate_cam_Rt(center, direction, right, up):
    def normalize_vector(v):
        v_norm = np.linalg.norm(v)
        return v if v_norm == 0 else v / v_norm

    center = center.reshape([-1])
    direction = direction.reshape([-1])
    right = right.reshape([-1])
    up = up.reshape([-1])

    rot_mat = np.eye(3)
    s = right
    s = normalize_vector(s)
    rot_mat[0, :] = s
    u = up
    u = normalize_vector(u)
    rot_mat[1, :] = -u
    rot_mat[2, :] = normalize_vector(direction)
    trans = -np.dot(rot_mat, center)
    return rot_mat, trans

def rotate_vertex(vertices,angle):
    angle=np.radians(angle)
    rotation_matrix=np.array([
        [np.cos(angle),0,np.sin(angle)],
        [0,1,0],
        [-np.sin(angle),0,np.cos(angle)]
    ])
    vertices_array=np.array(vertices)
    rotation_vertices=np.dot(vertices_array,rotation_matrix.T)
    rotation_vertices=np.float32(rotation_vertices)


    return rotation_vertices

def rotate_vertex_with_smooth(vertices,angle,faces):
    angle=np.radians(angle)
    rotation_matrix=np.array([
        [np.cos(angle),0,np.sin(angle)],
        [0,1,0],
        [-np.sin(angle),0,np.cos(angle)]
    ])
    vertices_array=np.array(vertices)
    rotation_vertices=np.dot(vertices_array,rotation_matrix.T)
    rotation_vertices=np.float32(rotation_vertices)
    # print(rotation_vertices.shape,faces.shape)
    mesh = trimesh.Trimesh(rotation_vertices,faces)
    mesh2_vert=trimesh.smoothing.filter_laplacian(mesh, lamb=0.2,iterations=5).vertices

    return np.array(mesh2_vert)

def rotate_vertex_tensor(vertices, angles):
    """
    将 vertices 中的点分别绕Y轴旋转指定角度。

    参数:
    - vertices: 形状为 (batch_size, num_points, 3) 的 PyTorch 张量，表示待旋转的点坐标。
    - angles: 形状为 (batch_size,) 的 PyTorch 张量，表示每个数据点的旋转角度。

    返回:
    - 形状为 (batch_size, num_points, 3) 的 PyTorch 张量，表示旋转后的点坐标。
    """
    angles = torch.deg2rad(angles)
    print(angles)
    rotation_matrices = torch.stack([
        torch.stack([
            torch.cos(angles), torch.zeros_like(angles), torch.sin(angles)
        ], dim=1),
        torch.stack([
            torch.zeros_like(angles) + 1, torch.zeros_like(angles) + 1, torch.zeros_like(angles) + 1
        ], dim=1),
        torch.stack([
            -torch.sin(angles), torch.zeros_like(angles), torch.cos(angles)
        ], dim=1)
    ], dim=2)
    print(torch.einsum('ijk,ikl->ijl', vertices, rotation_matrices))
    return torch.einsum('ijk,ikl->ijl', vertices, rotation_matrices)