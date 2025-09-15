"""
This file contains the Definition of GraphCNN
GraphCNN includes ResNet50 as a submodule
"""
from __future__ import division

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .graph_layers import GraphResBlock, GraphLinear
from .resnet import resnet50
from .smpl import  SMPL
import torchvision
import torch.nn.functional as F
from .get_joints_sample import remove_same_values,get_keypoints_features,get_nearest_neighbors,replace_vertex_features_with_joints,gaussian_sampling_around_keypoints
# from networks.graph_cmr.utils.mesh import Mesh
import math
import networks.util.obj_io as obj_io
class GraphCNN(nn.Module):
    
    def __init__(self, A, ref_vertices, num_layers=5, num_channels=512):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.resnet = resnet50(pretrained=True)
        layers = [GraphLinear(3 + 2048, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, A),
                                   GraphResBlock(64, 32, A),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, 3))
        self.gc = nn.Sequential(*layers)
        self.camera_fc = nn.Sequential(nn.GroupNorm(num_channels // 8, num_channels),
                                      nn.ReLU(inplace=True),
                                      GraphLinear(num_channels, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(A.shape[0], 3))
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.smpl=SMPL('/media/star/dataset_SSD/code/Cloth-shift/PaMIR-main/networks/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(self.device)
        # self.unsample_ref_vertices=unsample_ref_vertices[None,]
        # self.T_joints= self.smpl.get_joints(unsample_ref_vertices[None,])
        # self.resnet101_model = torchvision.models.resnet101(pretrained=True)
        # self.resnet101_model.eval()
        # self.resnet101=ResNet101()


    def forward(self, image,keypoints2d_nearby_points,near_keypoints_3d,no_near_keypoints_3d):
        """Forward pass
        Inputs:
            image: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = image.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        # print("ref_vertices:",ref_vertices.shape)#torch.Size([1, 3, 1723])
        image_resnet = self.resnet(image)
        image_enc = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, ref_vertices.shape[-1])
        image_enc=image_enc.transpose(1,2)
        ref_vertices=ref_vertices.transpose(1,2)

        # mask=~torch.isnan(keypoints_2d)
        # data_2d_keypoints=keypoints_2d[mask].view(keypoints_2d.shape[0],-1,3)
        # data_2d_keypoints=data_2d_keypoints[...,:-1]
        # coords=gaussian_sampling_around_keypoints(data_2d_keypoints,0.025, 32)
        # coords=coords.reshape(coords.shape[0],coords.shape[1]*coords.shape[2],coords.shape[3])
        # joints2d_features=get_sample_features_resnet(image_resnet,coords)
        joints2d_features=get_keypoints_features(image_resnet, keypoints2d_nearby_points.float(), (224, 224))
        # image_enc2=replace_vertex_features_with_joints(image_enc,joints2d_features,ref_vertices,near_keypoints_3d)
        # print(image_enc2.shape,near_keypoints_3d.shape,joints2d_features.shape)
        x1=torch.cat([near_keypoints_3d.transpose(1,2), joints2d_features.transpose(1,2)], dim=1)
        image_enc2 = image_resnet.view(batch_size, 2048, 1).expand(-1, -1, no_near_keypoints_3d.shape[-2])
        x2=torch.cat([no_near_keypoints_3d.transpose(1,2), image_enc2], dim=1)
        x = torch.cat([x1.transpose(1,2), x2.transpose(1,2)], dim=1).transpose(1,2)
        # print("x:",x.shape) #torch.Size([1, 2051, 1723])
        x = self.gc(x.float()) #torch.Size([1, 256, 1723])
        # print("gc(c):",x.shape)
        shape = self.shape(x) #torch.Size([1, 3, 1723])

        camera = self.camera_fc(x).view(batch_size, 3)
        return shape, camera

    # def get_sample_coords(self,coords, radius):
    #     # 获得batch size 和num_joints
    #
    #     batch_size, num_joints, _ = coords.size()
    #
    #     # 构建采样角度
    #     theta = torch.arange(0, 2 * math.pi, 2 * math.pi / 64).to(coords.device)
    #
    #     # 扩展坐标张量的维度，以便计算偏移量
    #     coords = coords.unsqueeze(2).expand(batch_size, num_joints, 64, 2)
    #
    #     # 计算偏移量并归一化到 [-1, 1]
    #     x_offset = radius * torch.cos(theta).view(1, 1, -1, 1).expand(batch_size, num_joints, -1, 1)
    #     y_offset = radius * torch.sin(theta).view(1, 1, -1, 1).expand(batch_size, num_joints, -1, 1)
    #     offsets = torch.cat([x_offset, y_offset], dim=3)
    #     sample_coords = (coords + offsets) / radius
    #
    #     return sample_coords


