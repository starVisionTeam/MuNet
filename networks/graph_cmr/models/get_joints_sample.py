import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
from torch.nn.functional import pairwise_distance
from scipy.spatial import distance_matrix
# import cupy as cp
from scipy.spatial.distance import cdist
class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)

    def forward(self, x):
        x = self.resnet101.conv1(x)
        x = self.resnet101.bn1(x)
        x = self.resnet101.relu(x)
        x = self.resnet101.maxpool(x)

        x = self.resnet101.layer1(x)
        x = self.resnet101.layer2(x)
        x = self.resnet101.layer3(x)
        x = self.resnet101.layer4(x)

        return x

def get_sample_coords(coords):
    # Create a meshgrid with coordinates of size 4x4
    grid = torch.meshgrid(torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4))
    # Reshape to (1, 1, 4, 4)
    grid = torch.stack(grid).unsqueeze(0).unsqueeze(0)

    # Add the original coordinates to each grid point
    coords = coords[:, :, None, None, :] + grid

    # Flatten the last two dimensions
    coords = coords.view(coords.size(0), coords.size(1), -1, 2)

    return coords

def get_sample_features_resnet101(features, sample_coords):
    # Normalize sample_coords to [-1, 1]
    batch_size,num_joints,near,_=sample_coords.shape
    sample_coords = sample_coords.clone()
    sample_coords[:, :, :, 0] /= features.size(3) - 1
    sample_coords[:, :, :, 1] /= features.size(2) - 1
    sample_coords = sample_coords * 2 - 1

    # Reshape features to (batch_size, feature_dim, H*W)
    features = features.view(features.size(0), features.size(1), -1)
    features=features.unsqueeze(2)

    # Reshape sample_coords to (batch_size, num_joints, H*W, 2)
    sample_coords = sample_coords.view(sample_coords.size(0), -1, sample_coords.size(2), sample_coords.size(3))
    # print(sample_coords .shape)
    # sample_coords=sample_coords.unsqueeze(1)

    # Reshape sample_coords to (batch_size*num_joints*H*W, 2)
    # sample_coords = sample_coords.view(-1, 2048,17,16,2)

    # Sample features using bilinear interpolation
    sample_features = nn.functional.grid_sample(features, sample_coords, mode='bilinear', align_corners=True)

    # Reshape sample_features to (batch_size, num_joints, H*W, feature_dim)
    sample_features = sample_features.view(features.size(0), -1, sample_coords.size(1), features.size(1))


    # Transpose sample_features to (batch_size, num_joints, feature_dim, H*W)
    sample_features = sample_features.transpose(1, 2)
    sample_features=sample_features.reshape(batch_size,sample_features.size(1)*sample_features.size(2),sample_features.size(3))

    return sample_features


def get_nearest_64_points(batch_images, batch_keypoints):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = batch_images.shape[0]
    num_keypoints = batch_keypoints.shape[1]
    num_points = 64

    batch_keypoints = batch_keypoints.view(batch_size * num_keypoints, -1)

    # Construct the KD-tree from the flattened image
    flattened_images = batch_images.view(batch_size, -1, 3).cpu().numpy()
    kd_tree = KDTree(flattened_images.reshape(-1,3))

    # Query the KD-tree for the nearest 64 points
    distances, indices = kd_tree.query(batch_keypoints.cpu().numpy(), k=num_points)

    # Convert the indices to coordinates
    indices = indices.reshape(batch_size, num_keypoints, num_points)
    flattened_indices = indices + np.arange(0, batch_size * num_keypoints * num_points, num_points).reshape(
        batch_size * num_keypoints, 1)
    flattened_images = flattened_images.reshape(batch_size * num_keypoints, -1, 3)
    nearest_points = flattened_images[flattened_indices].reshape(batch_size, num_keypoints, num_points, 3)

    # Normalize the coordinates to (-1, 1)
    nearest_points = torch.tensor(nearest_points, dtype=torch.float32).to(device)
    nearest_points = nearest_points / 127.5 - 1

    # Return the normalized coordinates
    return nearest_points.permute(0, 2, 1, 3)
def gaussian_sampling_around_keypoints(keypoints, sigma, num_points):
    """
    在关键点周围使用高斯采样采样num_points个点，标准差为sigma
    :param keypoints: 关键点，大小为(batch_size, num_keypoints, 2)的张量，范围是(-1, 1)
    :param sigma: 高斯分布的标准差
    :param num_points: 采样点数
    :return: 采样点，大小为(batch_size, num_keypoints, num_points, 2)的张量，范围是(-1, 1)
    """
    batch_size, num_keypoints = keypoints.shape[:2]
    # 生成高斯分布的采样点
    gaussian_samples = np.random.normal(loc=0, scale=sigma, size=(batch_size, num_keypoints, num_points, 2))
    gaussian_samples = torch.tensor(gaussian_samples, dtype=torch.float32, device=keypoints.device)

    # 将采样点缩放并平移至关键点
    samples = gaussian_samples + keypoints.view(batch_size, num_keypoints, 1, 2)

    # 将采样点限制在(-1, 1)范围内
    samples = torch.clamp(samples, -1, 1)

    return samples
def get_sample_features_resnet(feature_map, points):
    """
    在特征图中找到指定点的特征

    Args:
    feature_map: (batch_size, 2048) 的特征图
    points: (batch_size, 272, 2) 的点坐标，范围为 [-1, 1]

    Returns:
    (batch_size, 272, 2048) 的特征向量
    """

    # 将二维特征图转换为四维特征图
    device=feature_map.device
    feature_map = feature_map.unsqueeze(-1).unsqueeze(-1)

    # 将点坐标从 [-1, 1] 转换为像素坐标
    image_size = feature_map.shape[-2:]
    print(image_size)
    points = (points + 1) / 2 * torch.tensor(image_size).float().to(device)
    print(points.max(),points.min())

    # 使用 PyTorch 库的 grid_sample 函数找到指定点的特征
    features = torch.nn.functional.grid_sample(feature_map, points)

    # 将特征大小调整为 (batch_size, 272, 2048)
    features = features.squeeze(-1).squeeze(-1)
    features=features.reshape(features.size(0),features.size(1),features.size(2)*features.size(3))
    features=features.transpose(1,2)

    return features


def get_keypoints_features(features, keypoints, image_size):
    batch_size, num_keypoints, _ = keypoints.size()

    # 将归一化的坐标转化为像素坐标
    keypoints = (keypoints + 1) / 2 * torch.FloatTensor(image_size).to(keypoints.device)

    # 将像素坐标转化为对应的feature map坐标
    grid = (keypoints.view(batch_size, num_keypoints, 1, 2) / (image_size[0] - 1) * 2 - 1).flip(-1)

    # 使用grid_sample查询每个关节点对应的特征向量
    features = features.view(batch_size, 2048, 1, 1)
    keypoints_features = F.grid_sample(features, grid, align_corners=True)

    # 将查询到的特征向量从(batch_size, 2048, 1, 1)的形式变为(batch_size, 1008, 2048)的形式
    keypoints_features = keypoints_features.view(batch_size, num_keypoints, -1)

    return keypoints_features

def get_nearest_neighbors(template, points, k=16, device='cuda'):
    """
    使用 KD Tree 查询每个点周围的 K 个邻近点，并返回邻近点的坐标值
    :param template: 人体模板张量，大小为 (batch_size, num_points, 3)
    :param points: 三维关节点张量矩阵，大小为 (batch_size, num_joints, 3)
    :param k: 邻近点的数量
    :param device: 设备，可以是 'cpu' 或 'cuda'
    :return: 大小为 (batch_size, num_joints, k, 3) 的邻近点张量矩阵
    """
    batch_size, num_points, _ = template.size()
    batch_size, num_keypoints, _ = points.size()
    # print(batch_size)

    # 将人体模板张量转换为大小为 (batch_size * num_points, 3) 的二维张量
    template_flat = template.reshape(batch_size * num_points, 3)

    # 构建 KD Tree
    tree = KDTree(template_flat.cpu().numpy())

    # 将三维关节点张量矩阵展平为大小为 (batch_size * num_joints, 3) 的二维张量
    points_flat = points.view(batch_size * num_keypoints, 3)

    # 查询邻近点的索引
    dists, indices = tree.query(points_flat.cpu().numpy(), k=k)

    # 将邻近点的索引转换为坐标值
    # indices = indices.reshape(batch_size, num_keypoints, k)
    # print(indices.shape)
    # print(template_flat.shape)
    mask=torch.ones((batch_size * num_points,3),dtype=torch.bool)
    # mask=mask.unsqueeze(-1)
    mask[indices]=False

    nearest_no_neighbors = template_flat[mask].reshape(batch_size, -1,3)
    nearest_neighbors = template_flat[indices].reshape(batch_size, num_keypoints, k, 3)
    nearest_neighbors=nearest_neighbors.reshape(batch_size,num_keypoints*k,3)

    return nearest_neighbors.to(device),nearest_no_neighbors


def filter_joints(joints,joints3d):
    # 筛选出不为 (-1.0, -1.0, 0.0) 的关节点
    mask = joints != torch.tensor([-1.0, -1.0, 0.0], device=joints.device)

    # 统计每个样本中满足条件的关节点数量
    num_joints = mask.sum(dim=1)

    # 获取最大的关节点数量，即 num
    num = int(num_joints.max())

    # 将不满足条件的坐标设置为 NaN
    joints[~mask] = float("nan")
    joints3d[~mask] = float("nan")

    # 创建一个形状为 (batch_size, num, 3) 的张量，将所有满足条件的坐标放入其中
    filtered_joints = torch.empty(joints.size(0), num, 3, device=joints.device)
    filtered_joints3d = torch.empty(joints3d.size(0), num, 3, device=joints.device)
    for i in range(joints.size(0)):
        filtered_joints[i] = torch.where(
            mask[i],
            joints[i],
            torch.tensor([float("nan"), float("nan"), float("nan")], device=joints.device)
        )
        filtered_joints3d[i] = torch.where(
            mask[i],
            joints3d[i],
            torch.tensor([float("nan"), float("nan"), float("nan")], device=joints.device)
        )

    return filtered_joints,filtered_joints3d
# def remove_same_values(input1, input2):
#     # 获取第二个顶点矩阵的每个顶点在第一个顶点矩阵中的位置
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     input1_reshape = input1.reshape(-1, 3)
#     input2_reshape = input2.reshape(-1, 3)
#     same_indices = torch.eq(input1_reshape.unsqueeze(1), input2_reshape.unsqueeze(0)).all(-1).any(-1)
#
#     # 将同样的点在第一个矩阵中去除
#     mask = torch.logical_not(same_indices).to(device)
#
#     output = torch.masked_select(input1_reshape, mask.unsqueeze(-1)).reshape(input1.shape[0], -1, 3)
#
#     print(output.shape)


    # return output
def remove_same_values(mat1,mat2):
    batch_size = 2
    n1, n2 = 1088, 1723
    d = 3

    # 生成数据
    mat1 = torch.randn(batch_size, n1, d).cuda()  # 第一个顶点矩阵
    mat2 = torch.randn(batch_size, n2, d).cuda()  # 第二个顶点矩阵

    # 计算欧几里得距离
    dist = torch.cdist(mat1, mat2)  # shape: (batch_size, n1, n2)

    # 找到距离最近的点
    min_dist, min_indices = torch.min(dist, dim=2)  # shape: (batch_size, n1)
    print(min_indices.shape)

    # 构造掩码矩阵，去除距离最近的点
    mask = torch.ones(batch_size, n2).cuda().scatter(1, min_indices, 0)  # shape: (batch_size, n2)
    print(mask.shape)

    # 去除距离最近的点
    new_mat2 = torch.masked_select(mat2, mask.unsqueeze(-1).bool())
    new_mat2=new_mat2.reshape(mat2.shape[0],-1,3)
    return new_mat2

def replace_vertex_features_with_joints(image_enc, joint2d_features, ref_v, data_3d_keypoint):
    """
    Replace joint features in image_enc with those in joint2d_features.
    If a joint is not found in joint2d_features, replace it with the nearest joint.
    """
    # Calculate distance between each vertex and each joint
    batch_size, num_verts, _ = ref_v.size()
    _, num_joints, _ = data_3d_keypoint.size()
    ref_v = ref_v.unsqueeze(2).repeat(1, 1, num_joints, 1)
    data_3d_keypoint = data_3d_keypoint.unsqueeze(1).repeat(1, num_verts, 1, 1)

    dist = ((ref_v - data_3d_keypoint) ** 2).sum(dim=-1)  # shape: (batch_size, num_verts, num_joints)



    # Find nearest joint for each vertex
    _, nearest_joint_idxs = torch.min(dist, dim=-1)  # shape: (batch_size, num_verts)

    # image_enc_copy=image_enc.clone
    joint2d_features2=joint2d_features.clone()
    image_enc_copy=image_enc.clone()

    # Replace joint features
    for b in range(batch_size):
        for j in range(num_joints):

            joint_idx = torch.nonzero(nearest_joint_idxs[b] == j)
            if len(joint_idx) > 0:
                joint_idx = joint_idx.squeeze()
                image_enc_copy[b, joint_idx, :] = joint2d_features2[b, j, :]
                # print(joint_idx,j)

    return image_enc_copy