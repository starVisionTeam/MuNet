from __future__ import division, print_function

import glob
import os

import torch
import pickle as pkl
from tqdm import tqdm
import torch.nn as nn
from util import util
from util import obj_io
import constant as const
import cv2 as cv
from networks.neural_voxelization_layer.smpl_model import TetraSMPL
import numpy as np
import scipy.io as sio
from networks.graph_cmr.models.geometric_layers import rodrigues, orthographic_projection
from networks.graph_cmr.models import GraphCNN, SMPL
import sys
import time

def smpl_losses( pred_rotmat, pred_betas, gt_pose, gt_betas):
    """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
    pred_rotmat_valid = pred_rotmat.view(-1, 3, 3)
    gt_rotmat_valid = rodrigues(gt_pose.view(-1,3))
    print(gt_rotmat_valid.shape)
    pred_betas_valid = pred_betas
    gt_betas_valid = gt_betas
    device = torch.device("cuda")
    criterion_regr = nn.MSELoss().to(device)
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose = criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = criterion_regr(pred_betas_valid, gt_betas_valid)

    return loss_regr_pose, loss_regr_betas
def flatten(a):
    for each in a:
        if not isinstance(each, list):
            yield each
        else:
            yield from flatten(each)
def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2
def main_test_for_CHON(test_img_dir, out_dir, pretrained_checkpoint, pretrained_gnn_checkpoint,pretrained_reg_checkpoint,
                                   iternum=50,batch_size=1, num_workers=8):
    from networks.evaluator import Evaluator
    from networks.dataloader.dataloader import TrainingImgLoader
    from networks.util.pose_utils import reconstruction_error
    device = torch.device("cuda")
    smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
        util.read_smpl_constants('./data')
    smpl = SMPL('./data/basicModel_f_lbs_10_207_0_v1.0.0.pkl').to(device)
    tet_smpl = TetraSMPL(
        './data/basicModel_f_lbs_10_207_0_v1.0.0.pkl',
        './data/tetra_smpl.npz').to(device)
    os.makedirs(out_dir, exist_ok=True)
    loader = TrainingImgLoader(
        test_img_dir, img_h=const.img_res, img_w=const.img_res,
            training=False, testing_res=512,
            view_num_per_item=360,
            point_num=5000,
            load_pts2smpl_idx_wgt=False, batch_size=batch_size, num_workers=num_workers)

    evaluator = Evaluator(device, pretrained_checkpoint, pretrained_gnn_checkpoint,pretrained_reg_checkpoint)
    cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
    cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(device)
    cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(device)
    

    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        curr_batch_size = batch['img'].shape[0]
        #gt
        gt_smpl_v = batch['scale'] * smpl(batch['pose'], batch['betas'])+batch["trans"]



        pred_cam,pred_betas, pred_rotmat, scale, trans, pred_vert, pred_tetsmpl = evaluator.test_gcmr(batch['img'])

        scale, tranX, tranY = pred_cam.split(1, dim=1)

        scale = scale.unsqueeze(1).float()
        trans = (
            torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                      dim=1).unsqueeze(1).to(device).float()
        )





        img_dir = batch['img_dir'][0]
        img_fname = os.path.split(img_dir)[1]
        #
        img_folder=img_dir.split("/")[-3]
        os.makedirs(os.path.join(out_dir, img_folder,'smpl_results'), exist_ok=True)
        smpl_param_name=os.path.join(out_dir, img_folder, 'smpl_results',img_fname[:-4] + '_smpl.pkl')

        with open(smpl_param_name, 'wb') as fp:
            pkl.dump({'shape': batch["betas"].squeeze().detach().cpu().numpy(),
                      'pose': batch['pose'].squeeze().detach().cpu().numpy(),
                      'scale':scale,
                      'trans':trans,
                      'scale_to_mesh':batch['scale'].squeeze().detach().cpu().numpy(),
                       'trans_to_mesh':batch['trans'].squeeze().detach().cpu().numpy()},
                     fp)
        smpl_param_name = os.path.join(out_dir, img_folder, 'smpl_results', img_fname[:-4] + '_smpl.obj')

        obj_io.save_obj_data({'v': gt_smpl_v.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             smpl_param_name)







def main_test_meshcnn(test_img_dir, out_dir,  pretrained_gnn_checkpoint,
                                   iternum=50):
    from networks.dataloader.dataloader_test import TrainingImgLoader
    from networks.evaluator import Evaluator
    from networks.util.pose_utils import reconstruction_error
    device = torch.device("cuda")
    smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
        util.read_smpl_constants('./data')
    smpl = SMPL('./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(device)
    tet_smpl = TetraSMPL(
        './data/basicModel_f_lbs_10_207_0_v1.0.0.pkl',
        './data/tetra_smpl.npz').to(device)
    os.makedirs(out_dir, exist_ok=True)


    loader = TrainingImgLoader(
        test_img_dir, img_h=const.img_res, img_w=const.img_res,
            training=False, testing_res=512,
            view_num_per_item=6,
            point_num=5000,
            load_pts2smpl_idx_wgt=False, batch_size=1, num_workers=1)

    evaluator = Evaluator(device, pretrained_gnn_checkpoint)
    cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
    cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(device)
    cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(device)
    mpjpe =[]
    recon_err =[]
    shape_err =[]


    xyz_chamfer_losses = []
    normals_chamfer_losses = []

    for step, batch in enumerate(tqdm(loader, desc='Testing', total=len(loader), initial=0)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        curr_batch_size = batch['img'].shape[0]
        #gt
        # scale=torch.from_numpy(np.float32([[1, -1, -1]]).reshape((1, -1))).unsqueeze(0).to(device)

        gt_smpl_v = smpl(batch['pose'], batch['betas'])
        gt_keypoints_3d = smpl.get_joints(gt_smpl_v)
        gt_keypoints_2d = forward_point_sample_projection(
            gt_keypoints_3d, cam_r, cam_t, cam_f, cam_c)
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        #
        #
        pred_cam, scale_, trans_,pred_vert = evaluator.test_gcmr_6890(batch['img'])
        pred_vert_cam=scale_*pred_vert+trans_
     
        pred_vert2 = rigid_align((pred_vert_cam).squeeze().cpu().detach().numpy(),
                                 (gt_smpl_v).squeeze().cpu().detach().numpy())
        pred_vert2 = torch.from_numpy(pred_vert2).unsqueeze(0).to(device)
        # # # #
        pred_keypoints_3d = smpl.get_joints(pred_vert2)
        pred_keypoints_2d = forward_point_sample_projection(
            pred_keypoints_3d, cam_r, cam_t, cam_f, cam_c)
        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        # # # # #
        # # # # #
        # # # # # Absolute error (MPJPE)
        error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().detach().tolist()
        mpjpe.append(error)
        #
        # # Reconstuction_error
        r_error = reconstruction_error(pred_keypoints_3d.cpu().detach().numpy(), gt_keypoints_3d.cpu().detach().numpy(),
                                       reduction=None)
        recon_err.append(r_error)
        #
        se = torch.sqrt(((pred_vert2- gt_smpl_v) ** 2).sum(dim=-1)).mean(dim=-1).cpu().detach().numpy()
        shape_err.append(se)
        # # #
        #
        # #
       
        img_dir = batch['img_dir'][0]
        img_fname = os.path.split(img_dir)[1]
        img_folder = img_dir.split("/")[-3]
        
        os.makedirs(os.path.join(out_dir, img_folder,'ours_results'), exist_ok=True)
        mesh_fname = os.path.join(out_dir, img_folder, 'ours_results', img_fname[:-4] + '.obj')
        init_smpl_fname = os.path.join(out_dir, img_folder, img_fname[:-4] + '_pred_smpl.obj')
        optm_smpl_fname = os.path.join(out_dir, img_folder, 'ours_results',img_fname[:-4] + '_optm_smpl.obj')
        
        gt_smpl_fname = os.path.join(out_dir,img_folder, img_fname[:-4] + '_gt_smpl.obj')

        obj_io.save_obj_data({'v': (gt_smpl_v).squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             gt_smpl_fname)
        obj_io.save_obj_data({'v': (pred_vert2).squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             init_smpl_fname)

        
        obj_io.save_obj_data({'v': gt_smpl_v.squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             gt_smpl_fname)
        obj_io.save_obj_data({'v': (pred_vert2).squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             optm_smpl_fname)
        obj_io.save_obj_data({'v': (pred_vert).squeeze().detach().cpu().numpy(), 'f': smpl_faces},
                             gnn_vert_fname)

        # evaluator.test_meshcnn_thuman(batch, out_dir)
        evaluator.test_all_step(batch,out_dir,input_dir=test_img_dir)

    print('*** Final Results ***')
    print('*** Orignal Results ***')
    print('MPJPE: ' + str(1000 * np.mean(list(flatten(mpjpe)))))
    print('Reconstruction Error: ' + str(1000 * np.mean(list(flatten(recon_err)))))
    print('Shape Error: ' + str(1000 * np.mean(list(flatten(shape_err)))))
    print()



   



def shape_loss( pred_vertices, gt_vertices):
    """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
    pred_vertices_with_shape = pred_vertices
    gt_vertices_with_shape = gt_vertices
    device = torch.device("cuda")
    criterion_shape = nn.L1Loss().to(device)
    if len(gt_vertices_with_shape) > 0:
        return criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)
def keypoint_3d_loss( pred_keypoints_3d, gt_keypoints_3d):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence
    """
    device = torch.device("cuda")
    criterion_keypoints = nn.MSELoss(reduction='none').to(device)
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        return (criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
# import sys
# import time
# log_path = '/media/gpu/dataset_SSD/code/PaMIR-main/networks/Logs/'
# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# # 日志文件名按照程序运行时间设置
# log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
# f = open(log_file_name, 'w')
# sys.stdout = f
def project_points( pts, cam_R, cam_t):

    pts_proj=torch.einsum('bij,bjk->bik',pts,cam_R) + cam_t.unsqueeze(1)


    cam_f=5000
    pts_proj[:,:, 0] = pts_proj[:,:, 0] * cam_f / pts_proj[:,:, 2] / (512 / 2)
    pts_proj[:,:, 1] = pts_proj[:,:, 1] * cam_f / pts_proj[:,:, 2] / (512 / 2)
    pts_proj = pts_proj[:,:, :2]
    return pts_proj
def forward_point_sample_projection( points, cam_r, cam_t, cam_f, cam_c):
    points_proj = points * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
    points_proj = points_proj * (cam_f / cam_c) / points_proj[:, :, 2:3]
    points_proj = points_proj[:, :, :2]
    return points_proj

def orthographic_projection(X, camera):
    """Perform orthographic projection of 3D points X using the camera parameters
    Args:
        X: size = [B, N, 3]
        camera: size = [B, 3]
    Returns:
        Projected 2D points -- size = [B, N, 2]
    """
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    return X_2d
def forward_coordinate_conversion( pred_vert_tetsmpl, cam_f, cam_tz, cam_c,
                                  cam_r, cam_t, pred_cam, gt_trans):
    # calculates camera parameters
    device = torch.device("cuda")
    tet_smpl = TetraSMPL(
        '/media/star/GYQ-KESU/code/code/PaMIR-main/networks/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
        '/media/star/GYQ-KESU/code/code/PaMIR-main/networks/data/tetra_smpl.npz').to(device)
    with torch.no_grad():
        pred_smpl_joints = tet_smpl.get_smpl_joints(pred_vert_tetsmpl).detach()
        pred_root = pred_smpl_joints[:, 0:1, :]
        if gt_trans is not None:
            scale = pred_cam[:, 0:1] * cam_c * (cam_tz - gt_trans[:, 0, 2:3]) / cam_f
            trans_x = pred_cam[:, 1:2] * cam_c * (
                    cam_tz - gt_trans[:, 0, 2:3]) * pred_cam[:, 0:1] / cam_f
            trans_y = -pred_cam[:, 2:3] * cam_c * (
                    cam_tz - gt_trans[:, 0, 2:3]) * pred_cam[:, 0:1] / cam_f
            trans_z = gt_trans[:, 0, 2:3] + 2 * pred_root[:, 0, 2:3] * scale
        else:
            scale = pred_cam[:, 0:1] * cam_c * cam_tz / cam_f
            trans_x = pred_cam[:, 1:2] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
            trans_y = -pred_cam[:, 2:3] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
            trans_z = torch.zeros_like(trans_x)
        scale_ = torch.cat([scale, -scale, -scale], dim=-1).detach().view((-1, 1, 3))
        trans_ = torch.cat([trans_x, trans_y, trans_z], dim=-1).detach().view((-1, 1, 3))

    return scale_, trans_


if __name__ == '__main__':
    iternum=20
    input_image_dir = '/media/star/备份/GYQ/thuman2_CHON/thuman2_360views'
    output_dir = '/media/star/备份/GYQ/thuman2_CHON/result'
      
    path,filename=os.path.split(dis)
    output_dir=""
    a,b,c,d,e=main_test_meshcnn(input_image_dir,output_dir,pretrained_gnn_checkpoint="")
    
   