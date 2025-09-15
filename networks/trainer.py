"""
This file includes the full training procedure.
"""
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from skimage import measure
import scipy.io as sio
import datetime
import glob
import logging
import math
import torchvision
from networks.util.base_trainer import BaseTrainer
from networks.dataloader.dataloader import TrainingImgDataset
from networks.network.arch import PamirNet
from networks.neural_voxelization_layer.smpl_model import TetraSMPL
from networks.neural_voxelization_layer.voxelize import Voxelization
from networks.util.img_normalization import ImgNormalizerForResnet
from networks.graph_cmr.models import GraphCNN, SMPL
from networks.graph_cmr.utils.mesh_6890 import Mesh
from networks.graph_cmr.models.geometric_layers import rodrigues, orthographic_projection
import networks.util.obj_io as obj_io
import networks.util.util as util
import networks.constant as const
from models.layers.mesh import Mesh2,PartMesh
from models.networks import orthogonal, Index, populate_e, rotate_points
from networks.dataloader.util import is_mesh_file, load_obj, manifold_upsample, get_num_parts, compute_normal
from models import create_model
# from  networks.models.train_options import TrainOptions
from networks.models.networks import *
from networks.dataloader.utils import load_data_list, generate_cam_Rt,rotate_vertex,rotate_vertex_with_smooth
from networks.dataloader.util import print_network, sample_surface, local_nonuniform_penalty, cal_edge_loss2,projection_train,projection
from networks.models.loss import chamfer_distance1, BeamGapLoss
from networks.dataloader.util import print_network, sample_surface, closest_points_to_ori_on_src,projection
# from smpl2smplx.transfer.transfer_model_smpl2smplx import run_fitting_smpl2smplx
# from networks.smpl2smplx.transfer.config import parse_args_smpl2smplx
# from networks.smpl2smplx.transfer.utils import read_deformation_transfer
#
# from smplx2smpl.transfer.transfer_model_smplx2smpl import run_fitting_smplx2smpl
# from networks.smplx2smpl.transfer.config import parse_args_smplx2smpl
# from networks.smplx2smpl.transfer.utils import read_deformation_transfer
#
# from smplx import build_layer
from evaluator_6890_and_40000 import rigid_align
from models.networks import orthogonal, Index, populate_e, rotate_points
from util.render import Render
class Trainer(BaseTrainer):
    def __init__(self, options):
        super(Trainer, self).__init__(options)

    def init_fn(self):
        super(BaseTrainer, self).__init__()
        # dataset
        self.train_ds = TrainingImgDataset(
            self.options.dataset_dir, img_h=const.img_res, img_w=const.img_res,
            training=True, testing_res=256,
            view_num_per_item=40,
            point_num=self.options.point_num,
            load_pts2smpl_idx_wgt=True,
            smpl_data_folder='/media/gpu/dataset_SSD/code/PaMIR-main/networks/data')

        # GraphCMR components
        self.img_norm = ImgNormalizerForResnet().to(self.device)
        self.graph_mesh = Mesh()
        # trimesh.load_mesh()

        self.graph_cnn = GraphCNN(self.graph_mesh.adjmat, self.graph_mesh.ref_vertices.t(),
                                  const.cmr_num_layers, const.cmr_num_channels).to(self.device)


        # neural voxelization components
        self.smpl = SMPL('/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(self.device)

        self.tet_smpl = TetraSMPL('/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
                                  '/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/tetra_smpl.npz').to(self.device)
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = util.read_smpl_constants('/media/gpu/dataset_SSD/code/PaMIR-main/networks/data')
        self.smpl_faces = smpl_faces

        # self.is_train = self.options.is_train
        self.overlap = self.options.overlap
        self.net = init_net(self.device, self.options)
        #
        self.num_samples = 50000


        #
        self.optimizer = torch.optim.Adam(
            params=list(self.graph_cnn.parameters())+list(self.net.parameters()),
            lr=self.options.lr,
            betas=(self.options.adam_beta1, 0.999),
            weight_decay=self.options.wd)
        self.scheduler = get_scheduler(self.optimizer, self.options)
        # print_network(self.net)
        self.FOCAL_LENGTH = 5000.

        # Create loss functions
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_geo = nn.MSELoss().to(self.device)

        # Pack models and optimizers in a dict - necessary for checkpointing,'recon':self.net
        self.models_dict = {'graph_cnn': self.graph_cnn,'recon':self.net}
        self.optimizers_dict = {'optimizer': self.optimizer}

        assert self.options.pretrained_gcmr_checkpoint is not None, 'You must provide a pretrained GNN model!'
        self.load_pretrained_gcmr(self.options.pretrained_gcmr_checkpoint)

        self.load_reconstrution_network(self.options.net_checkpoint)


        # # read energy weights
        # self.loss_weights = {
        #     'smpl':self.options.weight_smpl,
        #     'gnn':self.options.weight_gnn,
        #     'geo': self.options.weight_geo,
        # }
        self.render = Render(size=512, device=self.device)

        # meta results
        now = datetime.datetime.now()
        self.log_file_path = os.path.join(
            self.options.log_dir, 'log_%s.npz' % now.strftime('%Y_%m_%d_%H_%M_%S'))




    def train_step(self, input_batch):

        self.graph_cnn.train()
        self.net.train()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        model_id=input_batch["model_id"]
        view_id = input_batch["view_id"]
        # openpose_keypoint=input_batch['keypoints']
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        gt_scale = input_batch['scale']
        gt_trans = input_batch['trans']
        feature_local=input_batch["feature_local"]
        feature_global = input_batch["feature_global"]
        calib=input_batch["calib"]
        label_path_vss = input_batch['label_path_vs']
        label_path_norms =input_batch['label_path_norm']
        calib_data=input_batch["calib_data"]
        # label_path_faces=input_batch["label_path_faces"]
        # gt_dis=input_batch["gt_dis"]
        # gt_vertices_cam_meshcnn=input_batch["gt_vertices_cam"]
        # gt_hrnet_keypoint=input_batch["hrnet_keypoint"]

        # print(input_batch['img_dir'])
        diff_S_gt=input_batch["diff_S_gt"]
        scan2d_gt= input_batch["scan2d_gt"]
        smpl2d_gt= input_batch["smpl2d_gt"]




        losses = dict()

        # prepare gt variables
        gt_vertices = self.smpl(gt_pose, gt_betas)
        # gt_keypoints_3d = self.smpl.get_joints(gt_vertices)
        batch_size = gt_vertices.shape[0]


        # gt_vertices_cam=gt_scale * gt_vertices + gt_trans
        # os.makedirs('/media/gpu/dataset_SSD/train_result2/%04d' % model_id[0],
        #             exist_ok=True)
        # obj_io.save_obj_data({'v': gt_vertices[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
        #                      "/media/gpu/dataset_SSD/train_result2/%s/%04d_gtsmpl.obj" % (model_id[0], view_id[0]))
        # obj_io.save_obj_data({'v': gt_vertices_cam_meshcnn[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
        #                      "/media/gpu/dataset_SSD/train_result2/%s/%04d_gtsmpl2.obj"%(model_id[0],view_id[0]))
        # gcmr body prediction
        img_ = self.img_norm(img)
        # cv.imwrite("/media/gpu/dataset_SSD/code/PaMIR-main/debug/tttt.png",img_.squeeze().permute(1,2,0).detach().cpu().numpy()*255)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert = pred_vert_sub.transpose(1, 2)
        scale_, trans_ = self.forward_coordinate_conversion(pred_vert, cam_f, cam_tz, cam_c, pred_cam, None)

        # pred_keypoints_2d = orthographic_projection(self.smpl.get_joints(pred_vert), pred_cam)[:, :, :2]
        pred_vert_meshcnn2 = scale_ * pred_vert + trans_
        pred_keypoints_3d_meshcnn = self.smpl.get_joints(pred_vert_meshcnn2)
        pred_keypoints_2d = self.forward_point_sample_projection(
            pred_keypoints_3d_meshcnn, cam_r, cam_t, cam_f, cam_c)
        pred_vert_meshcnn = rotate_vertex_with_smooth(pred_vert_meshcnn2[0:1].squeeze().detach().cpu(), -view_id[0].cpu(),faces=self.smpl_faces)
        pred_vert_meshcnn = torch.from_numpy(pred_vert_meshcnn).to(self.device).float()
        pred_vert_meshcnn = projection(pred_vert_meshcnn * 100, calib_data[0:1].squeeze())
        pred_vert_meshcnn[:, 1] *= -1




        gt_vert_meshcnn2 = gt_scale * gt_vertices + gt_trans
        gt_keypoints_3d_meshcnn = self.smpl.get_joints(gt_vert_meshcnn2)
        gt_keypoints_2d = self.forward_point_sample_projection(
            gt_keypoints_3d_meshcnn, cam_r, cam_t, cam_f, cam_c)
        gt_vert_meshcnn = rotate_vertex(gt_vert_meshcnn2[0:1].squeeze().detach().cpu(), -view_id[0].cpu())
        gt_vert_meshcnn = torch.from_numpy(gt_vert_meshcnn).to(self.device).float()
        gt_vert_meshcnn = projection(gt_vert_meshcnn * 100, calib_data[0:1].squeeze())
        gt_vert_meshcnn[:, 1] *= -1

        loss_chamfer_meshcnn, _ = chamfer_distance1(pred_vert_meshcnn2, gt_vert_meshcnn2, unoriented=False)
        losses['train'] = self.criterion_keypoints(trans_, gt_trans).mean()
        losses['gnn_keypoints_3d_cam'] = self.criterion_keypoints(pred_keypoints_3d_meshcnn,
                                                                          gt_keypoints_3d_meshcnn).mean()
        losses['gnn_shape'] = self.shape_loss(pred_vert_meshcnn2, gt_vert_meshcnn2)
        losses["gnn_loss_chamfer"] = loss_chamfer_meshcnn
        losses["gnn_keypoints_2d"] = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d)




        pred_vert_meshcnn = rigid_align(pred_vert_meshcnn.detach().cpu().numpy(),
                                        gt_vert_meshcnn.detach().cpu().numpy())
        pred_vert_meshcnn = torch.from_numpy(pred_vert_meshcnn).unsqueeze(0).to(self.device).float()
        gt_vert_meshcnn =gt_vert_meshcnn.unsqueeze(0).to(self.device).float()
        loss_chamfer_meshcnn, _ = chamfer_distance1(pred_vert_meshcnn, gt_vert_meshcnn, unoriented=False)

        losses['gnn_shape_meshcnn'] = self.shape_loss(pred_vert_meshcnn, gt_vert_meshcnn)
        losses["gnn_loss_chamfer_meshcnn"] = loss_chamfer_meshcnn

        print(losses['gnn_shape'],losses['gnn_shape_meshcnn'])
        #
        # obj_io.save_obj_data({'v': pred_vert_meshcnn2[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
        #                      "/media/gpu/dataset_SSD/train_result2/%04d/%04d_meshcnn_pred.obj" % (
        #                      model_id[0], view_id[0]))
        # obj_io.save_obj_data({'v': gt_vert_meshcnn2[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
        #                      "/media/gpu/dataset_SSD/train_result2/%04d/%04d_meshcnn_gt.obj" % (
        #                      model_id[0], view_id[0]))
        #
      #   obj_io.save_obj_data({'v': pred_vert_meshcnn[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
      #                                              "/media/gpu/dataset_SSD/train_result2/%04d/%04d_meshcnn_pred2.obj" % (model_id[0], view_id[0]))
      #   obj_io.save_obj_data({'v': gt_vert_meshcnn[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
      #                      "/media/gpu/dataset_SSD/train_result2/%04d/%04d_meshcnn_gt2.obj" % (model_id[0],view_id[0]))
      #
      #
      #
      #   #obj_io.save_obj_data({'v': label_path_vs[0].squeeze().detach().cpu().numpy()},
      #   #                  "/media/gpu/dataset_SSD/train_result2/%04s/%04d_meshcnn_gt_mesh.obj" % (model_id[0],view_id[0]))
      #
      #   # mesh_predict = Mesh2( file=None , device=self.device, vs=pred_vert_meshcnn, faces=self.smpl_faces,hold_history=True)
        mesh_predict = Mesh2(file=None, device=self.device, vs=pred_vert_meshcnn, faces=self.smpl_faces,

                        hold_history=True)
        losses['gnn_edge_loss'] = 10 * cal_edge_loss2(mesh_predict, self.device)
        vs = pred_vert_meshcnn.permute(0, 2, 1)
        xyz = orthogonal(vs, calib, transforms=None)
        xy = xyz[:, :2, :]  # B, 2, N
      #   #
      #
      #
      #
        vs_features_fine = Index(feature_local, xy)  # B, C, N[1,16,6890]
        vs_features_fine = vs_features_fine.permute(0, 2, 1)  # B, N, C [1,6890,16]
        x_fine = vs_features_fine[:, mesh_predict.edges, :]  # [1,20664,2,16]
        edge_feature_fine = x_fine.view(1, mesh_predict.edges_count, -1).permute(0, 2, 1).to(
            self.device)  # B, C, N[1,32,20664]

        vs_features_global = Index(feature_global, xy)  # B, C, N
        vs_features_global = vs_features_global.permute(0, 2, 1)  # B, N, C
        x_global = vs_features_global[:, mesh_predict.edges, :]
        edge_feature_global = x_global.view(1, mesh_predict.edges_count, -1).permute(0, 2, 1).to(
            self.device)  # B, C, N[1,512,20664]

        smpl_norm = compute_normal(vs.permute(0, 2, 1).squeeze().detach().cpu().numpy(), self.smpl_faces)
        smpl_norm = torch.tensor(smpl_norm[None]).to(self.device).float()  # [1,6890,3]

        vs_features = smpl_norm
        x = vs_features[:, mesh_predict.edges, :]
        edge_feature = x.view(1, mesh_predict.edges_count, -1).permute(0, 2, 1).to(self.device)  # B, C, N[1,6,20664]

        rand_verts = populate_e([mesh_predict]).to(self.device)
        edge_features = torch.cat([edge_feature, edge_feature_global, edge_feature_fine, rand_verts], dim=1)




        est_verts =self.net(edge_features, [mesh_predict])

        mesh_predict.update_verts(est_verts)
        recon_xyz, recon_normals = sample_surface(mesh_predict.faces.to(self.device),
                                                  mesh_predict.vs.to(self.device), self.num_samples)
        # closest_points_to_smpl=closest_points_to_ori_on_src(recon_xyz.squeeze(),pred_vert_meshcnn.squeeze())
        # closest_points_to_smpl=closest_points_to_smpl.unsqueeze(0)
        recon_xyz, recon_normals = recon_xyz.to('cuda').type(torch.float), recon_normals.to('cuda').type(torch.float)
        xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance1(recon_xyz, label_path_vss,
                                                                   x_normals=recon_normals, y_normals=label_path_norms,
                                                                   unoriented=False)


        xyz_chamfer_loss2, _ = chamfer_distance1(mesh_predict.vs.to('cuda'), label_path_vss)

        # cs2s_chamfer_loss, _ = chamfer_distance1(closest_points_to_smpl, gt_vert_meshcnn)
        gt_keypoints_3d_meshcnn = self.smpl.get_joints(gt_vert_meshcnn)

        # obj_io.save_obj_data({'v': recon_xyz[0].squeeze().detach().cpu().numpy()},
        #                      "/media/gpu/dataset_SSD/train_result2/%04d/%04d_recxyz.obj" % ( model_id[0],view_id[0]))
        # obj_io.save_obj_data({'v': closest_points_to_smpl[0].squeeze().detach().cpu().numpy()},
        #                      "/media/gpu/dataset_SSD/train_result2/%04d/%04d_closest_points_to_smpl.obj" % (model_id[0],view_id[0]))

        # trimesh.Trimesh(label_path_vss[0].squeeze().detach().cpu().numpy(),label_path_faces[0].squeeze().detach().cpu().numpy()).export(
        #                     "/media/gpu/dataset_SSD/train_result2/%04d/%04d_mesh_gt.obj" % (model_id[0], view_id[0]))

        # mesh_predict.export("/media/gpu/dataset_SSD/train_result2/%04d/%04d_mesh.obj" % (model_id[0],view_id[0]))

        # recon_keypoint_3d=self.smpl.get_joints(closest_points_to_smpl)
        # loss_recon_keypoint_3d = self.keypoint_3d_loss(recon_keypoint_3d, gt_keypoints_3d_meshcnn)
        # pred_dis=self.criterion_shape(pred_vert_meshcnn, mesh_predict.vs)
        # losses['distance'] = self.criterion_shape(pred_dis,gt_dis)

        T_mask_F, T_mask_E, T_mask_B, T_mask_W = self.render_normal(
            pred_vert_meshcnn,
            torch.from_numpy(self.smpl_faces).unsqueeze(0).to(self.device)
        )
        mask_F, mask_E, mask_B, mask_W = self.render_normal(
            mesh_predict.vs,
            torch.from_numpy(self.smpl_faces).unsqueeze(0).to(self.device)

        )

        smpl_arr = torch.cat([T_mask_F, T_mask_E, T_mask_B, T_mask_W], dim=-1)
        scan_arr = torch.cat([mask_F, mask_E, mask_B, mask_W], dim=-1)
        diff_S_pred = torch.abs(smpl_arr - scan_arr)
        diff_S =torch.abs(diff_S_pred - diff_S_gt)
        diff_S_smpl = torch.abs(smpl_arr - smpl2d_gt)
        diff_S_scan = torch.abs(scan_arr - scan2d_gt)

       # torchvision.utils.save_image(
            #    ((diff_S.detach().cpu() + 1.0) * 0.5),
            #    "/media/gpu/dataset_SSD/train_result2/%04d/%04d_diff_S.png" % (model_id[0], view_id[0]))
        #torchvision.utils.save_image(
         #   ((diff_S_smpl.detach().cpu() + 1.0) * 0.5),
         #   "/media/gpu/dataset_SSD/train_result2/%04d/%04d_diff_S_smpl.png" % (model_id[0], view_id[0]))
      #  torchvision.utils.save_image(
          #  ((diff_S_scan.detach().cpu() + 1.0) * 0.5),
         #   "/media/gpu/dataset_SSD/train_result2/%04d/%04d_diff_S_scan.png" % (model_id[0], view_id[0]))


        # design the losses
        losses['xyz_chamfer_loss'] = xyz_chamfer_loss
        losses['silhouette_gt_pred'] = diff_S.mean()
        losses['diff_S_smpl'] = diff_S_smpl.mean()
        losses[' diff_S_scan'] =  diff_S_scan.mean()
        losses["xyz_chamfer_loss2"]=xyz_chamfer_loss2
        losses['normals_chamfer_loss']=normals_chamfer_loss*0.1
        losses['edge_loss'] =  10*cal_edge_loss2(mesh_predict,self.device)

        print(xyz_chamfer_loss,losses['silhouette_gt_pred'],normals_chamfer_loss*0.1,losses['edge_loss'])
        #


        total_loss = 0
        # print(total_loss)
        for ln in losses.keys():
            w = 1.0
            total_loss += w * losses[ln]
        #
        losses.update({'total_loss': total_loss})

        # Do backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # save
        self.write_logs(losses)

        # update learning rate
        if self.step_count % 10000 == 0:
            learning_rate = self.options.lr * (0.9 ** (self.step_count//10000))
            logging.info('Epoch %d, LR = %f' % (self.step_count, learning_rate))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        return losses
    def render_normal( self,verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])

        self.render.load_meshes(verts, faces)
        return self.render.get_image(cam_type="four",type="mask")
    def perspective_projection(self,points, rotation, translation,
                               focal_length, camera_center, retain_z=False):
        """
        This function computes the perspective projection of a set of points.
        Input:
            points (bs, N, 3): 3D points
            rotation (bs, 3, 3): Camera rotation
            translation (bs, 3): Camera translation
            focal_length (bs,) or scalar: Focal length
            camera_center (bs, 2): Camera center
        """
        batch_size = points.shape[0]
        K = torch.zeros([batch_size, 3, 3], device=points.device)
        K[:, 0, 0] = focal_length
        K[:, 1, 1] = focal_length
        K[:, 2, 2] = 1.
        K[:, :-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij,bkj->bki', rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:, :, -1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

        if retain_z:
            return projected_points
        else:
            return projected_points[:, :, :-1]

    def project_points(self, pts, cam_R, cam_t):

        pts_proj=torch.einsum('bij,bjk->bik',pts,cam_R) + cam_t.unsqueeze(1)


        cam_f=5000
        pts_proj[:,:, 0] = pts_proj[:,:, 0] * cam_f / pts_proj[:,:, 2] / (512 / 2)
        pts_proj[:,:, 1] = pts_proj[:,:, 1] * cam_f / pts_proj[:,:, 2] / (512 / 2)
        pts_proj = pts_proj[:,:, :2]
        return pts_proj
    def compute_acc(self,pred, gt, thresh=0.5):
        """
        input
            res         : (1, 1, n_in + n_out), res[0] are estimated occupancy probs for the query points
            label_tensor: (1, 1, n_in + n_out), float 1.0-inside, 0.0-outside

        return
            IOU, precision, and recall
        """

        # compute {IOU, precision, recall} based on the current query 3D points
        with torch.no_grad():


            vol_pred = pred > thresh
            vol_gt = gt > thresh

            union = vol_pred | vol_gt
            inter = vol_pred & vol_gt

            true_pos = inter.sum().float()

            union    = union.sum().float()
            if union == 0:
                union = 1

            vol_pred = vol_pred.sum().float()
            if vol_pred == 0:
                vol_pred = 1

            vol_gt = vol_gt.sum().float()
            if vol_gt == 0:
                vol_gt = 1

        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt
    def calculate_gt_rotmat(self, gt_pose):
        gt_rotmat = rodrigues(gt_pose.view((-1, 3)))
        gt_rotmat = gt_rotmat.view((self.options.batch_size, -1, 3, 3))
        gt_rotmat[:, 0, 1:3, :] = gt_rotmat[:, 0, 1:3, :] * -1.0  # global rotation
        return gt_rotmat

    def forward_gcmr(self, img):
        # GraphCMR forward
        batch_size = img.size()[0]
        img_ = self.img_norm(img)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert_sub = pred_vert_sub.transpose(1, 2)
        pred_vert = self.graph_mesh.upsample(pred_vert_sub)
        x = torch.cat(
            [pred_vert_sub, self.graph_mesh.ref_vertices[None, :, :].expand(batch_size, -1, -1)],
            dim=-1)
        pred_rotmat, pred_betas = self.smpl_param_regressor(x)
        pred_vert_tetsmpl = self.tet_smpl(pred_rotmat, pred_betas)
        pred_keypoints = self.smpl.get_joints(pred_vert)
        pred_keypoints_2d = orthographic_projection(pred_keypoints, pred_cam)
        return pred_cam, pred_rotmat, pred_betas, pred_vert_sub, \
               pred_vert, pred_vert_tetsmpl, pred_keypoints_2d
    def forward_coordinate_conversion2(self, cam_f, cam_tz, cam_c, pred_cam):
        # calculates camera parameters
        with torch.no_grad():
            scale = pred_cam[:, 0:1] * cam_c * cam_tz / cam_f
            trans_x = pred_cam[:, 1:2] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
            trans_y = -pred_cam[:, 2:3] * cam_c * cam_tz * pred_cam[:, 0:1] / cam_f
            trans_z = torch.zeros_like(trans_x)
            scale_ = torch.cat([scale, -scale,scale], dim=-1).detach().view((-1, 1, 3))
            trans_ = torch.cat([trans_x, -trans_y, trans_z], dim=-1).detach().view((-1, 1, 3))

        return scale_, trans_
    def forward_coordinate_conversion(self, pred_vert_tetsmpl, cam_f, cam_tz, cam_c,pred_cam, gt_trans):
        # calculates camera parameters
        # with torch.no_grad():
        pred_smpl_joints = self.smpl.get_joints(pred_vert_tetsmpl).detach()
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

    def forward_warp_gt_field(self, pred_vert_tetsmpl_gtshape_cam, gt_vert_cam,
                              pts, pts2smpl_idx, pts2smpl_wgt):
        with torch.no_grad():
            trans_gt2pred = pred_vert_tetsmpl_gtshape_cam - gt_vert_cam

            trans_z_pt_list = []
            for bi in range(self.options.batch_size):
                trans_pt_bi = (
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 0], 2] * pts2smpl_wgt[bi, :, 0] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 1], 2] * pts2smpl_wgt[bi, :, 1] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 2], 2] * pts2smpl_wgt[bi, :, 2] +
                        trans_gt2pred[bi, pts2smpl_idx[bi, :, 3], 2] * pts2smpl_wgt[bi, :, 3]
                )
                trans_z_pt_list.append(trans_pt_bi.unsqueeze(0))
            trans_z_pts = torch.cat(trans_z_pt_list, dim=0)
            # translate along z-axis to resolve depth inconsistency
            # pts[:, :, 2] += trans_z_pts
            pts[:, :, 2] += torch.tanh(trans_z_pts * 20) / 20
        return pts

    def forward_calculate_smpl_sub_vertice_in_cam(self, pred_vert_cam, cam_r, cam_t, cam_f, cam_c):
        pred_vert_sub_cam = self.graph_mesh.downsample(pred_vert_cam)
        pred_vert_sub_proj = pred_vert_sub_cam * cam_r.view((1, 1, -1)) + cam_t.view(
            (1, 1, -1))
        pred_vert_sub_proj = \
            pred_vert_sub_proj * (cam_f / cam_c) / pred_vert_sub_proj[:, :, 2:3]
        pred_vert_sub_proj = pred_vert_sub_proj[:, :, :2]
        return pred_vert_sub_cam, pred_vert_sub_proj
    def forward_point_sample_projection(self, points, cam_r, cam_t, cam_f, cam_c):
        points_proj = points * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
        points_proj = points_proj * (cam_f / cam_c) / points_proj[:, :, 2:3]
        points_proj = points_proj[:, :, :2]
        return points_proj
    def geo_loss(self, pred_ov, gt_ov):
        """Computes per-sample loss of the occupancy value"""
        if self.options.use_multistage_loss:
            loss = 0
            for o in pred_ov:
                loss += self.criterion_geo(o, gt_ov)
        else:
            loss = self.criterion_geo(pred_ov[-1], gt_ov)
        return loss

    def train_summaries(self, input_batch, losses=None):
        assert losses is not None
        for ln in losses.keys():
            self.summary_writer.add_scalar(ln, losses[ln].item(), self.step_count)

    def write_logs(self, losses):
        data = dict()
        if os.path.exists(self.log_file_path):
            data = dict(np.load(self.log_file_path))
            for k in losses.keys():
                data[k] = np.append(data[k], losses[k].item())
        else:
            for k in losses.keys():
                data[k] = np.array([losses[k].item()])
        np.savez(self.log_file_path, **data)

    def load_pretrained_gcmr(self, model_path):
        if os.path.isdir(model_path):
            tmp = glob.glob(os.path.join(model_path, 'gcmr*.pt'))
            assert len(tmp) == 1
            logging.info('Loading REG from ' + tmp[0])
            data = torch.load(tmp[0])
        else:
            data = torch.load(model_path)
            logging.info('Loading GCMR from ' + model_path)
        self.graph_cnn.load_state_dict(data['graph_cnn'])
        # self.smpl_param_regressor.load_state_dict(data['smpl_param_regressor'])
        # self.net.load_state_dict(data["recon"])

    def load_pretrained_pamir_net(self, model_path):
        if os.path.isdir(model_path):
            tmp1 = glob.glob(os.path.join(model_path, 'pamir_net*.pt'))
            assert len(tmp1) == 1
            logging.info('Loading pamir_net from ' + tmp1[0])
            data = torch.load(tmp1[0])
        else:
            data = torch.load(model_path)
            logging.info('Loading pamir_net from ' + model_path)
        if 'pamir_net' in data:
            self.pamir_net.load_state_dict(data['pamir_net'])
        else:
            raise IOError('Failed to load pamir_net model from the specified checkpoint!!')

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        # conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = torch.mean(( pred_keypoints_2d -  gt_keypoints_2d) ** 2)
        return loss
    def keypoint_loss2(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = torch.mean((conf * pred_keypoints_2d - conf * gt_keypoints_2d[:, :, :-1]) ** 2)
        # loss = conf*( self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss
    def shape_loss(self, pred_vertices, gt_vertices):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices
        gt_vertices_with_shape = gt_vertices
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)
    def keypoint_loss_2(self, pred_keypoints_2d, gt_keypoints_2d):
        """Compute 2D reprojection loss on the keypoints.
        The confidence is binary and indicates whether the keypoints exist or not.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss
    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence
        """
        # return ( self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        # # gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        # # conf = conf[has_pose_3d == 1]
        # pred_keypoints_3d = pred_keypoints_3d[:, :, :-1].clone()
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return ( self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)
    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        """Compute SMPL parameter loss for the examples that SMPL annotations are available."""
        pred_rotmat_valid = pred_rotmat.view(-1, 3, 3)
        gt_rotmat_valid = rodrigues(gt_pose.view(-1, 3))
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
    def load_reconstrution_network(self, model_path):
        """load model from disk"""
        # save_filename = '%s_net.pth' % which_epoch
        # self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        # load_path = os.path.join(self.save_dir, save_filename)
        # net = self.net
        # print('loading for net ...', model_path)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
     #   data = torch.load(model_path)
        #self.net.load_state_dict(data['recon'])

        # save+img
        # img = cv.imread(input_batch['img_dir'][0])
        # # img=np.zeros(img2.shape)
        # points = xy * 256
        # points[:, 0:1, :] = points[:, 0:1, :] + 256
        # points[:, 1:2, :] = points[:, 1:2, :] + 256
        # points = points.squeeze().detach().cpu().permute(1, 0).numpy()
        #
        # for point in (points):
        #     img = cv.circle(img, tuple([int(point[0]), int(point[1])]), 1, (255, 0, 0), -1)


        # vs = pred_vert_meshcnn.permute(0, 2, 1)
        # xyz = orthogonal(vs, calib, transforms=None)
        # xy = xyz[:, :2, :]  # B, 2, N
        #
        # # save+img
        # # img=cv.imread(input_batch['img_dir'][0])
        # # img=np.zeros(img2.shape)
        # points = xy * 256
        # points[:, 0:1, :] = points[:, 0:1, :] + 256
        # points[:, 1:2, :] = points[:, 1:2, :] + 256
        # points = points.squeeze().detach().cpu().permute(1, 0).numpy()
        #
        # for point in (points):
        #     img = cv.circle(img, tuple([int(point[0]), int(point[1])]), 1, (0, 0, 255), -1)
        # # cv2.imshow()

        # cv.imwrite('/media/gpu/dataset_SSD/code/PaMIR-main/debug/debug_meshcnn2/test_%04d.png'%(view_id[0]), img)

        # img = cv.imread(input_batch['img_dir'][0])
        # conf = gt_hrnet_keypoint[:, :, -1].unsqueeze(-1).clone()
        # kp_detection_points2 = (conf * gt_keypoints_2d[:, :, :-1]).squeeze().detach().cpu().numpy()
        # for idx, point in enumerate(kp_detection_points2):
        #     x = int((point[0] + 1) * 256)
        #     y = int((point[1] + 1) * 256)
        #     # img2 = cv.circle(img2, tuple([x, y]), 3, (0, 255, 255), -1)
        #     img = cv.putText(img, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        #
        # kp_detection_points2 = (conf * pred_keypoints_2d).squeeze().detach().cpu().numpy()
        # for idx, point in enumerate(kp_detection_points2):
        #     x = int((point[0] + 1) * 256)
        #     y = int((point[1] + 1) * 256)
        #     # img2 = cv.circle(img2, tuple([x, y]), 3, (0, 255, 255), -1)
        #     img3 = cv.putText(img, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
        # cv.imwrite('/media/gpu/dataset_SSD/code/PaMIR-main/debug/test2.png', img)

        # img = cv.imread(input_batch['img_dir'][0])
        # conf = gt_hrnet_keypoint[:, :, -1].unsqueeze(-1).clone()
        # kp_detection_points2 = (gt_keypoints_2d).squeeze().detach().cpu().numpy()
        # for idx, point in enumerate(kp_detection_points2):
        #     x = int((point[0] + 1) * 256)
        #     y = int((point[1] + 1) * 256)
        #     # img2 = cv.circle(img2, tuple([x, y]), 3, (0, 255, 255), -1)
        #     img = cv.putText(img, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        #
        # kp_detection_points2 = (conf * pred_keypoints_2d).squeeze().detach().cpu().numpy()
        # for idx, point in enumerate(kp_detection_points2):
        #     x = int((point[0] + 1) * 256)
        #     y = int((point[1] + 1) * 256)
        #     # img2 = cv.circle(img2, tuple([x, y]), 3, (0, 255, 255), -1)
        #     img = cv.putText(img, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
        # cv.imwrite('/media/gpu/dataset_SSD/code/PaMIR-main/debug/test2.png', img)