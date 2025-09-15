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
from skimage import measure
from tqdm import tqdm
import scipy.io as sio
import datetime
import glob
import logging
import math
import torchvision
from networks.network.arch import PamirNet
from networks.neural_voxelization_layer.smpl_model import TetraSMPL
# from networks.neural_voxelization_layer.voxelize import Voxelization
from networks.util.img_normalization import ImgNormalizerForResnet
from networks.graph_cmr.models import GraphCNN, SMPL
from networks.graph_cmr.utils.mesh_6890 import Mesh
from networks.graph_cmr.models.geometric_layers import rodrigues, orthographic_projection
import networks.util.util as util
import networks.util.obj_io as obj_io
import constant as const
from util.render import Render
import copy
from networks.dataloader.utils import load_data_list, generate_cam_Rt,rotate_vertex,rotate_vertex_with_smooth
from networks.dataloader.util import print_network, sample_surface, local_nonuniform_penalty, cal_edge_loss,projection_train,projection
from models.layers.mesh import Mesh2
import trimesh
from networks.dataloader.util import is_mesh_file, load_obj, manifold_upsample, get_num_parts, compute_normal
from networks.models.networks import *
from networks.util.train_options import TrainOptions
from networks.models.loss import chamfer_distance1
from eval_meshcnn_thuman2 import rigid_align
# from smpl2smplx.transfer.__main__ import smpl2smplx
# from smplx2smpl.transfer.__main__ import smplx2smpl
from cloth_refine.infer_me import refine_cloth
from replace_hand.replace_hand import replace_hands
class Evaluator(object):
    def __init__(self, device,  pretrained_gnn_checkpoint):
        super(Evaluator, self).__init__()
        util.configure_logging(True, False, None)

        self.device = device

        # GraphCMR components
        self.img_norm = ImgNormalizerForResnet().to(self.device)
        self.graph_mesh = Mesh()
        # trimesh.load_mesh()

        self.graph_cnn = GraphCNN(self.graph_mesh.adjmat, self.graph_mesh.ref_vertices.t(),
                                  const.cmr_num_layers, const.cmr_num_channels).to(self.device)

        # neural voxelization components
        self.smpl = SMPL(
            '/media/gpu/dataset_SSD/code/PaMIR-main/networks/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl').to(
            self.device)

        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('/media/gpu/dataset_SSD/code/PaMIR-main/networks/data')

        # print("smpl_vertex_code:", smpl_vertex_code.shape)
        self.smpl_faces = smpl_faces
        self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)


        # self.graph_cnn.eval()
        self.render = Render(size=512, device=self.device)
        self.options = TrainOptions().parse_args()
        self.net = init_net(self.device, self.options)

        self.num_samples = 50000

        self.load_pretrained_gcmr(pretrained_gnn_checkpoint)
        # self.load_reconstrution_network(self.options.net_checkpoint)



    def render_normal(self, verts, faces):

        # render optimized mesh (normal, T_normal, image [-1,1])
        self.render.load_meshes(verts, faces)
        return self.render.get_image(type="rgb")

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
        # self.net.load_state_dict(data['recon'])
        if 'recon' in data.keys():
            print("111111111111111111111111111111111")
            self.net.load_state_dict(data['recon'])
        else:
            data = torch.load(
                "/media/gpu/dataset_SSD/code/PaMIR-main/networks/logs/6890_with_6890_meshcnn(thuman)/checkpoints/2024_04_11_19_26_43.pt")
            self.net.load_state_dict(data['recon'])
        # data = torch.load(
        #     "/media/gpu/dataset_SSD/code/PaMIR-main/networks/logs/6890_with_6890_meshcnn(thuman)/checkpoints/2024_04_11_19_26_43.pt")
        # self.net.load_state_dict(data['recon'])
        # self.smpl_param_regressor.load_state_dict(data['smpl_param_regressor'])
        # net.load_state_dict(torch.load(load_path, map_location=self.device))

        # self.net.load_state_dict(torch.load("/media/gpu/dataset_SSD/code/PaMIR-main/networks/checkpoints/50_net.pth", map_location=self.device))








    def test_gcmr_6890(self, img):
        self.graph_cnn.train()
        # self.net.train()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)


        # gcmr body prediction
        img_ = self.img_norm(img)

        # cv.imwrite("/media/gpu/dataset_SSD/code/PaMIR-main/debug/tttt.png",img_.squeeze().permute(1,2,0).detach().cpu().numpy()*255)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert = pred_vert_sub.transpose(1, 2)

        scale_, trans_ =self.forward_coordinate_conversion(pred_vert, cam_f, cam_tz, cam_c, pred_cam, None)


        return pred_cam,scale_, trans_,pred_vert
   


   
    def test_meshcnn_thuman(self,input_batch,img_folder):
        self.graph_cnn.train()
        # self.net.train()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        model_id = input_batch["model_id"]
        view_id = input_batch["view_id"]
        # openpose_keypoint=input_batch['keypoints']
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        gt_scale = input_batch['scale']
        gt_trans = input_batch['trans']
        # feature_local = input_batch["feature_local"]
        # feature_global = input_batch["feature_global"]
        # calib = input_batch["calib"]
        # label_path_vs = input_batch['label_path_vs']
        # label_path_norm = input_batch['label_path_norm']
        calib_data = input_batch["calib_data"]


        losses = dict()

        # prepare gt variables
        gt_vertices = self.smpl(gt_pose, gt_betas)
        gt_vertices_cam = gt_scale * gt_vertices + gt_trans
        # os.makedirs("/media/gpu/dataset_SSD/code/PaMIR-main/debug/%04s/" % (model_id[0]), exist_ok=True)
        # obj_io.save_obj_data({'v': gt_vertices_cam[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
        #                      "/media/gpu/dataset_SSD/code/PaMIR-main/debug/%04s/%04d_gtsmpl.obj"%(model_id[0],view_id[0]))

        # gt_keypoints_3d_cam = self.smpl.get_joints(gt_vertices_cam)
        # # gt_keypoints_2d=self.project_points(gt_keypoints_3d_cam, input_batch['cam_R'],
        # #                                  input_batch['cam_t'])
        # gt_keypoints_2d = self.forward_point_sample_projection(
        #     gt_keypoints_3d_cam, cam_r, cam_t, cam_f, cam_c)
        # gt_verts_2d = self.forward_point_sample_projection(
        #     gt_vertices_cam, cam_r, cam_t, cam_f, cam_c)
        # batch_size = gt_vertices_cam.shape[0]
        # print(gt_keypoints_2d.shape)

        # gcmr body prediction
        img_ = self.img_norm(img)

        # cv.imwrite("/media/gpu/dataset_SSD/code/PaMIR-main/debug/tttt.png",img_.squeeze().permute(1,2,0).detach().cpu().numpy()*255)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert = pred_vert_sub.transpose(1, 2)

        scale_, trans_ = self.forward_coordinate_conversion(pred_vert, cam_f, cam_tz, cam_c, pred_cam, None)



        os.makedirs(os.path.join(img_folder,  'ours_results'),
                    exist_ok=True)

        pred_vert_meshcnn = scale_ * pred_vert + trans_
        pred_vert_meshcnn= rigid_align(pred_vert_meshcnn.squeeze().cpu().detach().numpy(),
                                     gt_vertices_cam.squeeze().cpu().detach().numpy())
        pred_vert_meshcnn = rotate_vertex(pred_vert_meshcnn, -view_id[0].cpu())
        pred_vert_meshcnn=torch.from_numpy(pred_vert_meshcnn).unsqueeze(0).to(self.device)
        pred_vert_meshcnn = projection(pred_vert_meshcnn.squeeze() * 100, calib_data.squeeze())
        pred_vert_meshcnn[:, 1] *= -1

        # pred_vert_meshcnn, up_faces = util.up_sample(pred_vert_meshcnn, self.smpl_faces)


        fname_name=os.path.join(img_folder,  'ours_results','%04d' % model_id[0]+'_%03d_smpl.obj' % (
                view_id[0]))
        save_path=os.path.join(img_folder,  'ours_results','%04d' % model_id[0]+'_%03d_upsample.obj' % (
                view_id[0]))
        trimesh.Trimesh(pred_vert_meshcnn.cpu().detach().numpy(), self.smpl_faces).export(fname_name)
        
        util.manifold_upsample(fname_name, save_path, None, num_faces=40000, res=40000, simplify=True)

        # return pred_vert_cam,gt_vertices
    def test_all_step(self,input_batch,img_folder,input_dir):
        self.graph_cnn.train()
        self.net.train()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        model_id = input_batch["model_id"]
        view_id = input_batch["view_id"]
        # openpose_keypoint=input_batch['keypoints']
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        gt_scale = input_batch['scale']
        gt_trans = input_batch['trans']
        feature_local = input_batch["feature_local"]
        feature_global = input_batch["feature_global"]
        calib = input_batch["calib"]
        # label_path_vss = input_batch['label_path_vs']
        # label_path_norms = input_batch['label_path_norm']
        calib_data = input_batch["calib_data"]
        # label_path_faces = input_batch["label_path_faces"]
        # gt_dis=input_batch["gt_dis"]
        # gt_vertices_cam_meshcnn=input_batch["gt_vertices_cam"]
        # gt_hrnet_keypoint=input_batch["hrnet_keypoint"]
        #
        # print(input_batch['img_dir'])
        # diff_S_gt = input_batch["diff_S_gt"]
        # scan2d_gt = input_batch["scan2d_gt"]
        # smpl2d_gt = input_batch["smpl2d_gt"]

        losses = dict()

        # prepare gt variables
        gt_vertices = self.smpl(gt_pose, gt_betas)
        # gt_keypoints_3d = self.smpl.get_joints(gt_vertices)
        batch_size = gt_vertices.shape[0]

        # gt_vertices_cam=gt_scale * gt_vertices + gt_trans
        os.makedirs(os.path.join(img_folder, "%04d"%(model_id[0])),
                    exist_ok=True)


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
        pred_vert_meshcnn = rotate_vertex_with_smooth(pred_vert_meshcnn2[0:1].squeeze().detach().cpu(), -view_id[0].cpu(),
                                                      faces=self.smpl_faces)
        pred_vert_meshcnn = torch.from_numpy(pred_vert_meshcnn).to(self.device).float()
        pred_vert_meshcnn = projection(pred_vert_meshcnn * 100, calib_data[0:1].squeeze())
        pred_vert_meshcnn[:, 1] *= -1

        #
        # pred_vert_meshcnn=trimesh.load_mesh(os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2_smpl.obj" % ( view_id[0]))).vertices
        gt_vert_meshcnn2 = gt_scale * gt_vertices + gt_trans
        gt_keypoints_3d_meshcnn = self.smpl.get_joints(gt_vert_meshcnn2)
        gt_keypoints_2d = self.forward_point_sample_projection(
            gt_keypoints_3d_meshcnn, cam_r, cam_t, cam_f, cam_c)
        gt_vert_meshcnn = rotate_vertex(gt_vert_meshcnn2[0:1].squeeze().detach().cpu(), -view_id[0].cpu())
        gt_vert_meshcnn = torch.from_numpy(gt_vert_meshcnn).to(self.device).float()
        gt_vert_meshcnn = projection(gt_vert_meshcnn * 100, calib_data[0:1].squeeze())
        gt_vert_meshcnn[:, 1] *= -1
        #
        #
        pred_vert_meshcnn = rigid_align(np.array(pred_vert_meshcnn.squeeze().detach().cpu().numpy()),
                                        gt_vert_meshcnn.detach().cpu().numpy())
        pred_vert_meshcnn = torch.from_numpy(pred_vert_meshcnn).unsqueeze(0).to(self.device).float()
        gt_vert_meshcnn = gt_vert_meshcnn.unsqueeze(0).to(self.device).float()
        # loss_chamfer_meshcnn, _ = chamfer_distance1(pred_vert_meshcnn, gt_vert_meshcnn, unoriented=False)
        # #
        #
        obj_io.save_obj_data({'v': pred_vert_meshcnn[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
                             os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2.obj" % ( view_id[0])))
        obj_io.save_obj_data({'v': gt_vert_meshcnn[0].squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
                             os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_gt2.obj" % ( view_id[0])))

        # smpl2smplx_output_file = os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2_smplx.obj" % ( view_id[0]))
        # smplx_pose, smplx_shape = smpl2smplx(pred_vert_meshcnn,
        #                                      torch.from_numpy(self.smpl_faces.astype(np.int32)).unsqueeze(0).to(self.device),
        #                                      smpl2smplx_output_file)
        # smplx_mesh = trimesh.load_mesh(smpl2smplx_output_file)
        # smplx2smpl_output_file = os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2_smpl.obj" % ( view_id[0]))
        # smpl_pose, smpl_shape = smplx2smpl(
        #     torch.from_numpy(smplx_mesh.vertices.astype(np.float32)).unsqueeze(0).to(self.device),
        #     torch.from_numpy(smplx_mesh.faces.astype(np.int32)).unsqueeze(0).to(self.device), smplx2smpl_output_file)
        # smpl_shape = torch.from_numpy(np.zeros(10)).unsqueeze(0).to(self.device).float()
        # smpl_pose = torch.from_numpy(smpl_pose).unsqueeze(0).to(self.device).float()
        # trimesh.Trimesh(self.smpl(smpl_pose, smpl_shape).squeeze().detach().cpu().numpy(), self.smpl_faces).export(
        #     os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2_smpl_2.obj" % ( view_id[0])))
        # pred_vert_meshcnn = trimesh.load_mesh(os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2_smpl.obj" % ( view_id[0]))).vertices
        # pred_vert_meshcnn = rigid_align(np.array(pred_vert_meshcnn),
        #                                 gt_vert_meshcnn.squeeze().detach().cpu().numpy())
        # pred_vert_meshcnn = torch.from_numpy(pred_vert_meshcnn).unsqueeze(0).to(self.device).float()


        # # mesh_predict = Mesh2( file=None , device=self.device, vs=pred_vert_meshcnn, faces=self.smpl_faces,hold_history=True)
        mesh_predict = Mesh2(file=None, device=self.device, vs=pred_vert_meshcnn, faces=self.smpl_faces,

                             hold_history=True)
        # losses['gnn_edge_loss'] = 10 * cal_edge_loss2(mesh_predict, self.device)
        vs = pred_vert_meshcnn.permute(0, 2, 1)
        xyz = orthogonal(vs, calib, transforms=None)
        xy = xyz[:, :2, :]  # B, 2, N
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

        est_verts = self.net(edge_features, [mesh_predict])

        mesh_predict.update_verts(est_verts)


        mesh_path=os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_mesh.obj" % ( view_id[0]))
        mesh_predict.export(mesh_path)
        #
        
        
  
 
    




    def generate_point_grids(self, vol_res, cam_R, cam_t, cam_f, img_res):
        x_coords = np.array(range(0, vol_res), dtype=np.float32)
        y_coords = np.array(range(0, vol_res), dtype=np.float32)
        z_coords = np.array(range(0, vol_res), dtype=np.float32)

        yv, xv, zv = np.meshgrid(x_coords, y_coords, z_coords)
        xv = np.reshape(xv, (-1, 1))
        yv = np.reshape(yv, (-1, 1))
        zv = np.reshape(zv, (-1, 1))
        xv = xv / vol_res - 0.5 + 0.5 / vol_res
        yv = yv / vol_res - 0.5 + 0.5 / vol_res
        zv = zv / vol_res - 0.5 + 0.5 / vol_res
        pts = np.concatenate([xv, yv, zv], axis=-1)
        pts = np.float32(pts)
        pts_proj = np.dot(pts, cam_R.transpose()) + cam_t
        pts_proj[:, 0] = pts_proj[:, 0] * cam_f / pts_proj[:, 2] / (img_res / 2)
        pts_proj[:, 1] = pts_proj[:, 1] * cam_f / pts_proj[:, 2] / (img_res / 2)
        pts_proj = pts_proj[:, :2]

        return pts, pts_proj

    def forward_gcmr(self, img):
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')
        # GraphCMR forward
        batch_size = img.size()[0]
        img_ = self.img_norm(img)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert_sub = pred_vert_sub.transpose(1, 2)
        pred_vert = self.graph_mesh.upsample(pred_vert_sub)
        return pred_cam, pred_vert

    def forward_gcmr_6890(self, img):
        self.graph_cnn.eval()
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')
        # GraphCMR forward
        batch_size = img.size()[0]
        img_ = self.img_norm(img)
        pred_vert_sub, pred_cam = self.graph_cnn(img_)
        pred_vert = pred_vert_sub.transpose(1, 2)

        return pred_cam, pred_vert

    def forward_keypoint_projection(self, smpl_vert, cam):
        # print("smpl_vert:",smpl_vert.shape)
        pred_keypoints = self.smpl.get_joints(smpl_vert)
        # with open("/media/star/GYQ-KESU/code/code/PaMIR-main/results/test_data/results/gt_keypoints_3d.txt", "w") as f:
        #     f.write(str(pred_keypoints.detach()))
        pred_keypoints_2d = orthographic_projection(pred_keypoints, cam)
        return pred_keypoints_2d

    def forward_coordinate_conversion(self, pred_vert_tetsmpl,cam_f, cam_tz, cam_c, pred_cam,gt_trans):
        # calculates camera parameters
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
    def load_reconstrution_network(self, load_path):
        """load model from disk"""
        # save_filename = '%s_net.pth' % which_epoch
        # self.save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        # load_path = os.path.join(self.save_dir, save_filename)
        net = self.net
        print('loading for net ...', load_path)
        net.load_state_dict(torch.load(load_path, map_location=self.device))

    def forward_point_sample_projection(self, points, cam_r, cam_t, cam_f, cam_c):
        points_proj = points * cam_r.view((1, 1, -1)) + cam_t.view((1, 1, -1))
        points_proj = points_proj * (cam_f / cam_c) / points_proj[:, :, 2:3]
        points_proj = points_proj[:, :, :2]
        return points_proj

    def forward_infer_occupancy_value_grid_naive(self, img, vol, test_res, group_size):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts_ov = self.forward_infer_occupancy_value_group(img, vol, pts, pts_proj, group_size)
        pts_ov = pts_ov.reshape([test_res, test_res, test_res])
        return pts_ov

    def forward_infer_occupancy_value_grid_octree(self, img, vol, test_res, group_size,
                                             init_res=64, ignore_thres=0.05):
        pts, pts_proj = self.generate_point_grids(
            test_res, const.cam_R, const.cam_t, const.cam_f, img.size(2))
        pts = np.reshape(pts, (test_res, test_res, test_res, 3))
        pts_proj = np.reshape(pts_proj, (test_res, test_res, test_res, 2))

        pts_ov = np.zeros([test_res, test_res, test_res])
        dirty = np.ones_like(pts_ov, dtype=np.bool)
        grid_mask = np.zeros_like(pts_ov, dtype=np.bool)

        reso = test_res // init_res
        while reso > 0:
            grid_mask[0:test_res:reso, 0:test_res:reso, 0:test_res:reso] = True
            test_mask = np.logical_and(grid_mask, dirty)

            pts_ = pts[test_mask]
            pts_proj_ = pts_proj[test_mask]
            pts_ov[test_mask] = self.forward_infer_occupancy_value_group(
                img, vol, pts_, pts_proj_, group_size).squeeze()

            if reso <= 1:
                break
            for x in range(0, test_res - reso, reso):
                for y in range(0, test_res - reso, reso):
                    for z in range(0, test_res - reso, reso):
                        # if center marked, return
                        if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                            continue
                        v0 = pts_ov[x, y, z]
                        v1 = pts_ov[x, y, z + reso]
                        v2 = pts_ov[x, y + reso, z]
                        v3 = pts_ov[x, y + reso, z + reso]
                        v4 = pts_ov[x + reso, y, z]
                        v5 = pts_ov[x + reso, y, z + reso]
                        v6 = pts_ov[x + reso, y + reso, z]
                        v7 = pts_ov[x + reso, y + reso, z + reso]
                        v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                        v_min = v.min()
                        v_max = v.max()
                        # this cell is all the same
                        if (v_max - v_min) < ignore_thres:
                            pts_ov[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                            dirty[x:x + reso, y:y + reso, z:z + reso] = False
            reso //= 2
        return pts_ov

    def forward_infer_occupancy_value_group(self, img, vol, pts, pts_proj, group_size):
        assert isinstance(pts, np.ndarray)
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        pts_num = pts.shape[0]
        pts = torch.from_numpy(pts).unsqueeze(0).to(self.device)
        pts_proj = torch.from_numpy(pts_proj).unsqueeze(0).to(self.device)
        pts_group_num = (pts.size()[1] + group_size - 1) // group_size
        pts_ov = []
        for gi in tqdm(range(pts_group_num), desc='SDF query'):
            # print('Testing point group: %d/%d' % (gi + 1, pts_group_num))
            pts_group = pts[:, (gi * group_size):((gi + 1) * group_size), :]
            pts_proj_group = pts_proj[:, (gi * group_size):((gi + 1) * group_size), :]
            outputs = self.forward_infer_occupancy_value(
                img, pts_group, pts_proj_group, vol)
            pts_ov.append(np.squeeze(outputs.detach().cpu().numpy()))
        pts_ov = np.concatenate(pts_ov)
        pts_ov = np.array(pts_ov)
        return pts_ov

    def optm_smpl_cam_param_without_mesh(self, img, mask, img_gray, keypoint, pred_vert, smpl_pose, smpl_shape,
                                         scale2pic, trans2pic, calib, iter_num):
        assert iter_num > 0

        self.net.eval()
        smpl_vertex_code, smpl_face_code, smpl_faces, smpl_tetras = \
            util.read_smpl_constants('./data')
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)
        kp_conf = keypoint[:, :, -1:].clone()
        kp_detection = keypoint[:, :, :-1].clone().detach()
        img = img.squeeze().detach().cpu().numpy().transpose(1, 2, 0).copy() * 255

        kp_detection_points = (kp_conf * kp_detection).squeeze().detach().cpu().numpy()
        for idx, point in enumerate(kp_detection_points):
            x = int((point[0] + 1) * 256)
            y = int((point[1] + 1) * 256)
            # img = cv.circle(img, (x,y), 3, (255, 0, 0), -1)
            img = cv.putText(img, str(idx), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 3, cv.LINE_AA)





        pred_vert = copy.deepcopy(pred_vert.detach()).to(self.device).requires_grad_(False)
        # shape_new=copy.deepcopy(smpl_shape.detach()).to(self.device).requires_grad_(False)
        # pose_new = copy.deepcopy(smpl_pose.detach()).to(self.device).requires_grad_(False)
        # scale_new=copy.deepcopy(scale).to(self.device).requires_grad_(True)
        # trans_new=copy.deepcopy(trans).to(self.device).requires_grad_(True)
        scale2pic_new = copy.deepcopy(scale2pic.detach()).to(self.device).requires_grad_(True)
        trans2pic_new = copy.deepcopy(trans2pic.detach()).to(self.device).requires_grad_(True)

        optm = torch.optim.Adam(params=( scale2pic_new ,trans2pic_new ),
                                lr=5e-2)

        for i in tqdm(range(iter_num), desc='Body Fitting Optimization'):

            pred_vert_meshcnn =  scale2pic_new*pred_vert + trans2pic_new
            keypoint_new = self.smpl.get_joints(pred_vert_meshcnn[:, :6890])
            keypoint_new=kp_conf*keypoint_new
            keypoint_vs = keypoint_new.permute(0, 2, 1)
            keypoint_xyz = orthogonal(keypoint_vs, calib, transforms=None)
            keypoint_xy = keypoint_xyz[:, :2, :]  # B, 2, N
            keypoint_xy_cal=keypoint_xy.permute(0, 2, 1)

            points = keypoint_xy * 256
            points[:, 0:1, :] = points[:, 0:1, :] + 256
            points[:, 1:2, :] = points[:, 1:2, :] + 256
            points = points.squeeze().detach().cpu().permute(1, 0).numpy()

            for point in (points):
                img = cv.circle(img, tuple([int(point[0]), int(point[1])]), 1, (0, 0, 255), -1)



            cv.imwrite('/media/gpu/dataset_SSD/code/PaMIR-main/debug/test2.png', img)

            # print(keypoint_xy_cal.shape,kp_detection.shape)
            loss_kp = torch.mean((kp_conf * keypoint_xy_cal - kp_conf * kp_detection) ** 2)





            T_normal_F, T_normal_B = self.render_normal(
                pred_vert_meshcnn,
                torch.from_numpy(smpl_faces).unsqueeze(0).to(self.device),
            )
            T_mask_F, T_mask_B = self.render.get_image(cam_type="frontback", type="mask")

            torchvision.utils.save_image(
                ((T_normal_B.detach().cpu() + 1.0) * 0.5), '/media/gpu/dataset_SSD/code/PaMIR-main/debug/mask_b.png')

            torchvision.utils.save_image(
                ((T_normal_F.detach().cpu() + 1.0) * 0.5), '/media/gpu/dataset_SSD/code/PaMIR-main/debug/mask_f.png')


            smpl_arr = torch.cat([T_mask_F, T_mask_B], dim=-1)
            gt_arr = mask.repeat(1, 1, 2)
            diff_S = torch.abs(smpl_arr - gt_arr)
            torchvision.utils.save_image(
                ((diff_S.detach().cpu() + 1.0) * 0.5), '/media/gpu/dataset_SSD/code/PaMIR-main/debug/mask.png')

            losses_silhouette = diff_S.mean()

            loss =  losses_silhouette*10+loss_kp*6

            #
            optm.zero_grad()
            loss.backward()
            optm.step()
            print('Iter No.%d: loss_fitting = %f,' % (i, loss.item()))






        return  scale2pic_new,trans2pic_new
    def test_real_world_renderpeople(self,pred_vert_meshcnn,input_batch,img_folder,input_dir):
        self.graph_cnn.train()
        self.net.train()

        # some constants
        cam_f, cam_tz, cam_c = const.cam_f, const.cam_tz, const.cam_c
        cam_r = torch.tensor([1, -1, -1], dtype=torch.float32).to(self.device)
        cam_t = torch.tensor([0, 0, cam_tz], dtype=torch.float32).to(self.device)

        # training data
        img = input_batch['img']
        # model_id = input_batch["model_id"]
        # view_id = input_batch["view_id"]
        # openpose_keypoint=input_batch['keypoints']
        gt_betas = input_batch['betas']
        gt_pose = input_batch['pose']
        # gt_scale = input_batch['scale']
        # gt_trans = input_batch['trans']
        feature_local = input_batch["feature_local"]
        feature_global = input_batch["feature_global"]
        calib = input_batch["calib"]
        # label_path_vss = input_batch['label_path_vs']
        # label_path_norms = input_batch['label_path_norm']
        # calib_data = input_batch["calib_data"]
        # label_path_faces = input_batch["label_path_faces"]

        losses = dict()


        # util.manifold_upsample(os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2.obj" % ( view_id[0])), os.path.join(img_folder, "%04d"%(model_id[0]),"%04d_meshcnn_pred2_upsample.obj" % ( view_id[0])), None, num_faces=40000,
        #                        res=40000, simplify=True)
        _, filename = os.path.split(input_batch['img_dir'][0])
        # gt_vert=self.smpl(gt_pose,gt_betas)
        # gt_vert_meshcnn = rigid_align(gt_vert.squeeze().detach().cpu().numpy(),
        #                                 pred_vert_meshcnn.squeeze().detach().cpu().numpy())
        #
        # obj_io.save_obj_data({'v': pred_vert_meshcnn.squeeze().detach().cpu().numpy(), 'f': self.smpl_faces},
        #                      os.path.join(img_folder,  "%s_meshcnn_pred2.obj" % (filename[:-4])))
        # obj_io.save_obj_data({'v': gt_vert_meshcnn, 'f': self.smpl_faces},
        #                      os.path.join(img_folder,  "%s_meshcnn_gt2.obj" % (filename[:-4])))

        mesh = trimesh.Trimesh(pred_vert_meshcnn, self.smpl_faces)
        pred_vert_meshcnn = trimesh.smoothing.filter_laplacian(mesh, lamb=0.2, iterations=3).vertices
        pred_vert_meshcnn =torch.from_numpy(pred_vert_meshcnn).unsqueeze(0).to(self.device).float()

        mesh_predict = Mesh2(file=None, device=self.device, vs=pred_vert_meshcnn, faces=self.smpl_faces,

                             hold_history=True)
        vs = pred_vert_meshcnn.permute(0, 2, 1)
        xyz = orthogonal(vs, calib, transforms=None)
        xy = xyz[:, :2, :]  # B, 2, N
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

        est_verts = self.net(edge_features, [mesh_predict])

        mesh_predict.update_verts(est_verts)


        mesh_path=os.path.join(img_folder, "%s_mesh.obj" % ( filename[:-4]))
        mesh_predict.export(mesh_path)

        util.manifold_upsample(mesh_path, mesh_path[:-4] + '_upsample.obj', None, num_faces=40000,
                               res=40000, simplify=True)
        #
        rgb_normal_F = os.path.join(input_dir,  "normal_F", "%s.png" % (filename[:-4]))
        rgb_normal_B = os.path.join(input_dir, "normal_B", "%s.png" % (filename[:-4]))
        image_dir=os.path.join(input_dir, "render")
        refine_cloth(rgb_normal_F, rgb_normal_B, mesh_path[:-4] + "_upsample.obj", image_dir,
                     mesh_path[:-4] + "_refine.obj","%s.png" % ( filename[:-4]))
        replace_hands(os.path.join(mesh_path[:-4] + "_refine.obj", "%s_refine.obj" % (filename[:-4])),
                      os.path.join(img_folder,  "%s_meshcnn_gt2.obj" % (filename[:-4])),
                      os.path.join(mesh_path[:-4] + "_refine.obj", "%s_refine_hand.obj" % (filename[:-4])))


    # def forward_infer_occupancy_value(self, img, pts, pts_proj, vol):
    #     return self.pamir_net(img, vol, pts, pts_proj)[-1]
