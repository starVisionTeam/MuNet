import sys

import torch
import numpy as np
import os
from models.networks import *
from os.path import join
from util.util import print_network, sample_surface, local_nonuniform_penalty, cal_edge_loss
from models.loss import chamfer_distance1, BeamGapLoss
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes
#若要改变batch_size的值，需要改变beam_gap损失，改变partnet结构，改变ClassifierModel的set_input和forward结构
class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        # self.device = torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.num_samples = 50000
        self.label_path_vss = None
        self.label_path_norms = None
        self.part_mesh = None
        self.edge_features = None
        self.part_meshs = None
        self.label_path_vs = None
        self.label_path_norm = None
        self.edge_feature = None
        self.loss = 0
        self.n_loss = 0
        # self.number = 0



        #
        # self.nclasses = opt.nclasses

        # load/define networks##############################
        self.net = init_net(self.device, self.opt)
        self.net.train(self.is_train)
        # self.criterion = BeamGapLoss(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)##########还没改改改改
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):#改#####################
        if self.opt.phase == 'train':
            self.part_meshs = data['part_mesh']

            # print(sys.getsizeof(self.part_meshs))
            # self.label_path_vss = torch.Tensor(data['label_path_vs']).float().to(self.device)
            # self.label_path_norms = torch.Tensor(data['label_path_norm']).float().to(self.device)
            # self.edge_features = torch.Tensor(data['edge_feature']).float().to(self.device)
            self.label_path_vss = data['label_path_vs']
            self.label_path_norms = data['label_path_norm']
            self.edge_features = data['edge_feature']
            self.model_id = data['model_id']
            self.view_id = data['view_id']
            # self.mesh_vs_scale = data['mesh_vs_scale']
            # self.mesh_vs_translations = data['mesh_vs_translations']
        else:
            self.part_meshs = data['part_mesh']
            # self.edge_features = torch.Tensor(data['edge_feature']).float().to(self.device)
            self.edge_features = data['edge_feature']
            self.model_id = data['model_id']
            self.view_id = data['view_id']
            # self.mesh_vs_scale = data['mesh_vs_scale']
            # self.mesh_vs_translations = data['mesh_vs_translations']
        # input_edge_features = torch.from_numpy(data['edge_features']).float()
        # labels = torch.from_numpy(data['label']).long()
        # # set inputs
        # self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        # self.labels = labels.to(self.device)
        # self.mesh = data['mesh']
        # if self.opt.dataset_mode == 'segmentation' and not self.is_train:
        #     self.soft_label = torch.from_numpy(data['soft_label'])

    def forward(self, i):
        self.loss = 0
        for B_num, part_mesh in enumerate(self.part_meshs):
            self.part_mesh = part_mesh
            self.label_path_vs = self.label_path_vss[B_num].unsqueeze(0).to('cuda')
            self.label_path_norm = self.label_path_norms[B_num].unsqueeze(0).to('cuda')
            self.edge_feature = self.edge_features[B_num].unsqueeze(0).to('cuda')
            # if i < 5:
            # self.criterion.update_pm(self.part_mesh, torch.cat([self.label_path_vs, self.label_path_norm], dim=-1))

            # self.n_loss = torch.Tensor(np.array([0])).to(self.device).float()
            self.n_loss = 0
            for part_i, est_verts in enumerate(self.net(self.edge_feature, self.part_mesh)):
                self.part_mesh.update_verts(est_verts[0], part_i)
                recon_xyz, recon_normals = sample_surface(self.part_mesh.main_mesh.faces, self.part_mesh.main_mesh.vs.unsqueeze(0), self.num_samples)
                recon_xyz, recon_normals = recon_xyz.to('cuda').type(torch.float), recon_normals.to('cuda').type(torch.float)
                xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance1(recon_xyz, self.label_path_vs,  x_normals=recon_normals,  y_normals=self.label_path_norm, unoriented=False)

                loss = xyz_chamfer_loss + (0.1*normals_chamfer_loss)#loss:-1.267
                m = loss
                m.backward()

                self.part_mesh.main_mesh.vs.detach_()
                self.n_loss += float(m)

            # self.n_loss /= 2
            if i % 5 == 0:
            # if i == 1:
                os.makedirs('/media/star/2023/dataset/thuman2_CHON/train_data/train_result/%04d' % self.model_id[0], exist_ok=True)
                self.part_mesh.export(os.path.join("/media/star/2023/dataset/thuman2_CHON/train_data/train_result", '%04d' % self.model_id[0], '%03d.obj' % ((self.view_id[0]) * 20)))
                # self.number += 1
            self.loss += self.n_loss
        self.loss /= len(self.part_meshs)
        # self.loss *= 100
        # self.loss.backward()
        # out = self.net(self.edge_features, self.mesh)
        # return recon_xyz, recon_normals
    #####################################batch_size为1的版本###################需要改变beam_gap损失，改变partnet结构
        # self.criterion.update_pm(self.part_mesh, torch.cat([self.label_path_vs, self.label_path_norm], dim=-1))
        # for part_i, est_verts in enumerate(self.net(self.edge_feature, self.part_mesh)):
        #     self.part_mesh.update_verts(est_verts[0], part_i)
        #     recon_xyz, recon_normals = sample_surface(self.part_mesh.main_mesh.faces,
        #                                               self.part_mesh.main_mesh.vs.unsqueeze(0), self.num_samples)
        #     recon_xyz, recon_normals = recon_xyz.type(torch.float32), recon_normals.type(torch.float32)
        #     xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, self.label_path_vs,
        #                                                               x_normals=recon_normals,
        #                                                               y_normals=self.label_path_norm)
        #     beam_loss = self.criterion(self.part_mesh, part_i)
        #     if i < 27:
        #         self.loss = beam_loss
        #     else:
        #         self.loss = (xyz_chamfer_loss + (1e-1 * normals_chamfer_loss))
        #     self.loss += 0.1 * local_nonuniform_penalty(self.part_mesh.main_mesh).float()
        #     self.loss.backward()

    def backward(self):
        pass

        # self.loss.backward()

    def optimize_parameters(self, i):
        self.optimizer.zero_grad()
        self.forward(i)
        # self.backward()
        self.optimizer.step()



##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        print('loading for net ...', load_path)
        net.load_state_dict(torch.load(load_path, map_location=self.device))

        # if isinstance(net, torch.nn.DataParallel):
        #     net = net.module
        # print('loading the model from %s' % load_path)
        # # PyTorch newer than 0.4 (e.g., built from
        # # GitHub source), you can remove str() on self.device
        # state_dict = torch.load(load_path, map_location=str(self.device))
        # if hasattr(state_dict, '_metadata'):
        #     del state_dict._metadata
        # net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        torch.save(self.net.state_dict(), save_path)

        # if len(self.gpu_ids) > 0 and torch.cuda.is_available():
        #     torch.save(self.net.module.cpu().state_dict(), save_path)
        #     self.net.cuda(self.gpu_ids[0])
        # else:
        #     torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            for B_num, part_mesh in enumerate(self.part_meshs):
                edge_feature = self.edge_features[B_num].unsqueeze(0).to('cuda')
                for part_i, est_verts in enumerate(self.net(edge_feature, part_mesh)):
                    part_mesh.update_verts(est_verts[0], part_i)
                    # os.makedirs('/media/star/软件/LYH/test_result/cape_result/%04d' % self.model_id[0], exist_ok=True)
                    # part_mesh.export(os.path.join('/media/star/软件/LYH/test_result/cape_result', '%04d' % self.model_id[0], '%04d.obj' % ((self.view_id[0]) * 120)))
                    # os.makedirs('/media/amax/4C76448F76447C28/LYH/renderpeople/L1+L2/our/no_smpl_nml/%04d' % self.model_id[0], exist_ok=True)
                    # part_mesh.export(os.path.join('/media/amax/4C76448F76447C28/LYH/renderpeople/L1+L2/our/no_smpl_nml', '%04d' % self.model_id[0],'%04d.obj' % ((self.view_id[0]) * 60)))
                    os.makedirs('/media/amax/4C76448F76447C28/LYH/realworld/extreme/result/our', exist_ok=True)
                    part_mesh.export(os.path.join('/media/amax/4C76448F76447C28/LYH/realworld/extreme/result/our','%04d.obj' % (self.view_id[0])))
                    # os.makedirs('/media/amax/4C76448F76447C28/LYH/thuman2.0/test_result/our/%04d' % self.model_id[0], exist_ok=True)
                    # part_mesh.export(os.path.join('/media/amax/4C76448F76447C28/LYH/thuman2.0/test_result/our','%04d' % self.model_id[0],'%04d.obj' % (self.view_id[0]*60)))


            # out = self.forward()
            # # compute number of correct
            # pred_class = out.data.max(1)[1]
            # label_class = self.labels
            # self.export_segmentation(pred_class.cpu())
            # correct = self.get_accuracy(pred_class, label_class)
        # return correct, len(label_class)

    # def get_accuracy(self, pred, labels):
    #     """computes accuracy for classification / segmentation """
    #     if self.opt.dataset_mode == 'classification':
    #         correct = pred.eq(labels).sum()
    #     elif self.opt.dataset_mode == 'segmentation':
    #         correct = seg_accuracy(pred, self.soft_label, self.mesh)
    #     return correct

    # def export_segmentation(self, pred_seg):
    #     if self.opt.dataset_mode == 'segmentation':
    #         for meshi, mesh in enumerate(self.mesh):
    #             mesh.export_segments(pred_seg[meshi, :])
