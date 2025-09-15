import os
import json
import ast
import argparse
import numpy as np
from collections import namedtuple
from datetime import datetime

from .util import create_code_snapshot


class TrainOptions(object):
    """Object that handles command line options."""
    def __init__(self):
        self.parser = argparse.ArgumentParser()
     


        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', default='body',help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--resume', dest='resume', default=False, action='store_true', help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true', help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        io = self.parser.add_argument_group('io')
        io.add_argument('--log_dir', default='logs', help='Directory to store logs')
        
        io.add_argument('--from_json', default=None, help='Load options from json file instead of the command line')
        io.add_argument('--pretrained_gcmr_checkpoint', default="./xx.pt", help='Load a pretrained network when starting training')

        dataloading = self.parser.add_argument_group('Data Loading')
        dataloading.add_argument('--dataset_dir', default="/media/star/软件/GYQ/dataset",type=str, help='dataset directory')
        dataloading.add_argument('--view_num_per_item', type=int, default=40, help='view_num_per_item')
        dataloading.add_argument('--point_num', type=int, default=5000, help='number of point samples')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=500, help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=8, help='Batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=20000, help='Checkpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')
        train.add_argument('--use_adaptive_geo_loss', type=ast.literal_eval, dest='use_adaptive_geo_loss', default=False, help='use_adaptive_geo_loss')
        train.add_argument('--use_multistage_loss', type=ast.literal_eval, dest='use_multistage_loss', default=True, help='use_multistage_loss')
        train.add_argument('--use_gt_smpl_volume', type=ast.literal_eval, dest='use_gt_smpl_volume', default=False, help='use_gt_smpl_volume')
        train.add_argument('--use_attention_texture', type=ast.literal_eval, dest='use_attention_texture', default=False, help='use_gt_smpl_volume')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')

        weights = self.parser.add_argument_group('Loss Weights')
        weights.add_argument('--weight_smpl', type=float, default=0.4, help='weight_geo')
        weights.add_argument('--weight_gnn', type=float, default=0.4, help='weight_geo')
        weights.add_argument('--weight_geo', type=float, default=0.2, help='weight_geo')

        optim = self.parser.add_argument_group('Optimization')
        optim.add_argument('--adam_beta1', type=float, default=0.9, help='Value for Adam Beta 1')
        optim.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
        optim.add_argument('--wd', type=float, default=0, help='Weight decay weight')

        logging = self.parser.add_argument_group('Logging')
        logging.add_argument('--debug', dest='debug', default=False, action='store_true', help='If set, debugging messages will be printed')
        logging.add_argument('--quiet', '-q', dest='quiet', default=False, action='store_true', help='If set, only warnings will be printed')
        logging.add_argument('--logfile', dest='logfile', default=None, help='If set, the log will be saved using the specified filename.')

        reconsting = self.parser.add_argument_group('Reconsting')
        reconsting.add_argument('--dataset_mode', choices={"classification", "segmentation"}, default='reconstruction')
        reconsting.add_argument('--ninput_edges', type=int, default=750, help='# of input edges (will include dummy edges)')
        reconsting.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples per epoch')
        reconsting.add_argument('--datarood', nargs='+', type=str,
                                 default=None,
                                 help='path to meshes (should have subfolders train, up_sample, test)')
        # network params
        # self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        reconsting.add_argument('--arch', type=str, default='mconvnet', help='selects network to use') #todo add choices
        reconsting.add_argument('--res_blocks', type=int, default=3, help='# of res blocks')
        reconsting.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses') #todo make generic
        reconsting.add_argument('--convs', nargs='+', default=[1024,1024,1024,2048, 2048, 1024, 512, 256, 128],
                                 type=int, help='conv filters')
        reconsting.add_argument('--pools', nargs='+', default=[0.0, 0.0, 0.0, 0.0], type=float, help='pooling res')
        reconsting.add_argument('--norm', type=str, default='batch',help='instance normalization or batch normalization or group normalization')
        reconsting.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        reconsting.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        reconsting.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # general params
        reconsting.add_argument('--num_threads', default=16, type=int, help='# threads for loading data')
        reconsting.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # reconsting.add_argument('--name', type=str, default='debug', help='name of the experiment. It decides where to store samples and models')
        reconsting.add_argument('--checkpoints_dir', type=str, default='./networks/checkpoints', help='models are saved here')
        reconsting.add_argument('--net_checkpoint', type=str,
                                 default='',
                                 help='models are saved here')

        reconsting.add_argument('--serial_batches', action='store_true', help='if true, takes meshes in order, otherwise takes them randomly')
        reconsting.add_argument('--seed', type=int, help='if specified, uses seed')
        # visualization params
        reconsting.add_argument('--export_folder', type=str, default='', help='exports intermediate collapses to this folder')
        #
        reconsting.add_argument('--leaky_relu', type=float, metavar='1eN', default=0.01, help='slope for leaky relu')
        reconsting.add_argument('--init_weights', type=float, default=0.002, help='initialize NN with this size')
        reconsting.add_argument('--overlap', type=int, default=0, help='overlap for bfs')

        reconsting.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
        reconsting.add_argument('--save_latest_freq', type=int, default=360, help='frequency of saving the latest results')
        reconsting.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        reconsting.add_argument('--run_test_freq', type=int, default=1, help='frequency of running test in training script')
        reconsting.add_argument('--continue_train', action='store_true', default=True, help='continue training: load the latest model')
        reconsting.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        reconsting.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        reconsting.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        reconsting.add_argument('--niter', type=int, default=2, help='# of iter at starting learning rate')
        reconsting.add_argument('--niter_decay', type=int, default=48, help='# of iter to linearly decay learning rate to zero')
        reconsting.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        # reconsting.add_argument('--lr', type=float, default=0.4e-4, help='initial learning rate for adam')
        reconsting.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        reconsting.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # data augmentation stuff
        reconsting.add_argument('--num_aug', type=int, default=10, help='# of augmentation files')
        reconsting.add_argument('--scale_verts', action='store_true', help='non-uniformly scale the mesh e.g., in x, y or z')
        reconsting.add_argument('--slide_verts', type=float, default=0, help='percent vertices which will be shifted along the mesh surface')
        reconsting.add_argument('--flip_edges', type=float, default=0, help='percent of edges to randomly flip')
        # tensorboard visualization
        reconsting.add_argument('--no_vis', action='store_true', help='will not use tensorboard')
        reconsting.add_argument('--verbose_plot', action='store_true', help='plots network weights, etc.')
        reconsting.add_argument('--is_train', default=True, help='is  train, etc.')
        return

    def parse_args(self):
        """Parse input arguments."""
        self.start_time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.args = self.parser.parse_args()
        # If config file is passed, override all arguments with the values from the config file
        if self.args.from_json is not None:
            path_to_json = os.path.abspath(self.args.from_json)
            with open(path_to_json, 'r') as f:
                json_args = json.load(f)
                json_args = namedtuple('json_args', json_args.keys())(**json_args)
                return json_args
        else:
            self.args.log_dir = os.path.join(os.path.abspath(self.args.log_dir), self.args.name)
            self.args.summary_dir = os.path.join(self.args.log_dir, 'tensorboard')
            if not os.path.exists(self.args.log_dir):
                os.makedirs(self.args.log_dir)
            self.args.checkpoint_dir = os.path.join(self.args.log_dir, 'checkpoints')
            if not os.path.exists(self.args.checkpoint_dir):
                os.makedirs(self.args.checkpoint_dir)
            self.save_dump()
            return self.args

    def save_dump(self):
        """Store all argument values to a json file and create a code snapshot (useful for debugging)
        The default location is logs/expname/config_[...].json.
        """
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)
        with open(os.path.join(self.args.log_dir, 'config_%s.json' % self.start_time_str), 'w') as f:
            json.dump(vars(self.args), f, indent=4)
        create_code_snapshot('./', os.path.join(self.args.log_dir, 'code_bk_%s.tar.gz' % self.start_time_str))
        return
