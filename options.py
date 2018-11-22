import argparse
import os

import torch

from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='eyeglasses',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm_layer', type=str, default='instancenorm',
                                 help='instance normalization or batch normalization')

        # input/output sizes
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--img_size', type=int, default=128, help='scale images to this size')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of two networks')

        # for setting inputs
        self.parser.add_argument('--nThreads', default=10, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--num_epoch', default=100, type=int,
                                 help='number of epoches to run for each experiment')
        self.parser.add_argument('--n_layers_D', default=3, type=int, help='number of layers in discriminator')

        # for displays
        self.parser.add_argument('--print_freq', type=int, default=100)
        self.parser.add_argument('--display_freq', type=int, default=100)
        self.parser.add_argument('--save_freq', type=int, default=1000)
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for generator
        self.parser.add_argument('--ngf', type=int, default=32, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=32, help='# of dis filters in first conv layer')
        self.parser.add_argument('--n_downsample', type=int, default=3,
                                 help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks', type=int, default=9,
                                 help='number of residual blocks in the local enhancer network')
        self.parser.add_argument('--train', default=True, action='store_true', help="training mode")
        self.parser.add_argument('--debug', action='store_true', help="start debugging mode")
        self.parser.add_argument('--load_path', type=str, help='model path to load')
        self.parser.add_argument('--eps', type=float, default=1e-7, help='used for generating flow')
        self.parser.add_argument("--no_share_weight", default=False, action='store_true',
                                 help='whether to share weight between two generated flow')
        self.parser.add_argument('--no_html', default=False, action='store_true')
        self.parser.add_argument('--num_scale', default=3, type=int, help='number of images for different scale')
        self.parser.add_argument('--lambda_feat', type=float, default=10, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_gan_feat', type=float, default=1e-2,
                                 help='weight for gan feature matching loss')
        self.parser.add_argument('--lambda_tv', type=float, default=0.1, help='weight for tv loss')
        self.parser.add_argument('--which_epoch', type=int)
        self.parser.add_argument('--which_epoch_iter', type=int)
        self.parser.add_argument('--patch_size', type=int, default=5,
                                 help='patch size for calculation of flow regularization loss')
        self.parser.add_argument('--lambda_color', type=int, default=0.1, help='weights for color incoherence')
        self.parser.add_argument('--lambda_flow_reg', type=float, default=1e-1, help='weights for flow coherence')
        self.parser.add_argument('--lambda_mask', type=float, default=1e-1)
        self.parser.add_argument('--lambda_gp', type=float, default=1e-1)
        self.parser.add_argument('--lambda_cls', type=float, default=1e-1)
        self.parser.add_argument('--use_lsgan', action='store_true', default=False)
        self.parser.add_argument('--sel_attrs', nargs='+', type=str)
        self.parser.add_argument('--res_weight', type=float,default=1e-1)
        self.parser.add_argument('--only_res', action='store_true',default=False)
        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        if save:
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        self.opt.nc = len(self.opt.sel_attrs)
        return self.opt
