import argparse
import sys

from util.data_util import parse_opt
from faceAnaModel import FaceAnalogyModel
import os
import shutil
from face_dataset import create_dataloader_celeba_test
import face_dataset
from torch import nn
from torch.autograd import Variable
from util.visualizer import Visualizer
import util.util as pixutils
import torch
import numpy as np
import torch.nn.functional as F
from util import data_util

parser = argparse.ArgumentParser()
parser.add_argument('--exp_folder', type=str)
parser.add_argument('--dataset_size', type=int, default=10)
parser.add_argument('--which_epoch', type=int)
parser.add_argument('--which_iter', type=int)
parser.add_argument('--test_img_size', type=int)
parser.add_argument('--create_attr_folder', action='store_true')
parser.add_argument('--is_hd', action='store_true')
parser.add_argument('--upsample_flow', action='store_true')
parser.add_argument('--result_folder', type=str)
parser.add_argument('--attr_folder', type=str)

new_opts = parser.parse_args()
opt = parse_opt(os.path.join(new_opts.exp_folder, "opt.txt"))


def upsample_flow(raw_img, lowres_flow):

    flow_grid_np = np.mgrid[-1:1 + opt.eps:2 / (opt.img_size - 1),
                   -1:1 + opt.eps:2 / (opt.img_size - 1)]
    flow_grid_tensor = torch.from_numpy(flow_grid_np.astype(np.float32)).cuda().unsqueeze(0)
    img_grid = flow_grid_tensor.permute(0,1,3,2) + lowres_flow
    upsampled_grid = F.upsample(img_grid,
                                size=new_opts.test_img_size,
                                mode='bilinear')
    sampled_img = F.grid_sample(raw_img,
                                upsampled_grid.permute(0,2,3,1))
    return sampled_img


def compose_img(Ax, By, By_flow, Ax_mask):
    warpped_By = upsample_flow(By, By_flow)
    upsampled_mask = F.upsample(Ax_mask,
                                size=new_opts.test_img_size,
                                mode='nearest')
    composed_img = Ax * upsampled_mask + warpped_By * (1 - upsampled_mask)
    return (Ax*upsampled_mask).data,(warpped_By*(1-upsampled_mask)).data,composed_img.data


setattr(opt, 'nc', 1)
setattr(opt, 'checkpoints_dir',
        new_opts.result_folder)
setattr(opt, 'train', False)


attr_folder = new_opts.attr_folder

if new_opts.create_attr_folder:
    if os.path.exists(attr_folder):
        shutil.rmtree(attr_folder)
    if new_opts.is_hd:
        face_dataset.create_hd_attr_file_dir(opt.sel_attrs[0],
                                             attr_folder,
                                             max_size=50)
    else:
        face_dataset.create_attr_file_dir(opt.sel_attrs[0],
                                          attr_folder,
                                          max_size=50)

else:
    model = FaceAnalogyModel(opt)
    model.load(new_opts.exp_folder, new_opts.which_epoch,
               new_opts.which_iter)
    model.cuda()
    model.eval()
    if os.path.exists(os.path.join(new_opts.exp_folder, 'test_dir')):
        #shutil.rmtree(os.path.join(new_opts.exp_folder, 'test_dir'))
        pass
    dataloader = create_dataloader_celeba_test(attr_folder,
                                               new_opts.dataset_size,
                                               new_opts.test_img_size,
                                               opt.sel_attrs[0],
                                               new_opts.is_hd
                                               )
    visualizer = Visualizer(opt)
    iter = 0
    for Ax, Ax_label, By, By_label in dataloader:
        raw_Ax, raw_By = Ax.cuda(), By.cuda()
        Ax = F.upsample(Ax,
                        size=opt.img_size,
                        mode='bilinear')
        By = F.upsample(By,
                        size=opt.img_size,
                        mode='bilinear')
        Ax = Ax.cuda()
        By = By.cuda()
        Ax_label = Variable(Ax_label.cuda())
        By_label = Variable(By_label.cuda())
        _, visual_images = model(Ax, Ax_label,
                                 None,
                                 By, By_label,
                                 None,
                                 new_opts.which_epoch,
                                 visual=True)
        visual_images = [variable.data for variable in visual_images]
        visual_images = dict(zip(model.visualizer_names, visual_images))
        visuals = {}
        for item_name in visual_images:
            if 'flow' in item_name:
                flow_numpy = visual_images[item_name].squeeze(0).permute(1,2,0).cpu().numpy()
                flow_numpy = opt.img_size * flow_numpy
                flow_vis = data_util.visualize_opt_flow(flow_numpy,opt.img_size)
                visuals[item_name] = flow_vis
            else:
                tensor = visual_images[item_name]
                img_list = []
                img_list.append(pixutils.tensor2im(tensor[0]))
                visuals[item_name] = img_list
        no_flow = visual_images['By'] * visual_images['By_mask'] + visual_images['Ax'] * (1-visual_images['By_mask'])
        visuals['no_flow'] = pixutils.tensor2im(no_flow[0])
        if new_opts.test_img_size != opt.img_size:
            masked_Ax,masked_By,raw_img_with_flow = compose_img(raw_Ax,
                                            raw_By,
                                            visual_images['By_flow'],
                                            1-visual_images['By_mask'])
            #the mask here is wrong
            visuals['img_with_flow'] = pixutils.tensor2im(raw_img_with_flow[0])
            visuals['masked_Ax'] = pixutils.tensor2im(masked_Ax[0])
            visuals['masked_By'] = pixutils.tensor2im(masked_By[0])
        visualizer.display_current_results(visuals, 0, iter)
        print("iter {}".format(iter))
        iter += 1
