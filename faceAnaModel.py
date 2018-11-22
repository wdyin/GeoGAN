import torch
from torch import nn,optim

from networks import *
import itertools as it

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if 'weight' not in dir(m):
            return
        m.weight.data.normal_(1.0, 0.01)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


class FaceAnalogyModel(nn.Module):
    def __init__(self, opt):
        super(FaceAnalogyModel, self).__init__()
        self.generator = FlowGenerator(opt)
        self.disc = AttrMultiScalePatchDisc(opt, 3)
        self.crit_gan = GANLoss(use_lsgan=opt.use_lsgan)
        self.crit_gp = GPLoss()
        self.crit_Feat = nn.L1Loss()
        self.crit_landmark = LandmarkRegLoss()
        self.loss_names = ['GAN_loss', 'GAN_Feat_loss',
                           'G_cls', 'G_flow_loss', 'G_mask_loss',
                           'D_real', 'D_fake', 'D_cls',
                           'G_landmark_loss']
        if opt.train:
            self.visualizer_names = [
                'Ax', 'By',
                'fake_Ay',
                'fake_Bx',
                'Ax_res','By_mask',
                'Ay_nores','By_warpped'
            ]
        else:
            self.visualizer_names = [
                'Ax', 'By',
                'fake_Ay', 'fake_Bx',
                'By_flow',
                'By_mask','Ax_res',
                'By_warpped',
                'Ay_nores'
            ]
        only_res = False
        if not only_res:
            self.optimizer_G = optim.Adam(self.generator.parameters(), lr=opt.lr)
        else:
            self.optimizer_G = optim.Adam(it.chain(self.generator.flow_res_head.parameters(),
                                          self.generator.flow_res_tail.parameters()))
        self.optimizer_D = optim.Adam(filter(lambda p: p.requires_grad,
                                             self.disc.parameters()), lr=opt.lr)

        self.opt = opt
        self.crit_tv = TVLoss()
        self.crit_flow = FlowRegLoss(opt)
        self.crit_attr = nn.BCELoss()
        # self.apply(weights_init)

    def flip_tensor(self,tensor):
        inv_idx = torch.arange(tensor.size(3) - 1, -1, -1).long().to(tensor.get_device())
        # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
        inv_tensor = tensor[:,:,:,inv_idx]
        return inv_tensor

    def forward(self, Ax, label_Ax,landmark_Ax,
                By, label_By,landmark_By,
                epoch, visual=False):
        dev_id = Ax.get_device()
        loss_dict = {}
        label_org = label_Ax.unsqueeze(-1).unsqueeze(-1)
        label_trg = label_By.unsqueeze(-1).unsqueeze(-1)
        label_org = label_org.expand(*label_Ax.size(),
                                     self.opt.img_size,
                                     self.opt.img_size)
        label_trg = label_trg.expand(*label_By.size(),
                                     self.opt.img_size,
                                     self.opt.img_size)
        returned_items = self.generator(Ax, label_Ax,
                                        By, label_By,
                                        epoch,
                                        trans=None)
        returned_items = dict(zip(self.generator.return_names,
                                  returned_items))
        ## loss for D5
        for loss_name in self.loss_names:
            loss_dict.setdefault(loss_name, 0)
        for item_name in returned_items:
            if 'fake' not in item_name:
                continue
            if 'x' in item_name:
                label = label_org
            else:
                label = label_trg

            _, pred, attr_pred = self.disc(returned_items[item_name])
            loss_dict['D_fake'] += 0.5 * self.crit_gan(pred, False)
            loss_dict['GAN_loss'] += self.crit_gan(pred, True)
            if item_name == 'fake_Ax':
                real_im, label = Ax, label_Ax
            elif item_name == 'fake_Ay':
                real_im, label = Ax, label_By
            elif item_name == 'fake_Bx':
                real_im, label = By, label_Ax
            else:
                real_im, label = By, label_By
            for attr_pred_i in attr_pred:
                loss_dict['G_cls'] += self.opt.lambda_cls * self.crit_attr(attr_pred_i, label)

        _, pred, attr_pred = self.disc(Ax)
        loss_dict['D_real'] += self.crit_gan(pred, True)
        for attr_pred_i in attr_pred:
            loss_dict['D_cls'] += self.opt.lambda_cls * self.crit_attr(attr_pred_i, label_Ax)
        _, pred, attr_pred = self.disc(By)
        loss_dict['D_real'] += self.crit_gan(pred, True)
        for attr_pred_i in attr_pred:
            loss_dict['D_cls'] += self.opt.lambda_cls * self.crit_attr(attr_pred_i, label_By)

        loss_dict.setdefault('G_flow_loss', 0)
        for flow, img in returned_items['flow_pairs']:
            loss_dict['G_flow_loss'] += self.crit_tv(flow*self.opt.img_size) * self.opt.lambda_flow_reg

        loss_dict.setdefault('G_mask_loss', 0)
        for mask in returned_items['masks']:
            loss_dict['G_mask_loss'] += self.crit_Feat(mask, torch.zeros_like(mask)) \
                                        * self.opt.lambda_mask
        if self.opt.train:
            loss_dict['G_landmark_loss'] = 10*self.crit_landmark(landmark_By,
                                                             landmark_Ax,
                                                              returned_items['By_flow']
                                                             )
        def img_recon_loss(fake_im, real_im, label, disc):
            recon_loss = self.crit_Feat(fake_im, real_im) * self.opt.lambda_gan_feat
            return recon_loss
        recon_By = By
        recon_tuple = [
            (returned_items['fake_Ax'], Ax, label_org),
            (returned_items['fake_By'], recon_By, label_trg)
        ]
        for fake_img, real_img, label in recon_tuple:
            gan_feat = img_recon_loss(fake_img, real_img,
                                                label, self.disc)
            loss_dict.setdefault('GAN_Feat_loss', 0)
            loss_dict.setdefault('FaceVGG_loss', 0)
            loss_dict['GAN_Feat_loss'] += gan_feat

        zero_var = torch.zeros(1,requires_grad=False).to(Ax.get_device())
        if visual:
            visual_items = [returned_items[name]
                            for name in self.visualizer_names]
        else:
            visual_items = zero_var

        for name in loss_dict:
            if loss_dict[name] == 0:
                loss_dict[name] = zero_var
            else:
                loss_dict[name] = loss_dict[name].unsqueeze(0)
        return [[loss_dict[name] for name in self.loss_names], visual_items]

    def save(self, root_dir, epoch, iter):
        torch.save(self.generator.state_dict(), '{}/epoch_{}_iter_{}_G.pth'.format(root_dir, epoch, iter))
        torch.save(self.disc.state_dict(), '{}/epoch_{}_iter_{}_D.pth'.format(root_dir, epoch, iter))

    def load(self, root_dir, epoch, iter):
        map_location = {'cuda:{}'.format(i):'cuda:0' for i in range(8)}
        generator = torch.load('{}/epoch_{}_iter_{}_G.pth'.format(root_dir, epoch, iter),map_location=map_location)
        disc = torch.load('{}/epoch_{}_iter_{}_D.pth'.format(root_dir, epoch, iter),map_location=map_location)
        try:
            self.generator.load_state_dict(generator.state_dict())
            self.disc.load_state_dict(disc.state_dict())
            print('load state dict of generator')
        except:
            print('load state dict directly')
            self.generator.load_state_dict(generator)
            self.disc.load_state_dict(disc)
