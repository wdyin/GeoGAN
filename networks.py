from models import *


class FlowGenerator(nn.Module):
    def __init__(self, opt):
        '''
        use skip connection for flow and mask generation
        '''
        super(FlowGenerator, self).__init__()
        self.flow_grid_np = np.mgrid[-1:1 + opt.eps:2 / (opt.img_size - 1),
                            -1:1 + opt.eps:2 / (opt.img_size - 1)]

        if opt.norm_layer == 'batchnorm':
            self.norm_layer = nn.BatchNorm2d
        elif opt.norm_layer == 'instancenorm':
            self.norm_layer = nn.InstanceNorm2d
        elif opt.norm_layer == 'layernorm':
            self.norm_layer = LayerNorm
        else:
            raise Exception("Unrecognized Norm Layer:{}".format(opt.norm_layer))

        self.res_head = SkipConnHead(3 + 1, nf=opt.ngf,
                                     n_downsample=opt.n_downsample,
                                     norm_layer=self.norm_layer)
        self.res_block = nn.Sequential(*[
            ResnetBlock(opt.ngf * 2 ** opt.n_downsample,
                        padding_type="reflect",
                        norm_layer=self.norm_layer)
            for _ in range(opt.n_blocks)
        ])
        self.res_tail = SkipConnTail(3, nf=opt.ngf,
                                     n_upsample=opt.n_downsample,
                                     norm_layer=self.norm_layer,
                                     skip_factor=2)
        self.res_flow_tail = SkipConnTail(2,nf = opt.ngf,
                                          n_upsample=opt.n_downsample,
                                          norm_layer=self.norm_layer,
                                          skip_factor=2)
        self.res_blend_tail = SkipConnTail(1,nf=opt.ngf,
                                          n_upsample=opt.n_downsample,
                                          norm_layer=self.norm_layer,
                                          skip_factor=2)
        self.flow_head = SkipConnHead(3, nf=opt.ngf,
                                      n_downsample=opt.n_downsample,
                                      norm_layer=self.norm_layer)
        self.mask_head = SkipConnHead(3, nf=opt.ngf,
                                      n_downsample=opt.n_downsample,
                                      norm_layer=self.norm_layer)
        #self.mask_head = self.flow_head
        self.flow_block = [
            nn.Conv2d(opt.ngf * 2 ** (opt.n_downsample+1),
                      opt.ngf * 2 ** opt.n_downsample,
                      kernel_size=3,
                      padding=1),
            self.norm_layer(opt.ngf * 2 ** opt.n_downsample),
            nn.ReLU(True)
        ]
        self.flow_block += [
            ResnetBlock(opt.ngf * 2 ** (opt.n_downsample),
                        padding_type='reflect',
                        norm_layer=self.norm_layer)
            for _ in range(opt.n_blocks)
        ]
        self.flow_block = nn.Sequential(*self.flow_block)
        self.mask_block = [
            nn.Conv2d(opt.ngf * 2 ** (opt.n_downsample+1),
                      opt.ngf * 2 ** opt.n_downsample,
                      kernel_size=3,
                      padding=1),
            self.norm_layer(opt.ngf * 2 ** opt.n_downsample),
            nn.ReLU(True)
        ]
        self.mask_block += [
            ResnetBlock(opt.ngf * 2 ** (opt.n_downsample),
                        padding_type='reflect',
                        norm_layer=self.norm_layer)
            for _ in range(opt.n_blocks)
        ]
        self.mask_block = nn.Sequential(*self.mask_block)
        self.flow_tail_Ax = SkipConnTail(2, nf=opt.ngf,
                                      n_upsample=opt.n_downsample,
                                      norm_layer=self.norm_layer,
                                      skip_factor=3)
        self.flow_tail_By = SkipConnTail(2, nf=opt.ngf,
                                      n_upsample=opt.n_downsample,
                                      norm_layer=self.norm_layer,
                                      skip_factor=3)
        self.flow_mask_tail = SkipConnTail(1, nf=opt.ngf,
                                      n_upsample=opt.n_downsample,
                                      norm_layer=self.norm_layer,
                                      skip_factor=3)
        self.flow_res_head = SkipConnHead(4,nf=opt.ngf,
                                          n_downsample=opt.n_downsample,
                                          norm_layer=self.norm_layer)
        self.flow_res_tail = SkipConnTail(3,nf=opt.ngf,
                                          n_upsample=opt.n_downsample,
                                          norm_layer=self.norm_layer,
                                          skip_factor=2)
        if opt.only_res:
            self.only_res_tail = SkipConnTail(3,nf=opt.ngf,
                                              n_upsample=opt.n_downsample,
                                              norm_layer=self.norm_layer,
                                              skip_factor=3)
        # self.flow_gen = FlowNet(opt,2+2)
        # self.masknet = SkipConnTail(output_nc=1,n_upsample=opt.n_downsample,
        #                            norm_layer=self.norm_layer,nf=opt.ngf)
        self.opt = opt
        self.return_names = [
            'fake_Ax',
            'fake_By',
            'fake_Ay',
            'fake_Bx',
            'flow_pairs',
            'masks',
            'By_flow',
            'By_mask',
            'Ax',
            'By',
            'By_warpped',
            'Ax_res',
            'Ay_nores'
        ]

    def expand_label_like(self, label, tensor):
        return (label.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(*label.size(),
                        tensor.size(-1), tensor.size(-1))
                )

    def remove_attr(self, fake_im, label,epoch,mask):
        skip_features = self.res_head(torch.cat([fake_im, mask], dim=1))
        res_out = self.res_block(skip_features[-1])
        _,res_im = self.res_tail(res_out, skip_features)
        #res_fake_im = torch.clamp(fake_im + 2 * F.tanh(res_im) * mask, -1, 1)
        res_fake_im = torch.clamp(fake_im + 2 * F.tanh(res_im), -1, 1)
        ret_im = res_fake_im
        return ret_im

    def warp_image_with_flow(self, image, flow,debug=False):
        device_id = image.data.get_device()
        flow_grid = Variable(torch.from_numpy(self.flow_grid_np)
                             .type(torch.FloatTensor)
                             .unsqueeze(0)
                             .cuda(device_id), requires_grad=False)
        img_grid = flow_grid.permute(0,1,3,2) + flow
        img_grid = img_grid.permute(0,2,3,1)
        warpped_img = F.grid_sample(image, img_grid)
        return warpped_img

    @staticmethod
    def stn_transform(stn_out, img):
        theta = stn_out.view(-1, 2, 3)
        if theta.size(0) ==1:
            theta = theta.expand(img.size(0),2,3)
        grid = F.affine_grid(theta, img.size())
        transformed = F.grid_sample(img, grid)
        return transformed

    def fuse_skip_feat(self, Ax_feats, By_feats):
        feats = [torch.cat([Ax_feat, By_feat], dim=1)
                 for Ax_feat, By_feat in zip(Ax_feats, By_feats)]
        return feats


    def add_attr(self, Ax, By, label_Ax, label_By, epoch):
        dev = Ax.get_device()
        Ax_skip_feats = self.flow_head(Ax)
        By_skip_feats = self.flow_head(By)
        skip_feats = self.fuse_skip_feat(Ax_skip_feats, By_skip_feats)
        resout = self.flow_block(skip_feats[-1])
        _,By_flow = self.flow_tail_By(resout,skip_feats)
        if self.opt.only_res:
            _,res_fake_im = self.only_res_tail(resout,skip_feats)
            res_fake_im = 2 * F.tanh(res_fake_im)
            ret_im = res_fake_im + Ax
            return ret_im,(torch.zeros_like(By_flow).to(dev),Ax),(torch.zeros_like(By_flow).to(dev),By),\
                torch.zeros(By.size(0),1,By.size(2),By.size(3)).to(dev),\
                torch.zeros_like(By).to(dev),torch.zeros_like(By).to(dev)

        By_flow = By_flow /10

        warpped_By = self.warp_image_with_flow(By, By_flow)
        mask_Axfeat = self.mask_head(Ax)
        mask_Byfeat = self.mask_head(By)
        mask_feats = self.fuse_skip_feat(mask_Axfeat,mask_Byfeat)
        resout = self.mask_block(mask_feats[-1])
        By_mask = F.sigmoid(self.flow_mask_tail(resout, mask_feats)[1])
        res_weight = min(self.opt.res_weight, 0.1 * max(epoch - 10, 0))

        flow_fake_im = Ax * (1-By_mask) + warpped_By * By_mask
        mask_input = By_mask.detach()
        res_skip_feats = self.flow_res_head(torch.cat([flow_fake_im, mask_input], dim=1))
        _, res_fake_im = self.flow_res_tail(res_skip_feats[-1],
                                            res_skip_feats)
        res_fake_im = 2 * F.tanh(res_fake_im) * res_weight * By_mask
        raw_By_mask = self.warp_image_with_flow(By_mask,-By_flow)
        ret_im = torch.clamp(flow_fake_im + res_fake_im, -1, 1)
        return ret_im, (torch.zeros_like(By_flow).to(dev), Ax), (By_flow, By),\
               raw_By_mask,res_fake_im,warpped_By

    def forward(self, Ax, label_Ax, By, label_By,use_blend,trans=None):
        return_items = {}
        self.grads={}
        fake_Ay, Ax_flow_pair, By_flow_pair, By_mask,Ax_res,warpped_By = self.add_attr(
            Ax, By, label_Ax,
            label_By,use_blend
        )
        fake_Bx = self.remove_attr(By, label_Ax,use_blend,By_mask)
        if trans is not None:
            fake_Bx = trans(fake_Bx)
        fake_By, fake_Bx_flow_pair, \
        fake_Ay_flow_pair, Ax_mask,_,_ = self.add_attr(
            fake_Bx, fake_Ay,
            label_Ax, label_By,
            use_blend
        )
        fake_Ax = self.remove_attr(fake_Ay, label_Ax, use_blend,Ax_mask)

        return_items['fake_Ay'] = fake_Ay
        return_items['fake_Ax'] = fake_Ax
        return_items['fake_By'] = fake_By
        return_items['fake_Bx'] = fake_Bx
        return_items['flow_pairs'] = [By_flow_pair, Ax_flow_pair,
                                      fake_Ay_flow_pair, fake_Bx_flow_pair]
        return_items['masks'] = [Ax_mask, By_mask]
        return_items['Ax_flow'] = Ax_flow_pair[0]
        return_items['By_flow'] = By_flow_pair[0]
        return_items['By_mask'] = By_mask
        return_items['Ax'] = Ax
        return_items['By'] = By
        return_items['Ax_res'] = Ax_res
        return_items['Ay_nores'] = fake_Ay - Ax_res
        return_items['By_warpped'] = warpped_By
        #for item in return_items:
        #    if type(return_items[item]) == Variable and return_items[item].requires_grad:
        #        return_items[item].retain_grad()
        return [return_items[name] for name in self.return_names]


class AttrMultiScalePatchDisc(nn.Module):
    def __init__(self, opt, input_nc):
        '''
        use feature matching loss and multiscale patch discrimination loss.
        Refer to the code of pix2pixHD for more details.
        '''
        super(AttrMultiScalePatchDisc, self).__init__()
        if opt.norm_layer == 'batchnorm':
            self.norm_layer = nn.BatchNorm2d
        elif opt.norm_layer == 'instancenorm':
            self.norm_layer = nn.InstanceNorm2d
        elif opt.norm_layer == 'layernorm':
            self.norm_layer = LayerNorm
        else:
            raise Exception("Unrecognized Norm Layer:{}".format(opt.norm_layer))
        for rank in range(opt.num_scale):
            disc = SinglePatchDisc(input_nc=input_nc, ndf=opt.ndf, use_sigmoid=False,
                                   n_layers=opt.n_layers_D, norm_layer=self.norm_layer,
                                   nc=opt.nc)
            setattr(self, 'disc_{}'.format(rank), disc)
        self.opt = opt
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)

    def forward(self, inputs):
        features, discrims = [], []
        attr_outs = []
        image_downsampled = inputs
        for i in range(self.opt.num_scale):
            disc = getattr(self, 'disc_{}'.format(i))
            feat, out, attr_out = disc(image_downsampled)
            features.append(feat)
            discrims.append(out)
            attr_outs.append(attr_out)
            image_downsampled = self.downsample(image_downsampled)
        return features, discrims, attr_outs


class LandmarkRegLoss(nn.Module):

    def __init__(self):
        super(LandmarkRegLoss,self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, source_landmark,target_landmark,flow_field):
        '''

        :param source_landmark: attr image,[-1,1]
        :param target_landmark: id image,[-1,1]
        :param flow_field: flow on attr image,[delta on -1,1]
        :return:
        '''
        source_landmark.unsqueeze_(1)
        target_landmark.unsqueeze_(1)
        sampled_flow = F.grid_sample(flow_field,target_landmark)
        target_flow = (source_landmark-target_landmark).permute(0,3,1,2)
        loss = 256*self.mse(sampled_flow,target_flow)
        return loss
