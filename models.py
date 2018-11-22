import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torchvision import models


class SkipConnHead(nn.Module):
    def __init__(self, input_nc, n_downsample, nf, norm_layer):
        super(SkipConnHead, self).__init__()
        self.layers = []
        self.layers += nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, nf, kernel_size=7),
            norm_layer(nf),
            nn.ReLU()
        )
        for i in range(n_downsample):
            self.layers += nn.Sequential(
                nn.Conv2d(nf * 2 ** i, nf * 2 ** (i + 1),
                          kernel_size=3, stride=2, padding=1),
                norm_layer(nf * 2 ** (i + 1)),
                nn.ReLU()
            )
        self.layers = nn.ModuleList(self.layers)
        self.n_downsample = n_downsample

    def forward(self, image):
        result = []
        out = image
        for layer in self.layers:
            out = layer(out)
            if type(layer) == nn.ReLU and len(result) <= (self.n_downsample + 1):
                result.append(out)
        return result


class SkipConnTail(nn.Module):
    def __init__(self, output_nc, n_upsample,
                 nf, norm_layer, skip_factor=2):
        '''

        :param output_nc:
        :param n_upsample: equal to head
        :param nf: equal to head
        '''
        super(SkipConnTail, self).__init__()
        self.layers = []
        self.skip_factor = skip_factor
        for i in range(n_upsample):
            layer = nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                  nn.Conv2d(nf * 2 ** (n_upsample - i) * skip_factor,
                                            nf * 2 ** (n_upsample - i - 1),
                                            kernel_size=1, padding=0),
                                  nn.ReflectionPad2d(1),
                                  nn.Conv2d(nf*2**(n_upsample-i-1),
                                            nf*2**(n_upsample-i-1),
                                            kernel_size=3,padding=0),
                                  norm_layer(nf * 2 ** (n_upsample - i - 1)),
                                  nn.ReLU())
            # layer = nn.Sequential(
            #    nn.ConvTranspose2d(nf*2**(n_upsample-i)*2,
            #                       nf*2**(n_upsample-i-1),
            #                       stride=2,
            #                       kernel_size=4,
            #                       padding=1),
            #    norm_layer(nf*2**(n_upsample-i-1)),
            #    nn.ReLU()
            # )
            setattr(self, 'layer_{}'.format(i), layer)
        self.output = nn.Sequential(nn.Conv2d(skip_factor * nf, output_nc, kernel_size=7, padding=3))

    def forward(self, input_feature, skip_features):
        out = [input_feature]
        for idx, skip_feature in enumerate(reversed(skip_features[1:])):
            layer = getattr(self, 'layer_{}'.format(idx))
            if self.skip_factor > 1:
                out.append(layer(torch.cat([out[-1], skip_feature], dim=1)))
            else:
                out.append(layer(out[-1]))
        if self.skip_factor > 1:
            result = self.output(torch.cat([out[-1], skip_features[0]], dim=1))
        else:
            result = self.output(out[-1])
        return reversed(out),result


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class SinglePatchDisc(nn.Module):
    def __init__(self, n_layers, input_nc, ndf,
                 use_sigmoid, norm_layer,
                 nc, use_spec=True):
        super(SinglePatchDisc, self).__init__()
        layers = [nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1),
                  nn.LeakyReLU(0.2, True)]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers += [
                nn.Conv2d(nf_prev, nf, kernel_size=3, stride=2, padding=1),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]
        nf_prev = nf
        nf = min(2 * nf, 512)
        self.dilated = True
        if self.dilated:
            for dilate_rate in [2,4,6]:
                setattr(self,'dilate_layer_{}'.format(dilate_rate),
                        nn.Conv2d(nf_prev,nf,3,stride=1,dilation=dilate_rate,padding=dilate_rate))
            setattr(self,'dilate_concat',
                    nn.Conv2d(3*nf,nf,kernel_size=1))
        else:
            layers += [nn.Conv2d(nf_prev, nf, kernel_size=3, stride=1, padding=1),
                   norm_layer(nf),
                   nn.LeakyReLU(0.2, True)]
        self.disc_layer = nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding=1)
        self.attr_layer = nn.Sequential(
            nn.Conv2d(nf, nc, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid())

        self.d_layers = nn.ModuleList(layers)

    def forward(self, image):
        out = image
        feat = []
        for layer in self.d_layers:
            out = layer(out)
            if type(layer) == nn.LeakyReLU:
                feat.append(out)
        if self.dilated:
            dilate_outs = []
            for dilate_rate in [2,4,6]:
                layer = getattr(self,'dilate_layer_{}'.format(dilate_rate))
                dilate_outs.append(layer(out))
            out = self.dilate_concat(torch.cat(dilate_outs,dim=1 ))
        disc_out = self.disc_layer(out)
        attr_out = self.attr_layer(out)
        return feat, disc_out, attr_out


class FaceVgg16(nn.Module):
    def __init__(self):
        super(FaceVgg16, self).__init__()
        try:
            from vgg_face_torch.converted_facevgg import VGGface
            self.pretrained_facevgg = VGGface
            model_state_dict = torch.load("../vgg_face_torch/converted_facevgg.pth")
            self.pretrained_facevgg.load_state_dict(model_state_dict)
        except:
            # raise Exception("FaceVGG failed to import")
            print("facevgg not imported")
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), self.pretrained_facevgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), self.pretrained_facevgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), self.pretrained_facevgg[x])
        for x in range(12, 19):
            self.slice4.add_module(str(x), self.pretrained_facevgg[x])
        for x in range(19, 26):
            self.slice5.add_module(str(x), self.pretrained_facevgg[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = FaceVgg16()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                pred = input_i
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class GPGANLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(GPGANLoss, self).__init__()

    def __call__(self, input, target_is_real):
        flag = 1
        if target_is_real:
            flag = -1
        else:
            flag = 1
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                loss += flag * torch.mean(input_i)
            return loss
        else:
            return flag * torch.mean(input)


class GPLoss(nn.Module):
    def __init__(self, tensor=torch.cuda.FloatTensor):
        super(GPLoss, self).__init__()
        self.tensor = tensor

    def gradient_penalty(self, y, x, dtype=torch.FloatTensor):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).type(dtype)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def __call__(self, fake_im, real_im, disc):
        dev_id = fake_im.data.get_device()
        alpha = torch.rand(fake_im.size(0), 1, 1, 1).cuda(dev_id)
        x_hat = Variable(alpha * real_im.data + (1 - alpha) * fake_im.data, requires_grad=True)
        _, pred = disc(x_hat)
        gp = 0
        for pred_i in pred:
            gp += self.gradient_penalty(torch.mean(pred_i), x_hat, self.tensor)
        return gp


class TVLoss(nn.Module):
    def __init__(self, eps=1e-3, beta=2):
        super(TVLoss, self).__init__()
        self.eps = eps
        self.beta = beta

    def forward(self, input):
        x_diff = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
        y_diff = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]

        sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, self.eps, 10000000)
        return torch.mean(sq_diff)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features).view(1, -1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(features).view(1, -1, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FlowRegLoss(nn.Module):
    def __init__(self, opt):
        super(FlowRegLoss, self).__init__()
        patch_size = opt.patch_size
        assert patch_size % 2 == 1
        self.opt = opt
        self.lambda_color = opt.lambda_color
        self.padding = (opt.patch_size - 1) // 2
        weights = (np.diag(np.ones(patch_size * patch_size))
                   .reshape(patch_size * patch_size, 1, 1, patch_size, patch_size)
                   .astype(np.float32))
        self.weights = Variable(torch.from_numpy(weights).cuda(), requires_grad=False)
        spatial_grid_np = np.mgrid[-1:1 + opt.eps:2 / (opt.img_size - 1),
                          -1:1 + opt.eps:2 / (opt.img_size - 1)]
        self.spatial_grid = Variable(torch.FloatTensor(spatial_grid_np.reshape(1, 1, 2,
                                                                               opt.img_size, opt.img_size)).cuda(),
                                     requires_grad=False)

    def forward(self, flow, image):
        flow *=self.opt.img_size
        flow = flow.unsqueeze(1)
        image = image.unsqueeze(1)
        device_id = flow.data.get_device()
        self.weights = self.weights.cuda(device_id)
        self.spatial_grid = self.spatial_grid.cuda(device_id)
        flow_neighbors = F.conv3d(flow, self.weights, padding=(0, self.padding, self.padding))
        image_neighbors = F.conv3d(image, self.weights, padding=(0, self.padding, self.padding))
        spatial_neighbors = F.conv3d(self.spatial_grid, self.weights, padding=(0, self.padding, self.padding))
        im_weights = torch.sum((image - image_neighbors) ** 2, dim=2)
        spatial_weights = torch.sum((self.spatial_grid - spatial_neighbors) ** 2, dim=2)
        flow_loss = torch.sum((flow_neighbors - flow) ** 2, dim=2)
        total_loss = torch.mean(flow_loss * torch.exp(-(im_weights * self.opt.lambda_color + spatial_weights)))
        return total_loss
