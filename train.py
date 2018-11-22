import sys
import time

from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler

sys.path.append('..')
from faceAnaModel import FaceAnalogyModel
from options import BaseOptions
from face_dataset import *
from util.visualizer import Visualizer
from util import util as pixutils
from torch.autograd import grad

opt = BaseOptions().parse()
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.max_dataset_size = 10
    opt.batch_size = 1
    opt.gpu_ids = [0]

dataloader = create_dataloader_celeba(opt)

model = FaceAnalogyModel(opt).cuda()
if opt.which_epoch is not None:
    model.load(os.path.join(opt.checkpoints_dir, opt.name),
                      opt.which_epoch,
                      opt.which_epoch_iter)
    print('model {} loaded.'.format(opt.which_epoch))
model = nn.DataParallel(model, device_ids=opt.gpu_ids)
visualizer = Visualizer(opt)

total_steps = 0
d_scheduler = lr_scheduler.LambdaLR(model.module.optimizer_D,
                                    lambda epoch: 0.2 ** (epoch // 20))
g_scheduler = lr_scheduler.LambdaLR(model.module.optimizer_G,
                                    lambda epoch: 0.4 ** (epoch // 20))
print(model)
for epoch in range(opt.num_epoch):
    epoch_iter = 0
    # opt.lambda_cls *= epoch // 20
    d_scheduler.step(epoch)
    g_scheduler.step(epoch)
    for Ax, label_Ax,landmark_Ax, By, label_By,landmark_By in dataloader:
        Ax = Variable(Ax.cuda())
        By = Variable(By.cuda())
        label_Ax = Variable(label_Ax.cuda())
        label_By = Variable(label_By.cuda())
        landmark_Ax = Variable(landmark_Ax.cuda())
        landmark_By = Variable(landmark_By.cuda())
        iter_start_time = time.time()
        total_steps += 1
        epoch_iter += 1
        is_visual = False
        if total_steps % opt.display_freq == 0 or total_steps % opt.print_freq == 0:
            is_visual = True
        losses, model_vis = model(Ax, label_Ax,
                                  landmark_Ax,
                                      By, label_By,
                                  landmark_By,
                                      epoch,
                                      is_visual)
        losses = [torch.mean(loss) for loss in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))
        loss_D = loss_dict['D_real'] + loss_dict['D_fake'] + loss_dict['D_cls']
        loss_G = loss_dict['GAN_loss'] + loss_dict['GAN_Feat_loss'] + loss_dict['G_cls'] \
                 + loss_dict['G_flow_loss'] + loss_dict['G_mask_loss'] + loss_dict['G_landmark_loss']

        model.module.optimizer_G.zero_grad()
        loss_G.backward(retain_graph=True)


        model.module.optimizer_G.step()

        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()
        if total_steps % opt.print_freq == 0:
            visual_images = dict(zip(model.module.visualizer_names, model_vis))
            errors = {k: v.data[0] for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)
            #for item_name in visual_images:
            #    if visual_images[item_name].requires_grad:
            #        try:
            #            print("{}:grad {}".format(item_name,visual_images[item_name].grad.norm(2).data[0]))
            #        except AttributeError:
            #            continue

        if total_steps % opt.display_freq == 0:
            visual_images = dict(zip(model.module.visualizer_names, model_vis))
            visuals = {}
            for item_name in visual_images:
                if 'flow' in item_name:
                    continue
                if 'mask' in item_name:
                    visual_images[item_name] = visual_images[item_name]*2-1
                tensor = visual_images[item_name]
                img_list = []
                for i in range(0, 1):
                    img_list.append(pixutils.tensor2im(tensor[i:i + 1].data[0]))
                visuals[item_name] = img_list
            visualizer.display_current_results(visuals, epoch, epoch_iter)


        if total_steps % opt.save_freq == 0:
            print("saving model at epoch {} step {}".format(epoch, epoch_iter))
            model.module.save(os.path.join(opt.checkpoints_dir, opt.name), epoch, epoch_iter)
