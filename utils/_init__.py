import os

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import *
from utils.reg_utils import *
from utils.sim_utils import *


def print_options(opt, parser, dir):
    """Print and save options"""
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(dir, '{}_opts.txt'.format(opt.exp_name))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')


def print_msg_to_file(msg, dir, opt):
    file_name = os.path.join(dir, '{}_running.txt'.format(opt.exp_name))
    with open(file_name, 'a+') as opt_file:
        opt_file.write(msg)
        opt_file.write('\n')


def comput_fig(img, cmap='gray', colors=None):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        if colors is not None:
            plt.imshow(colors[img[i, :, :].astype(int)])
        else:
            plt.imshow(img[i, :, :], cmap=cmap)
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz, dtype=np.float32)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def random_colors(num=255):
    color_array = np.zeros((num + 1, 3))
    color_array[1:, :] = np.random.rand(num, 3)
    return color_array


class AverageMeter():
    def __init__(self, opts, mov_img, fix_img, mov_seg, fix_seg, alpha, beta, img_warper, dist_type='NCC'):
        self.mov_img = mov_img
        self.fix_img = fix_img
        self.mov_seg = mov_seg
        self.fix_seg = fix_seg
        self.alpha = alpha
        self.beta = beta
        self.img_warper = img_warper

        self.grid_img = mk_grid_img(8, 1, opts.img_size)
        self.colors = random_colors(255)

        self.logger = SummaryWriter(log_dir='logs/' + opts.exp_name)

        if dist_type == 'SSIM':
            self.dist_loss = SSIM()
        elif dist_type == 'SSD':
            self.dist_loss = SSD()
        elif dist_type == 'NCC':
            self.dist_loss = NCC()
        elif dist_type == 'MI':
            self.dist_loss = MutualInformation()
        elif dist_type == 'MIND':
            self.dist_loss = MIND()
        elif dist_type == 'local MI':
            self.dist_loss = Localmutualinformation()

        self.grad_loss = SpatialGrad(penalty='l2')

    def print_metrics(self, i, dsp_mov_self, dsp_fix_self, dsp_fw, dsp_bw, loss1=0, loss2=0, eps=0, norm1=0, norm2=0,
                      norm3=0):
        with torch.no_grad():
            mov_seg_def = self.img_warper(self.mov_seg.float(), dsp_fw, mode='nearest')
            mov_img_def = self.img_warper(self.mov_img, dsp_fw, mode='bilinear')
            mov_seg_self = self.img_warper(self.mov_seg.float(), dsp_mov_self, mode='nearest')

            dice = dice_IXI(mov_seg_def.long(), self.fix_seg.long())
            self_dice = dice_IXI(mov_seg_self.long(), self.mov_seg.long())

            dist = self.dist_loss(self.fix_img, mov_img_def)
            grad_loss = self.grad_loss(dsp_fw)
            jet, folds = compute_jacobi_and_folds(dsp_fw + self.img_warper.grid)  # transformation
            self_sanity = torch.mean(dsp_mov_self ** 2) + torch.mean(dsp_fix_self ** 2)
            cross_sanity = cross_sanity_error(dsp_fw, dsp_bw, self.alpha, self.beta, self.img_warper)

        if i == -1:
            print(
                "[Metrcis] -\t Dice = [\u2191],\t Self_Dice = [\u2191],\t Image_Distance = [\u2193],\t Reg_Grad = [\u2193],\t Loss1 = [\u2193],\t Folds = [\u2193],\t Abs Jacobian = [\u2193],\t Self_Sanity_Erorr = [\u2193],\t Cross_Sanity_Erorr = [\u2193],\t Loss2 = [\u2193]")

        print(
            "i = {:03d},\t Dice = {:.3f},\t Self_Dice = {:.3f},\t Image_Distance = {:.3f},\t Reg_Grad = {:.3f},\t Loss1 = {:.3f},\t Folds = {:.3f},\t Abs Jacobian = {:.3f},\t Self_Sanity_Erorr = {:.3f},\t Cross_Sanity_Erorr = {:.3f},\t Loss2 = {:.3f},\t Eps = {:.3f},\t Norm1 = {:.3f},\t Norm2 = {:.3f},\t Norm3 = {:.3f}".format(
                i + 1,
                dice * 100,
                self_dice * 100,
                dist,
                grad_loss,
                loss1,
                folds,
                jet,
                self_sanity,
                cross_sanity,
                loss2,
                eps,
                norm1,
                norm2,
                norm3
            ))

        self.logger.add_scalar('test/loss1', loss1, i)
        self.logger.add_scalar('test/loss2', loss2, i)

        self.logger.add_scalar('test/image_dist', dist, i)
        self.logger.add_scalar('test/grad_loss', grad_loss, i)

        self.logger.add_scalar('test/dice', dice * 100, i)
        self.logger.add_scalar('test/self_dice', self_dice * 100, i)
        self.logger.add_scalar('test/folds', folds, i)
        self.logger.add_scalar('test/abs_jab', jet, i)

        self.logger.add_scalar('test/self_sanity', self_sanity, i)
        self.logger.add_scalar('test/cross_sanity', cross_sanity, i)

        self.logger.add_scalar('monitor/eps', eps, i)

        self.logger.add_scalar('monitor/norm1', norm1, i)
        self.logger.add_scalar('monitor/norm2', norm2, i)
        self.logger.add_scalar('monitor/norm3', norm3, i)

        # if i % 20 == 0:
        #     dsp_x2f_fig = comput_fig(self.img_warper(self.grid_img, dsp_fw))
        #     dsp_f2x_fig = comput_fig(self.img_warper(self.grid_img, dsp_bw))
        #     metric_meter.logger.add_figure('dsp/atlas_to_sub', dsp_x2f_fig, i)
        #     metric_meter.logger.add_figure('dsp/sub_to_atlas', dsp_f2x_fig, i)

        #     dsp_mov_self_fig = comput_fig(dsp_mov_self, cmap='viridis')
        #     dsp_fix_self_fig = comput_fig(dsp_fix_self, cmap='viridis')
        #     metric_meter.logger.add_figure('dsp/atlas_to_atlas', dsp_mov_self_fig, i)
        #     metric_meter.logger.add_figure('dsp/sub_to_sub', dsp_fix_self_fig, i)

        #     mov_seg_def_fig = comput_fig(mov_seg_def, colors=self.colors)
        #     fix_seg_fig = comput_fig(self.fix_seg, colors=self.colors)
        #     metric_meter.logger.add_figure('seg/mov_seg_def', mov_seg_def_fig, i)
        #     metric_meter.logger.add_figure('seg/fix_seg', fix_seg_fig, i)
