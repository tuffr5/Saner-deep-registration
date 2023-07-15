"""
    IXI Image Registration
    Authors: Duan, Bin (bduan2@hawk.iit.edu)
"""

import argparse
import glob
import random

import torch.cuda
import torch.optim as optim
from batchgenerators.utilities.file_and_folder_operations import *
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from data import datasets, trans
from utils._init__ import print_options, print_msg_to_file
from utils.metrics import *
from utils.model_utils import load_model, SpatialWarper
from utils.sim_utils import *
from utils.reg_utils import *
import torch.nn as nn


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class LossBundle():
    def __init__(self, alpha, beta, img_warper, dist_type='NCC', spatial_penalty='l2'):
        super().__init__()
        assert dist_type in ['SSIM', 'SSD', 'NCC', 'MI', 'MIND', 'local MI']
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

        self.spatial_grad = SpatialGrad(penalty=spatial_penalty)
        self.cross_sanity_loss = CrossSanityLoss(alpha, beta, img_warper)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


@torch.no_grad()
def argmax_interpolation(x_seg, flow, img_warper, num_classes=46):
    x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_classes)
    x_seg_oh = torch.squeeze(x_seg_oh, 1)
    x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
    x_segs = []
    for i in range(num_classes):
        def_seg = img_warper(x_seg_oh[:, i:i + 1, ...].float(), flow.float())
        x_segs.append(def_seg)
    x_segs = torch.cat(x_segs, dim=1)
    def_out = torch.argmax(x_segs, dim=1, keepdim=True)

    return def_out


def sanity_checks_test(register, img_warper, data, loss_bundle, flag='test'):
    mov = data[0]
    fix = data[1]
    mov_seg = data[2]
    fix_seg = data[3]

    with torch.no_grad():
        dsp_m2f = register(mov, fix)
        dsp_f2m = register(fix, mov)
        dsp_m2m = register(mov, mov)
        dsp_f2f = register(fix, fix)

        loss = {}
        # sim_loss
        loss_sim = loss_bundle.dist_loss(fix, img_warper(mov, dsp_m2f))
        loss_sim += loss_bundle.dist_loss(mov, img_warper(fix, dsp_f2m))
        loss['loss_sim'] = loss_sim

        # grad_loss
        loss_grad = loss_bundle.spatial_grad(dsp_m2f) + loss_bundle.spatial_grad(dsp_f2m)
        loss['loss_grad'] = loss_grad

        # cross_sanity
        loss_cross = loss_bundle.cross_sanity_loss(dsp_m2f, dsp_f2m)
        loss['loss_cross'] = loss_cross

        # self_sanity
        loss_self = torch.mean(dsp_m2m ** 2) + torch.mean(dsp_f2f ** 2)
        loss['loss_self'] = loss_self

        # folds and abs jacobian
        jet, folds = compute_jacobi_and_folds(dsp_m2f + img_warper.grid)  # transformation

        loss['folds'] = folds
        loss['abs_jab'] = jet

        if flag == 'test':
            # dice and self_dice
            mov_seg_def = argmax_interpolation(mov_seg, dsp_m2f, img_warper, 46)
            mov_seg_self = argmax_interpolation(mov_seg, dsp_m2m, img_warper, 46)

            # To align with transmorph, where they set specifical classes
            # dice = dice_fn(mov_seg_def.long(), fix_seg.long(), 46)
            # self_dice = dice_fn(mov_seg_self.long(), mov_seg.long(), 46)
        else:
            mov_seg_def = img_warper(mov_seg.float(), dsp_m2f, mode='nearest')
            mov_seg_self = img_warper(mov_seg.float(), dsp_m2m, mode='nearest')

        # To align with transmorph, where they set specifical classes
        dice = dice_IXI(mov_seg_def.long(), fix_seg.long())
        self_dice = dice_IXI(mov_seg_self.long(), mov_seg.long())


        loss['dice'] = dice
        loss['self_dice'] = self_dice

    return loss


def sanity_test(epoch, register, img_warper, dataloader, loss_bundle, logger, opts, flag='val'):
    register.model.eval()
    loss_dict = {}

    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            data = [t.to('cuda:0', non_blocking=True) for t in data]
            loss = sanity_checks_test(register, img_warper, data, loss_bundle, flag=flag)
            for k, v in loss.items():
                if idx == 0:
                    loss_dict[k] = []
                loss_dict[k].append(v)

        message = f'{flag}: epoch {epoch} of {opts.epoch}'
        dsc = 0.
        for k, v in loss_dict.items():
            mean_v = torch.mean(torch.FloatTensor(v))
            std_v = torch.std(torch.FloatTensor(v))
            if 'loss' in k:
                logger.add_scalar(f'sanity_{flag}/loss/{k}', mean_v, epoch)
            else:
                logger.add_scalar(f'sanity_{flag}/metric/{k}', mean_v, epoch)

            if 'dice' == k:
                dsc = mean_v
            message += f', {k}: {mean_v.item():.3f}\u00B1{std_v.item():.3f}'

        print(message)
        print_msg_to_file(message, 'logs', opts)

    # to save memory
    # torch.cuda.empty_cache()

    return dsc


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, join(save_dir, filename))
    torch.save(state, join(save_dir, 'latest_checkpoint.pth.tar'))
    model_lists = natsorted(glob.glob(save_dir + 'dsc*.pth.tar'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + 'dsc*.pth.tar'))


def sanity_checks_train(register, img_warper, data, loss_bundle, opts):
    mov = data[0]
    fix = data[1]
    dsp_m2f = register(mov, fix)
    dsp_f2m = register(fix, mov)

    loss = {}
    # sim_loss
    loss_sim = loss_bundle.dist_loss(fix, img_warper(mov, dsp_m2f))
    if opts.flag_sim:
        loss_sim += loss_bundle.dist_loss(mov, img_warper(fix, dsp_f2m))

    loss['loss_sim'] = loss_sim

    # grad_loss
    loss_grad = loss_bundle.spatial_grad(dsp_m2f)
    if opts.flag_grad:
        loss_grad += loss_bundle.spatial_grad(dsp_f2m)

    loss['loss_grad'] = loss_grad

    # cross_sanity
    if opts.flag_cross:
        loss_cross = loss_bundle.cross_sanity_loss(dsp_m2f, dsp_f2m)
        loss['loss_cross'] = loss_cross

    # self_sanity
    if opts.flag_self:
        loss_self = torch.mean(register(mov, mov) ** 2) + torch.mean(register(fix, fix) ** 2)
        loss['loss_self'] = loss_self

    return loss


def sanity_train(epoch, register, img_warper, dataloader, optimizer, loss_bundle, logger, opts):
    register.model.train()
    for idx, data in enumerate(dataloader):
        adjust_learning_rate(optimizer, epoch, opts.epoch + 1, lr)
        data = [t.to('cuda:0', non_blocking=True) for t in data]
        loss_dict = sanity_checks_train(register, img_warper, data, loss_bundle, opts)
        losses = 0.
        message = f'Iter {idx + 1} of {len(dataloader)}'
        for k, v in loss_dict.items():
            # print(v)
            if k == 'loss_cross':
                losses += opts.weight[0] * v
            elif k == 'loss_self':
                losses += opts.weight[1] * v
            else:
                losses += v

            logger.add_scalar(f'sanity_train/{k}', v, (epoch - 1) * len(dataloader) + idx)
            message += f', {k}: {v.item():.3f}'

        print(message)

        # losses = torch.stack(losses)

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()

        # just for saving memory
        # del losses
        # torch.cuda.empty_cache()

    return register, img_warper, optimizer


if __name__ == '__main__':
    # ---- input arguments ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="VoxelMorph-Sanity", help='Experiment name')
    parser.add_argument("--model_type", type=str, default="voxelmorph2", help='Model type')
    parser.add_argument("--model_dir", type=str,
                        default="/data/duanbin/image_registration/experiments/CNN/VoxelMorph/Vxm_2_ncc_1_diffusion_1/",
                        help='pretrained model file')
    parser.add_argument("--ckpt", type=str,
                        default="/data/duanbin/image_registration/experiments/CNN/VoxelMorph/sanity_checked/",
                        help='where to save the trained model file')
    parser.add_argument("--epoch", type=int, default=500, help="How many epochs to tune")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gpu_id", type=str, default='7', help="Which GPU to be used")
    parser.add_argument("--weight", type=float, nargs='+', default=[0.001, 0.1], help='loss weight for checks')
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha weight for cross-sanity check")
    parser.add_argument("--beta", type=float, default=12, help="Beta weight for cross-sanity check")
    parser.add_argument("--flag_sim", default=False, action="store_true", help="Whether add image similarity regularization")
    parser.add_argument("--flag_grad", default=False, action="store_true", help="Whether add grad regularization")
    parser.add_argument("--flag_self", default=False, action="store_true", help="Whether add self-sanity check")
    parser.add_argument("--flag_cross", default=False, action="store_true", help="Whether add cross-sanity check")
    parser.add_argument("--cont", default=False, action="store_true", help="Whether continue previous training")
    parser.add_argument("--train_on_subset", default=False, action="store_true", help="Whether training on subset of the training dataset")
    parser.add_argument("--train_from_scratch", default=False, action="store_true", help="Whether training a model from scratch")
    parser.add_argument("--testing", default=False, action="store_true", help="Whether do testing")
    opts = parser.parse_args()

    # opts.exp_name = 'Train_on_subset_' + opts.train_on_subset + 'epoch_' + str(opts.epoch) + 'lr_' + str(opts.lr) + 'weight_' + str(
    #     opts.weight) + 'alpha_' + str(opts.alpha) + 'beta_' + str(opts.beta) + 'flag_sim_' + str(
    #     opts.flag_sim) + 'flag_grad_' + str(opts.flag_grad) + 'flag_self_' + str(opts.flag_self) + 'flag_cross_' + str(
    #     opts.flag_cross)

    print(f'cross-sanity check weight: {opts.weight[0]}, self-sanity check weight: {opts.weight[1]}')

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = '7'

    opts.ckpt = join(opts.ckpt, opts.exp_name) + '/'

    # seeding
    torch.manual_seed(521)
    torch.cuda.manual_seed_all(521)
    np.random.seed(521)
    random.seed(521)

    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    opts.img_size = (160, 192, 224)
    if opts.train_on_subset:
        opts.train_dir = '/data/duanbin/image_registration/dataset/IXI_data/Train_Sub/'
    else:
        opts.train_dir = '/data/duanbin/image_registration/dataset/IXI_data/Train/'

    opts.val_dir = '/data/duanbin/image_registration/dataset/IXI_data/Val/'
    opts.test_dir = '/data/duanbin/image_registration/dataset/IXI_data/Test/'
    opts.atlas_dir = '/data/duanbin/image_registration/dataset/IXI_data/atlas.pkl'

    # print out and save parameters
    print_options(opts, parser, 'logs')

    img_warper = SpatialWarper(opts.img_size)
    img_warper.cuda()

    lr = opts.lr

    """ load the model and initialize optimizer"""
    if opts.train_from_scratch:
        register = load_model(opts.model_type, None, opts.img_size)
        print(f'Training a model from scratch!')
        optimizer = optim.Adam(register.model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
        epoch_start = 1
    else:
        if opts.cont:
            checkpoint = torch.load(opts.ckpt + natsorted(os.listdir(opts.ckpt))[-1], map_location='cuda:0')
            register = load_model(opts.model_type, checkpoint, opts.img_size)
            register.model.cuda()
            print(f'Model: {opts.ckpt + natsorted(os.listdir(opts.ckpt))[-1]} has been loaded.')
            optimizer = optim.Adam(register.model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch_start = checkpoint['epoch']
        else:
            checkpoint = torch.load(opts.model_dir + natsorted(os.listdir(opts.model_dir))[-1], map_location='cuda:0')
            register = load_model(opts.model_type, checkpoint, opts.img_size)
            print(f'Model: {opts.model_dir + natsorted(os.listdir(opts.model_dir))[-1]} has been loaded.')
            optimizer = optim.Adam(register.model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
            epoch_start = 1

    """ Loss bundle """
    loss_bundle = LossBundle(alpha=opts.alpha, beta=opts.beta, img_warper=img_warper, dist_type='NCC',
                             spatial_penalty='l2')

    """ Logger """
    logger = SummaryWriter(log_dir='logs/' + opts.exp_name)

    """ Checkpoints dir """
    maybe_mkdir_p(opts.ckpt)

    if not opts.testing:
        """ dataloader """
        train_composed = transforms.Compose([trans.RandomFlip(0),
                                             trans.NumpyType((np.float32, np.float32)),
                                             ])

        train_set = datasets.IXIBrainDataset(glob.glob(opts.train_dir + '*.pkl'), opts.atlas_dir,
                                             transforms=train_composed)
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

        val_composed = transforms.Compose([trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
                                           trans.NumpyType((np.float32, np.int16))])
        val_set = datasets.IXIBrainInferDataset(glob.glob(opts.val_dir + '*.pkl'), opts.atlas_dir,
                                                transforms=val_composed)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

        """ initial metric recording """
        sanity_test(epoch_start - 1, register, img_warper, val_loader, loss_bundle, logger, opts, flag='val')

        for epoch in range(epoch_start, opts.epoch + 1):
            """ sanity checked training """
            register, img_warper, optimizer = sanity_train(epoch, register, img_warper, train_loader, optimizer,
                                                           loss_bundle, logger, opts)

            # torch.cuda.empty_cache()

            """ sanity checked validation """
            dsc = sanity_test(epoch, register, img_warper, val_loader, loss_bundle, logger, opts, flag='val')
            # save checkpoints
            save_checkpoint({
                'epoch': epoch,
                'state_dict': register.model.state_dict(),
                'dsc': dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir=opts.ckpt, filename='dsc{:.3f}.pth.tar'.format(dsc))

        test_composed = transforms.Compose([trans.Seg_norm(),
                                            trans.NumpyType((np.float32, np.int16)),
                                            ])
        test_set = datasets.IXIBrainInferDataset(glob.glob(opts.test_dir + '*.pkl'), opts.atlas_dir,
                                                 transforms=test_composed)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

        # load initial mode and run test
        checkpoint = torch.load(opts.model_dir + natsorted(os.listdir(opts.model_dir))[-1], map_location='cuda:0')
        register = load_model(opts.model_type, checkpoint, opts.img_size)
        print(f'Model: {opts.model_dir + natsorted(os.listdir(opts.model_dir))[-1]} has been loaded.')
        sanity_test(0, register, img_warper, test_loader, loss_bundle, logger, opts, flag='test')

        # load the best_model and run test
        checkpoint = torch.load(opts.ckpt + natsorted(os.listdir(opts.ckpt))[-2], map_location='cuda:0')
        register = load_model(opts.model_type, checkpoint, opts.img_size)
        print(f'Model: {opts.ckpt + natsorted(os.listdir(opts.ckpt))[-2]} has been loaded.')
        sanity_test(checkpoint['epoch'], register, img_warper, test_loader, loss_bundle, logger, opts, flag='test')
    else:
        # torch.cuda.empty_cache()

        test_composed = transforms.Compose([trans.Seg_norm(),
                                            trans.NumpyType((np.float32, np.int16)),
                                            ])
        test_set = datasets.IXIBrainInferDataset(glob.glob(opts.test_dir + '*.pkl'), opts.atlas_dir,
                                                 transforms=test_composed)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

        # load initial mode and run test
        # checkpoint = torch.load(opts.model_dir + natsorted(os.listdir(opts.model_dir))[-1], map_location='cuda:0')
        # register = load_model(opts.model_type, checkpoint, opts.img_size)
        # print(f'Model: {opts.model_dir + natsorted(os.listdir(opts.model_dir))[-1]} has been loaded.')
        # sanity_test(0, register, img_warper, test_loader, loss_bundle, logger, opts, flag='test')

        # load the best_model and run test
        # checkpoint = torch.load(opts.ckpt + natsorted(os.listdir(opts.ckpt))[-2], map_location='cuda:0')
        checkpoint = torch.load('/data/duanbin/image_registration/experiments/gen_ckpts/VM-ESC-dsc0.734.pth.tar', map_location='cuda:0')
        register = load_model(opts.model_type, checkpoint, opts.img_size)
        # print(f'Model: {opts.ckpt + natsorted(os.listdir(opts.ckpt))[-2]} has been loaded.')
        sanity_test(checkpoint['epoch'], register, img_warper, test_loader, loss_bundle, logger, opts, flag='test')
