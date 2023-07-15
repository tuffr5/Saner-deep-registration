import argparse
import glob
import random

import numpy as np
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
import torch.nn.functional as F
from model_zoo import transformation


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

        self.dice_loss = DiceLoss() # differentiable

        self.spatial_grad = SpatialGrad(penalty=spatial_penalty)
        self.cross_sanity_loss = CrossSanityLoss(alpha, beta, img_warper)


@torch.no_grad()
def argmax_interpolation(x_seg, flow, img_warper, num_classes):
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


@torch.no_grad()
def argmax_interpolation_bspl(x_seg, flow, num_classes=46):
    x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=num_classes)
    x_seg_oh = torch.squeeze(x_seg_oh, 1)
    x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
    x_segs = []
    for i in range(num_classes):
        def_seg = transformation.warp(x_seg_oh[:, i:i + 1, ...].float(), flow.float(), interp_mode='bilinear')
        x_segs.append(def_seg)
    x_segs = torch.cat(x_segs, dim=1)
    def_out = torch.argmax(x_segs, dim=1, keepdim=True)

    return def_out


@torch.no_grad()
def sanity_checks_test_bspl(register, img_warper, data, loss_bundle, flag):
    mov = data[0]
    fix = data[1]
    mov_seg = data[2]
    fix_seg = data[3]

    flow_m2f, dsp_m2f = register(mov, fix)
    flow_f2m, dsp_f2m = register(fix, mov)
    flow_m2m, dsp_m2m = register(mov, mov)
    flow_f2f, dsp_f2f = register(fix, fix)

    loss = {}
    # sim_loss
    loss_sim = loss_bundle.dist_loss(fix, transformation.warp(mov, dsp_m2f))
    loss_sim += loss_bundle.dist_loss(mov, transformation.warp(fix, dsp_f2m))
    loss['loss_sim'] = loss_sim.item()

    # grad_loss
    loss_grad = loss_bundle.spatial_grad(flow_m2f) + loss_bundle.spatial_grad(flow_f2m)
    loss['loss_grad'] = loss_grad.item()

    # cross_sanity
    loss_cross = loss_bundle.cross_sanity_loss(flow_m2f, flow_f2m)
    loss['loss_cross'] = loss_cross.item()

    # self_sanity
    loss_self = torch.mean(flow_m2m ** 2) + torch.mean(flow_f2f ** 2)
    loss['loss_self'] = loss_self.item()

    # folds and abs jacobian
    jet, folds = compute_jacobi_and_folds(dsp_m2f + img_warper.grid)  # transformation

    loss['folds'] = folds
    loss['abs_jab'] = jet

    if flag == 'OASIS':
        def_seg = argmax_interpolation_bspl(mov_seg, dsp_m2f, 36)
        # hd95 and log_jet
        hd95 = compute_hd95_oasis(mov_seg.clone().squeeze().cpu().numpy(), def_seg.clone().squeeze().cpu().numpy(),
                                  fix_seg.clone().squeeze().cpu().numpy())
        log_std_jet = compute_std_log_jacobi_det(dsp_m2f.clone().squeeze(0).cpu().numpy())

        mdice = dice_OASIS(def_seg.long(), fix_seg.long())

        def_seg = argmax_interpolation_bspl(fix_seg, dsp_f2m, 36)
        fdice = dice_OASIS(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation_bspl(mov_seg, dsp_m2m, 36)
        self_dice = dice_OASIS(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation_bspl(fix_seg, dsp_f2f, 36)
        self_dice += dice_OASIS(def_seg.long(), fix_seg.long())

        self_dice = self_dice / 2
    else:
        def_seg = argmax_interpolation_bspl(mov_seg, dsp_m2f, 46)
        # hd95 and log_jet
        hd95 = compute_hd95_oasis(mov_seg.clone().squeeze().cpu().numpy(), def_seg.clone().squeeze().cpu().numpy(),
                                  fix_seg.clone().squeeze().cpu().numpy())
        log_std_jet = compute_std_log_jacobi_det(dsp_m2f.clone().squeeze(0).cpu().numpy())

        mdice = dice_IXI(def_seg.long(), fix_seg.long())

        def_seg = argmax_interpolation_bspl(fix_seg, dsp_f2m, 46)
        fdice = dice_IXI(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation_bspl(mov_seg, dsp_m2m, 46)
        self_dice = dice_IXI(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation_bspl(fix_seg, dsp_f2f, 46)
        self_dice += dice_IXI(def_seg.long(), fix_seg.long())

        self_dice = self_dice / 2

    loss['hd95'] = hd95
    loss['log_std_jet'] = log_std_jet.item()

    loss['m2f_dice'] = mdice.item()
    loss['f2m_dice'] = fdice.item()
    loss['dice'] = (mdice.item() + fdice.item()) / 2
    loss['self_dice'] = self_dice.item()
    return loss


@torch.no_grad()
def sanity_checks_test(register, img_warper, data, loss_bundle, flag):
    mov = data[0]
    fix = data[1]
    mov_seg = data[2]
    fix_seg = data[3]

    dsp_m2f = register(mov, fix)
    dsp_f2m = register(fix, mov)
    dsp_m2m = register(mov, mov)
    dsp_f2f = register(fix, fix)

    loss = {}
    # sim_loss
    loss_sim = loss_bundle.dist_loss(fix, img_warper(mov, dsp_m2f))
    loss_sim += loss_bundle.dist_loss(mov, img_warper(fix, dsp_f2m))
    loss['loss_sim'] = loss_sim.item()

    # grad_loss
    loss_grad = loss_bundle.spatial_grad(dsp_m2f) + loss_bundle.spatial_grad(dsp_f2m)
    loss['loss_grad'] = loss_grad.item()

    # cross_sanity
    loss_cross = loss_bundle.cross_sanity_loss(dsp_m2f, dsp_f2m)
    loss['loss_cross'] = loss_cross.item()

    # self_sanity
    loss_self = torch.mean(dsp_m2m ** 2) + torch.mean(dsp_f2f ** 2)
    loss['loss_self'] = loss_self.item()

    # folds and abs jacobian
    jet, folds = compute_jacobi_and_folds(dsp_m2f + img_warper.grid)  # transformation

    loss['folds'] = folds
    loss['abs_jab'] = jet

    if flag == 'OASIS':
        def_seg = argmax_interpolation(mov_seg, dsp_m2f, img_warper, 36)
        # hd95 and log_jet
        hd95 = compute_hd95_oasis(mov_seg.clone().squeeze().cpu().numpy(), def_seg.clone().squeeze().cpu().numpy(),
                                  fix_seg.clone().squeeze().cpu().numpy())
        log_std_jet = compute_std_log_jacobi_det(dsp_m2f.clone().squeeze(0).cpu().numpy())



        mdice = dice_OASIS(def_seg.long(), fix_seg.long())

        def_seg = argmax_interpolation(fix_seg, dsp_f2m, img_warper, 36)
        fdice = dice_OASIS(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation(mov_seg, dsp_m2m, img_warper, 36)
        self_dice = dice_OASIS(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation(fix_seg, dsp_f2f, img_warper, 36)
        self_dice += dice_OASIS(def_seg.long(), fix_seg.long())

        self_dice = self_dice / 2
    else:
        def_seg = argmax_interpolation(mov_seg, dsp_m2f, img_warper, 46)
        # hd95 and log_jet
        hd95 = compute_hd95_oasis(mov_seg.clone().squeeze().cpu().numpy(), def_seg.clone().squeeze().cpu().numpy(),
                                  fix_seg.clone().squeeze().cpu().numpy())
        log_std_jet = compute_std_log_jacobi_det(dsp_m2f.clone().squeeze(0).cpu().numpy())

        mdice = dice_IXI(def_seg.long(), fix_seg.long())

        def_seg = argmax_interpolation(fix_seg, dsp_f2m, img_warper, 46)
        fdice = dice_IXI(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation(mov_seg, dsp_m2m, img_warper, 46)
        self_dice = dice_IXI(def_seg.long(), mov_seg.long())

        def_seg = argmax_interpolation(fix_seg, dsp_f2f, img_warper, 46)
        self_dice += dice_IXI(def_seg.long(), fix_seg.long())

        self_dice = self_dice / 2

    loss['hd95'] = hd95
    loss['log_std_jet'] = log_std_jet.item()

    loss['m2f_dice'] = mdice.item()
    loss['f2m_dice'] = fdice.item()
    loss['dice'] = (mdice.item() + fdice.item()) / 2
    loss['self_dice'] = self_dice.item()
    return loss


@torch.no_grad()
def sanity_test(epoch, register, img_warper, dataloader, loss_bundle, flag='OASIS', is_bspl=False):
    register.model.eval()
    loss_dict = {}

    for idx, data in enumerate(dataloader):
        data = [t.to('cuda:0', non_blocking=True) for t in data]
        if is_bspl:
            loss = sanity_checks_test_bspl(register, img_warper, data, loss_bundle, flag)
        else:
            loss = sanity_checks_test(register, img_warper, data, loss_bundle, flag)
        for k, v in loss.items():
            if idx == 0:
                loss_dict[k] = []
            loss_dict[k].append(v)

    message = f'{flag}: epoch {epoch}:'

    for k, v in loss_dict.items():
        mean_v = torch.mean(torch.FloatTensor(v))
        std_v = torch.std(torch.FloatTensor(v))

        message += f', {k}: {mean_v.item():.3f}\u00B1{std_v.item():.3f}'

    print(message)


def main():
    # IXI -> OASIS
    for ckpt in IXI_checkpoints:
        # load model
        checkpoint = torch.load(ckpts_dir + ckpt, map_location='cuda:0')
        print(ckpt)
        register = load_model('voxelmorph2', checkpoint, (160, 192, 224))

        # prediction
        sanity_test(checkpoint['epoch'], register, img_warper, OASIS_test_loader, loss_bundle, flag='OASIS')

    for ckpt in bspl_checkpoints:
        # load model
        checkpoint = torch.load(ckpts_dir + ckpt, map_location='cuda:0')
        print(ckpt)
        register = load_model('transmorph-bspl', checkpoint, (160, 192, 224))

        # prediction
        sanity_test(checkpoint['epoch'], register, img_warper, OASIS_test_loader, loss_bundle, flag='OASIS', is_bspl=True)

    # OASIS -> IXI
    for ckpt in OASIS_checkpoints:
        # load model
        checkpoint = torch.load(ckpts_dir + ckpt, map_location='cuda:0')
        print(ckpt)
        register = load_model('transmorph-large', checkpoint, (160, 192, 224))

        # prediction
        sanity_test(checkpoint['epoch'], register, img_warper, IXI_test_loader, loss_bundle, flag='IXI')


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '4'

    ckpts_dir = '/data/duanbin/image_registration/experiments/gen_ckpts/'

    ckpts = os.listdir(ckpts_dir)
    # checkpoints
    IXI_checkpoints = []
    OASIS_checkpoints = []
    bspl_checkpoints = []

    for ckpt in sorted(ckpts):
        if 'TM' in ckpt:
            OASIS_checkpoints.append(ckpt)
        elif 'VM' in ckpt:
            IXI_checkpoints.append(ckpt)
        else:
            bspl_checkpoints.append(ckpt)

    print(IXI_checkpoints)
    print(OASIS_checkpoints)
    print(bspl_checkpoints)

    # dataloaders
    IXI_test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    IXI_test_set = datasets.IXIBrainInferDataset(glob.glob('/data/duanbin/image_registration/dataset/IXI_data/Test/' + '*.pkl'),
                                                 '/data/duanbin/image_registration/dataset/IXI_data/atlas.pkl',
                                                 transforms=IXI_test_composed)
    IXI_test_loader = DataLoader(IXI_test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)


    OASIS_test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),
                                        ])
    OASIS_test_set = datasets.OASISBrainInferDataset(glob.glob('/data/duanbin/image_registration/dataset/OASIS_data/Test/' + '*.pkl'),
                                                     transforms=OASIS_test_composed)
    OASIS_test_loader = DataLoader(OASIS_test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    # warper
    img_warper = SpatialWarper((160, 192, 224))
    img_warper.cuda()

    # loss
    loss_bundle = LossBundle(alpha=0.1, beta=12, img_warper=img_warper, dist_type='NCC', spatial_penalty='l2')
    main()


