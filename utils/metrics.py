"""
Author: Duan, Bin
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from surface_distance.metrics import compute_surface_distances, compute_robust_hausdorff
import utils.finite_differences as FD
import scipy.ndimage


def dice_IXI(y_pred, y_true):
    # use same number of classes used in voxel_morph paper
    # ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebral-White-Matter', 'Cerebellum-White-Matter', 'Putamen',
    #  'VentralDC', 'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
    #  '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF', 'choroid-plexus']

    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32,
                34, 36]

    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2. * intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        idx += 1
    return np.mean(DSCs)


def dice_OASIS(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                29, 30, 31, 32, 33, 34, 35]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)


def dice_LPBA(y_pred, y_true):
    # use same number of classes used in lapIRN paper
    # ['Brain-Stem', 'Thalamus', 'Cerebellum-Cortex', 'Cerebellum-White-Matter', 'Putamen',
    #  'Pallidum', 'Caudate', 'Lateral-Ventricle', 'Hippocampus',
    #  '3rd-Ventricle', '4th-Ventricle', 'Amygdala', 'Cerebral-Cortex', 'CSF']

    im1 = y_pred.detach().cpu().numpy()[0, 0, ...]
    atlas = y_true.detach().cpu().numpy()[0, 0, ...]

    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / num_count


def dice_fn(y_pred, y_true, num_clus=46):
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2. * intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def compute_determinant_of_jacobian(phi, spacing):
    fdt = FD.FD_torch(spacing)
    dim = len(spacing)

    if dim == 1:
        p0x = fdt.dXc(phi[:, 0, ...])
        det = p0x
    elif dim == 2:
        p0x = fdt.dXc(phi[:, 0, ...])
        p0y = fdt.dYc(phi[:, 0, ...])
        p1x = fdt.dXc(phi[:, 1, ...])
        p1y = fdt.dYc(phi[:, 1, ...])

        det = p0x * p1y - p0y * p1x
    elif dim == 3:
        p0x = fdt.dXc(phi[:, 0, ...])
        p0y = fdt.dYc(phi[:, 0, ...])
        p0z = fdt.dZc(phi[:, 0, ...])
        p1x = fdt.dXc(phi[:, 1, ...])
        p1y = fdt.dYc(phi[:, 1, ...])
        p1z = fdt.dZc(phi[:, 1, ...])
        p2x = fdt.dXc(phi[:, 2, ...])
        p2y = fdt.dYc(phi[:, 2, ...])
        p2z = fdt.dZc(phi[:, 2, ...])

        det = p0x * p1y * p2z + p0y * p1z * p2x + p0z * p1x * p2y - p0z * p1y * p2x - p0y * p1x * p2z - p0x * p1z * p2y
    else:
        raise ValueError('Can only compute the determinant of Jacobian for dimensions 1, 2 and 3')

    return det


def compute_jacobi_and_folds(map):
    """
    compute determinant jacobian and folds on transformatiom map,  the coordinate should be canonical.
    """
    assert len(map.size()) == 5

    jacobi_det = compute_determinant_of_jacobian(map, spacing=np.ones(3))

    jacobi_abs = - torch.sum(jacobi_det[jacobi_det < 0.])
    jacobi_num = torch.sum(jacobi_det < 0.)
    jacobi_abs_mean = jacobi_abs / map.shape[0]
    folds = jacobi_num / map.shape[0]

    return jacobi_abs_mean, 100 * folds / np.prod(map.shape[2:])


def compute_std_log_jacobi_det(disp_field):
    def jacobian_determinant(disp):
        _, _, H, W, D = disp.shape

        gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
        grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
        gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

        gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                               scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                               scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)

        grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                               scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                               scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)

        gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                               scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                               scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

        grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

        jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
        jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
        jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
                 jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) + \
                 jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])

        return jacdet

    jac_det = (jacobian_determinant(disp_field[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)

    return log_jac_det.std()


def compute_hd95_oasis(moving, moving_warped, fixed):

    hd95 = 0
    count = 0
    for i in range(1, 36):
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            continue
        hd95 += compute_robust_hausdorff(compute_surface_distances((fixed == i), (moving_warped == i), np.ones(3)), 95.)
        count += 1
    hd95 /= count

    return hd95