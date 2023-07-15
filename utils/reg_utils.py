import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SpatialGrad(nn.Module):
    """
    N-D Spatial gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(SpatialGrad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class DisplacementRegularizer(nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv):
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv):
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv):
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:, i, ...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
        return torch.mean(norms) / 3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx ** 2 + dTdyy ** 2 + dTdzz ** 2 + 2 * dTdxy ** 2 + 2 * dTdxz ** 2 + 2 * dTdyz ** 2)

    def forward(self, disp, _):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy


class CrossSanityLoss(nn.Module):
    def __init__(self, alpha, beta, dsp_warper):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.dsp_warper = dsp_warper

    def forward(self, dsp_fw, dsp_bw):
        return cross_sanity_error(dsp_fw, dsp_bw, self.alpha, self.beta, self.dsp_warper)


def _cross_sanity_error(dsp_fw, dsp_bw, alpha, beta, spatial_warper):
    """pixels which fails the cross sanity check"""
    tilde_dsp_fw = spatial_warper(dsp_fw, dsp_bw)
    tilde_dsp_bw = spatial_warper(dsp_bw, dsp_fw)

    # |dsp_fw + tilde_dsp_bw|^2 < alpha * (|dsp_fw|^2 +|tilde_dsp_bw|^2) + beta
    mag_sq_fw = torch.sum(dsp_fw ** 2 + tilde_dsp_bw ** 2, dim=1)
    mag_sq_bw = torch.sum(dsp_bw ** 2 + tilde_dsp_fw ** 2, dim=1)

    flow_sq_diff_fw = torch.sum((dsp_fw + tilde_dsp_bw) ** 2, dim=1)
    flow_sq_diff_bw = torch.sum((dsp_bw + tilde_dsp_fw) ** 2, dim=1)

    # A = flow_sq_diff_fw - (alpha * mag_sq_fw + beta)
    # B = flow_sq_diff_bw - (alpha * mag_sq_bw + beta)
    #
    # A_mean = torch.mean(A[A >= 0])
    # B_mean = torch.mean(B[B >= 0])

    A = flow_sq_diff_fw - (alpha * mag_sq_fw + beta)
    B = flow_sq_diff_bw - (alpha * mag_sq_bw + beta)

    A_mask = (A >= 0).float()
    B_mask = (B >= 0).float()

    A_mean = torch.sum(A * A_mask) / (torch.sum(A_mask) + 1e-5)
    B_mean = torch.sum(B * B_mask) / (torch.sum(B_mask) + 1e-5)

    if torch.isnan(A_mean):
        A_mean = torch.as_tensor(1e-5, device='cuda:0')

    if torch.isnan(B_mean):
        B_mean = torch.as_tensor(1e-5, device='cuda:0')

    return 0.5 * (A_mean + B_mean)


def _cross_sanity_error_plot(dsp_fw, dsp_bw, alpha, beta, spatial_warper):
    """pixels which fails the cross sanity check"""
    tilde_dsp_fw = spatial_warper(dsp_fw, dsp_bw)
    tilde_dsp_bw = spatial_warper(dsp_bw, dsp_fw)

    # |dsp_fw + tilde_dsp_bw|^2 < alpha * (|dsp_fw|^2 +|tilde_dsp_bw|^2) + beta
    mag_sq_fw = torch.sum(dsp_fw ** 2 + tilde_dsp_bw ** 2, dim=1)
    mag_sq_bw = torch.sum(dsp_bw ** 2 + tilde_dsp_fw ** 2, dim=1)

    flow_sq_diff_fw = torch.sum((dsp_fw + tilde_dsp_bw) ** 2, dim=1)
    flow_sq_diff_bw = torch.sum((dsp_bw + tilde_dsp_fw) ** 2, dim=1)

    A = flow_sq_diff_fw - (alpha * mag_sq_fw + beta)
    B = flow_sq_diff_bw - (alpha * mag_sq_bw + beta)

    A_mean = torch.mean(A[A >= 0])
    B_mean = torch.mean(B[B >= 0])

    if torch.isnan(A_mean):
        A_mean = 1e-5

    if torch.isnan(B_mean):
        B_mean = 1e-5


    ax1 = plt.subplot(211)
    ax1.set_title(f'CS. error:{A_mean:.2f}, Num:{torch.count_nonzero(A >= 0)}')
    ax1.hist(torch.flatten(A).cpu().numpy(), color='skyblue')

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.set_title(f'CS. error:{B_mean:.2f}, Num:{torch.count_nonzero(B >= 0)}')
    ax2.hist(torch.flatten(B).cpu().numpy(), color='skyblue')

    plt.tight_layout()
    plt.show()


def cross_sanity_error(dsp_fw, dsp_bw, alpha, beta, spatial_warper):
    # err_A = _cross_sanity_error(dsp_fw, -dsp_fw, alpha, beta, spatial_warper) # interpolation error
    # err_B = _cross_sanity_error(dsp_bw, -dsp_bw, alpha, beta, spatial_warper) # interpolation error

    err_AB = _cross_sanity_error(dsp_fw, dsp_bw, alpha, beta, spatial_warper)

    return err_AB


if __name__ == '__main__':
    from model_utils import SpatialWarper

    dsp_fw = torch.rand(1, 3, 100, 100, 100)
    dsp_bw = torch.rand(1, 3, 100, 100, 100)
    alpha = 0.1
    beta = 5
    spatial_warper = SpatialWarper((100, 100, 100))
    _cross_sanity_error_plot(dsp_fw, dsp_bw, alpha, beta, spatial_warper)