import numpy as np
import torch

from utils.reg_utils import *
from utils.sim_utils import *


class SanityProximal():
    def __init__(self, moving, fixed, register, img_warper, alpha, beta, dist_type='NCC', spatial_penalty='l2',
                 flag_sim=True, flag_grad=True, flag_self=True, flag_cross=True):
        super().__init__()
        self.moving = moving
        self.fixed = fixed

        self.register = register
        self.img_warper = img_warper

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
        self.flag_sim = flag_sim
        self.flag_self = flag_self
        self.flag_cross = flag_cross
        self.flag_grad = flag_grad
        self.cross_sanity_loss = CrossSanityLoss(alpha, beta, img_warper)

    def __call__(self, x, flag, optimizer):
        """to save memory we do multiple backwards and accumulate gradients"""
        total_loss = 0.
        grad = torch.zeros_like(x)

        if flag == 'stage_1':
            if self.flag_sim:
                total_loss, grad = self.image_similarity(x.clone(), optimizer)

            if self.flag_grad:
                loss, tmp_grad = self.dsp_grad(x.clone(), optimizer)
                total_loss += loss
                grad += tmp_grad

            if self.flag_self:
                loss, tmp_grad = self.self_sanity(x.clone(), optimizer)
                total_loss += loss
                grad += tmp_grad

            if self.flag_cross:
                loss, tmp_grad = self.cross_sanity(x.clone(), optimizer)
                total_loss += loss
                grad += tmp_grad

        return total_loss, grad

    def image_similarity(self, x, optimizer):
        if x.grad is not None:
            x.grad.data.zero_()

        x.requires_grad_()
        # clear all gradients in all models
        optimizer.zero_grad()

        movtmp, fixtmp = torch.split(x, [1, 1], dim=1)

        # similarity between orginal images and modified images
        # loss = self.dist_loss(self.fixed, fixtmp) + self.dist_loss(self.moving, movtmp)
        loss = 0.

        dsp = self.register(movtmp, fixtmp)
        loss += self.dist_loss(self.fixed, self.img_warper(self.moving, dsp))

        dsp = self.register(fixtmp, movtmp)
        loss += self.dist_loss(self.moving, self.img_warper(self.fixed, dsp))
        loss.backward()

        return loss.detach().cpu().numpy(), x.grad.detach()

    def dsp_grad(self, x, optimizer):
        if x.grad is not None:
            x.grad.data.zero_()

        x.requires_grad_()
        # clear all gradients in all models
        optimizer.zero_grad()

        movtmp, fixtmp = torch.split(x, [1, 1], dim=1)
        loss = self.spatial_grad(self.register(movtmp, fixtmp)) + self.spatial_grad(self.register(fixtmp, movtmp))
        loss.backward()

        return loss.detach().cpu().numpy(), x.grad.detach()

    def self_sanity(self, x, optimizer):
        if x.grad is not None:
            x.grad.data.zero_()

        x.requires_grad_()
        # clear all gradients in all models
        optimizer.zero_grad()

        movtmp, fixtmp = torch.split(x, [1, 1], dim=1)
        loss = 0.01 * torch.mean(self.register(movtmp, movtmp) ** 2) + torch.mean(self.register(fixtmp, fixtmp) ** 2)
        loss.backward()

        return loss.detach().cpu().numpy(), x.grad.detach()

    def cross_sanity(self, x, optimizer):
        if x.grad is not None:
            x.grad.data.zero_()

        x.requires_grad_()
        # clear all gradients in all models
        optimizer.zero_grad()

        movtmp, fixtmp = torch.split(x, [1, 1], dim=1)
        loss = 0.01 * self.cross_sanity_loss(self.register(movtmp, fixtmp), self.register(fixtmp, movtmp))
        loss.backward()

        return loss.detach().cpu().numpy(), x.grad.detach()
