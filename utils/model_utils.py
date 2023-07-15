import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_zoo import voxelmorph, transmorph, lapIRN, transmorph_bspl


def load_model(model_type, checkpoint, img_size):
    wrapper = find_model_type(model_type, img_size)

    # load the actual model
    try:
        if model_type.lower() == 'lapIRN'.lower():
            if checkpoint is not None:
                wrapper.model.load_state_dict(checkpoint)

            wrapper.model.model_lvl2.model_lvl1.cuda()
            wrapper.model.model_lvl2.cuda()
            wrapper.model.cuda()
        else:
            if checkpoint is not None:
                best_model = checkpoint['state_dict']
                wrapper.model.load_state_dict(best_model)

            wrapper.model.cuda()

    except Exception as e:
        print(e)
        exit()

    return wrapper


def find_model_type(model_type, img_size=(160, 192, 224)):
    if model_type == 'voxelmorph1':
        wrapper = VoxelMorph1(img_size)

    if model_type == 'voxelmorph2':
        wrapper = VoxelMorph2(img_size)

    if model_type == 'transmorph':
        config = transmorph.CONFIGS['TransMorph']
        wrapper = TransMorph(config)

    if model_type == 'transmorph-large':
        config = transmorph.CONFIGS['TransMorph-Large']
        wrapper = TransMorph(config)

    if model_type == 'lapIRN':
        wrapper = LapIRN(img_size)

    if model_type == 'transmorph-bspl':
        config = transmorph_bspl.CONFIGS['TransMorphBSpline']
        wrapper = TransMorphBSpline(config)

    return wrapper


class SpatialWarper(nn.Module):
    def __init__(self, img_size):
        super(SpatialWarper, self).__init__()
        sz = np.array(img_size)

        # identity map
        grid = [torch.arange(0, s) for s in sz]
        grid = torch.meshgrid(grid)
        grid = torch.stack(grid)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid)

    def forward(self, I0, dsp, mode='bilinear'):
        """Warps image.
        :param I0: image to warp, image size BxCxXxYxZ
        :param dsp: displacement map for the warping, size BxdimxXxYxZ
        :return: returns the warped image of size BxCxXxYxZ
        """
        # compose with identity map coordinates
        shape = dsp.shape[2:]
        new_locs = dsp + self.grid

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(I0, new_locs, align_corners=True, mode=mode)


class ModelWrapper(nn.Module):
    def __init__(self):
        super(ModelWrapper, self).__init__()
        self.model = None

    def forward(self, moving_image, fixed_image):
        """
        Args:
            moving_image:
            fixed_image:

        Returns:
            dsp: displacement
        """

        raise NotImplementedError("Each subclass has to implement its own forward_model function")


class VoxelMorph1(ModelWrapper):
    def __init__(self, img_size):
        super(VoxelMorph1, self).__init__()

        self.model = voxelmorph.VxmDense_1(img_size)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat((x, y), dim=1)

        x = self.model(x)

        return x


class VoxelMorph2(ModelWrapper):
    def __init__(self, img_size):
        super(VoxelMorph2, self).__init__()

        self.model = voxelmorph.VxmDense_2(img_size)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat((x, y), dim=1)

        x = self.model(x)

        return x


class TransMorph(ModelWrapper):
    def __init__(self, config):
        super(TransMorph, self).__init__()

        self.model = transmorph.TransMorph(config=config)

    def forward(self, x, y=None):
        if y is not None:
            x = torch.cat((x, y), dim=1)

        x = self.model(x)

        return x


class TransMorphBSpline(ModelWrapper):
    def __init__(self, config):
        super(TransMorphBSpline, self).__init__()

        self.model = transmorph_bspl.TranMorphBSplineNet(config=config)

    def forward(self, x, y=None, tuple=True):
        if y is not None:
            x = (x, y)

        if tuple:
            _, x, y = self.model(x)
            x = (x, y)
        else:
            _, _, x = self.model(x)

        return x


class LapIRN(ModelWrapper):
    def __init__(self, img_size):
        super(LapIRN, self).__init__()

        range_flow = 0.4
        start_channel = 7
        img_size_4 = totuple(np.array(img_size) / 4)
        img_size_2 = totuple(np.array(img_size) / 2)

        model_lvl1 = lapIRN.LDR_laplacian_unit_disp_add_lvl1(2, 3, start_channel, is_train=True, imgshape=img_size_4,
                                                                 range_flow=range_flow)
        model_lvl2 = lapIRN.LDR_laplacian_unit_disp_add_lvl2(2, 3, start_channel, is_train=True, imgshape=img_size_2,
                                                                 range_flow=range_flow, model_lvl1=model_lvl1)

        self.model = lapIRN.LDR_laplacian_unit_disp_add_lvl3(2, 3, start_channel, is_train=False, imgshape=img_size,
                                                            range_flow=range_flow, model_lvl2=model_lvl2)

    def forward(self, x, y=None):

        x = self.model(x, y)

        return x


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a