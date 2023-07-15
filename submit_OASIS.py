import glob
import os, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from model_zoo.transmorph import CONFIGS as CONFIGS_TM
import model_zoo.transmorph as TransMorph
from scipy.ndimage.interpolation import map_coordinates, zoom
from data.data_utils import pkload
from utils.sim_utils import NCC


def main():
    loss_sim = NCC()
    test_dir = '/data/duanbin/image_registration/dataset/OASIS_data/Challenge_test_no_gt/'
    save_dir = '/data/duanbin/image_registration/dataset/OASIS_data/Submit/submission/task_03/'
    model_idx = -2  # -1 is latest_checkpoint
    model_dir = '/data/duanbin/image_registration/experiments/TransMorph-Large-ESC/'
    config = CONFIGS_TM['TransMorph-Large']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    file_names = glob.glob(test_dir + '*.pkl')
    with torch.no_grad():
        stdy_idx = 0
        for data in file_names:
            x, y = pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            file_name = file_names[stdy_idx].split('/')[-1].split('.')[0][2:]
            print(file_name)
            model.eval()
            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)
            print(f'similarity: {loss_sim(x_def, y)}')
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(np.float16)
            print(flow.shape)
            np.savez(save_dir + 'disp_{}.npz'.format(file_name), flow)
            stdy_idx += 1


if __name__ == '__main__':
    '''
    GPU configuration
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'

    main()
