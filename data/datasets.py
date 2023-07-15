import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import itertools
import random
import numpy as np


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class OASISBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class LPBABrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

        self.index_pair = list(itertools.permutations(sorted(self.paths), 2))

    def __getitem__(self, index):
        x, x_seg = pkload(self.index_pair[index][0])
        y, y_seg = pkload(self.index_pair[index][1])

        x, y = x[None, ...], y[None, ...]
        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(imgnorm(x)), torch.from_numpy(imgnorm(y))

        return x.float(), y.float()

    def __len__(self):
        return len(self.index_pair)


class OASISBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class LPBABrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

        self.index_pair = list(itertools.permutations(sorted(self.paths), 2))

    def __getitem__(self, index):
        x, x_seg = pkload(self.index_pair[index][0])
        y, y_seg = pkload(self.index_pair[index][1])

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(imgnorm(x)), torch.from_numpy(imgnorm(y)), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x.float(), y.float(), x_seg.float(), y_seg.float()

    def __len__(self):
        return len(self.index_pair)


class LPBABrainTestDataset(Dataset):
    def __init__(self, data_path, alta_path, transforms):
        self.mov_paths = data_path
        self.alta_paths = alta_path
        self.transforms = transforms

        self.index_pair = []
        for i in range(len(self.mov_paths)):
            for j in range(len(self.alta_paths)):
                self.index_pair.append([self.mov_paths[i], self.alta_paths[j]])

        self.index_pair.append([self.alta_paths[0], self.alta_paths[1]])
        self.index_pair.append([self.alta_paths[1], self.alta_paths[0]])

    def __getitem__(self, index):
        x, x_seg = pkload(self.index_pair[index][0])
        y, y_seg = pkload(self.index_pair[index][1])

        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]

        if self.transforms is not None:
            x, x_seg = self.transforms([x, x_seg])
            y, y_seg = self.transforms([y, y_seg])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)

        x, y, x_seg, y_seg = torch.from_numpy(imgnorm(x)), torch.from_numpy(imgnorm(y)), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x.float(), y.float(), x_seg.float(), y_seg.float()

    def __len__(self):
        return len(self.index_pair)