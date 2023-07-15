import glob
from natsort import natsorted
import random
import shutil
import os
from batchgenerators.utilities.file_and_folder_operations import *


def random_IXI():
    train_dir = '/data/duanbin/image_registration/dataset/IXI_data/Train/'

    sample_list = natsorted(glob.glob(train_dir + '*.pkl'))

    sub_sample_list = random.sample(sample_list, 50)

    for s in sub_sample_list:
        new_place = s.replace('/Train/', '/Train_Sub/')
        shutil.copy(s, new_place)

    print('IXI Dataset Done!')


def random_LPBA():
    dir = '/data/duanbin/image_registration/dataset/LPBA_data/'

    train_dir = join(dir, 'Train')
    val_dir = join(dir, 'Val')
    test_dir = join(dir, 'Test')

    maybe_mkdir_p(train_dir)
    maybe_mkdir_p(val_dir)
    maybe_mkdir_p(test_dir)

    sample_list = natsorted(glob.glob(dir + 'all_data/' + '*.pkl'))

    sub_sample_list = random.sample(sample_list, 28)

    for s in sub_sample_list:
        new_place = s.replace('/all_data/', '/Train/')
        shutil.copy(s, new_place)

    left_sample_list = list(set(sample_list) - set(sub_sample_list))

    sub_sample_list = random.sample(left_sample_list, 2)

    for s in sub_sample_list:
        new_place = s.replace('/all_data/', '/Val/')
        shutil.copy(s, new_place)

    left_sample_list = list(set(left_sample_list) - set(sub_sample_list))

    for s in left_sample_list:
        new_place = s.replace('/all_data/', '/Test/')
        shutil.copy(s, new_place)

    print('LPBA Dataset Done!')


if __name__ == '__main__':
    random.seed(521)
    random_LPBA()
