import os
import os.path
import numpy as np
import torch
import csv
import pandas
from lib.train.data import jpeg4py_loader
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings

from .HyperTools import X2Cube, X2Cube2
from PIL import Image
import cv2
class HOT(BaseVideoDataset):
    """ LasHeR dataset(aligned version).

    Publication:
        A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/pdf/2104.13202.pdf

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None,image_loader=jpeg4py_loader):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().hot_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in HOT, got {}'.format(split)
        super().__init__('HOT', root, image_loader)
        self.dtype = dtype

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'hot'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _get_sequence_list(self, split): #这里感觉要修改
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'hot_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list

    def _read_bb_anno(self, seq_path):
        # in hot dataset, visible.txt is same as infrared.txt
        rgb_bb_anno_file = os.path.join(seq_path, 'HSI-FalseColor', "groundtruth_rect.txt") # 注意路径有些出入，HOT数据集中标签文件在下一级目录中，且rgb和红外数据集的标注不同
       # ir_bb_anno_file = os.path.join(seq_path, 'HSI-FalseColor', "groundtruth_rect.txt")
        # rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values 如果报错，启用这句
       # ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        try: #如果报错，注释这四句
            gt = np.loadtxt(rgb_bb_anno_file, delimiter=',', dtype=np.float32)
        except:
            gt = np.loadtxt(rgb_bb_anno_file, dtype=np.float32)
        return torch.tensor(gt)#, torch.tensor(ir_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])


    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}
    def _get_frame_path(self, seq_path, frame_id):
        # Note original filename is chaotic, we rename them
        rgb_frame_path = os.path.join(seq_path, 'HSI-FalseColor', '{:04d}.jpg'.format(frame_id))  # frames start from 0
        hsi_frame_path = os.path.join(seq_path, 'HSI_NPY', '{:04d}.npy'.format(frame_id))

        return (rgb_frame_path, hsi_frame_path)  # jpg png

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id) #jpg npy
        rgb = cv2.imread(rgb_frame_path)
        hsi = np.load(ir_frame_path)
        img = np.concatenate((rgb, hsi), axis=2)
        #img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img  # (h,w,6) =>(h,w,19)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key == 'seq_belong_mask':
                continue
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
