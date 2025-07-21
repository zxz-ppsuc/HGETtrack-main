import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
class HOTDataset(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.hot_path
        self.sequence_list = self._get_sequence_list()


    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path_rgb = '{}/{}/HSI-FalseColor/groundtruth_rect.txt'.format(self.base_path, sequence_name)
        anno_path_x = '{}/{}/HSI/groundtruth_rect.txt'.format(self.base_path, sequence_name)

        ground_truth_rect_rgb = load_text(str(anno_path_rgb), delimiter=',', dtype=np.float64)
        ground_truth_rect_x = load_text(str(anno_path_x), delimiter=',', dtype=np.float64)
        ground_truth_rect = np.concatenate([ground_truth_rect_rgb, ground_truth_rect_x], axis=1)
        # /mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term/test_ST_001/animal_001/rgb/000001.jpg
        rgb_frames_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'HSI-FalseColor')
        rgb_frame_list = sorted([frame for frame in os.listdir(rgb_frames_path) if frame.endswith(".jpg")])
        # rgb_frame_list.sort(key=lambda f: int(f.split('.')[0]))
        rgb_frames_list = [os.path.join(rgb_frames_path, frame) for frame in rgb_frame_list]

        x_frames_path = '{}/{}/{}'.format(self.base_path, sequence_name, 'HSI')
        x_frame_list = sorted([frame for frame in os.listdir(x_frames_path) if frame.endswith(".png")])
        # x_frame_list.sort(key=lambda f: int(f.split('.')[0]))
        x_frames_list = [os.path.join(x_frames_path, frame) for frame in x_frame_list]
        frames_list = list(zip(rgb_frames_list, x_frames_list))

        return Sequence(sequence_name, frames_list, 'challenge2023', ground_truth_rect.reshape(-1, 8))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [
            'fruit',
            'board',
            'car3',
            'card',
            'ball',
            'player',
            'toy1',
            'bus2',
            'face2',
            'forest2',
            'rider2',
            'excavator',
            'forest',
            'coke',
            'car2',
            'campus',
            'basketball',
            'pedestrain',
            'coin',
            'face',
            'student',
            'worker',
            'rubik',
            'hand',
            'toy2',
            'car',
            'bus',
            'rider1',
            'paper',
            'book',
            'playground',
            #'truck',
            'kangaroo',
            'pedestrian2',
            'drive',
        ]

        return sequence_list
