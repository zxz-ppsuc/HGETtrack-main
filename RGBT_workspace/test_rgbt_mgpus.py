import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)

from lib.test.tracker.ostrack import OSTrack
#from lib.test.tracker.ostrack import ViPTTrack
import lib.test.parameter.ostrack as rgbt_prompt_params
import multiprocessing
import torch
#from lib.train.dataset.depth_utils import get_x_frame
import time
from glob import glob

def X2Cube(img, B=[4, 4], skip = [4, 4], bandNumber=16):
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//B[0], N//B[1], bandNumber)

    hsi_min = np.min(DataCube, axis=(0,1), keepdims=True)
    hsi_max = np.max(DataCube, axis=(0,1), keepdims=True)
    hsi_normed = (DataCube - hsi_min) / ((hsi_max - hsi_min) + 1e-6) * 255
    hsi_normed = hsi_normed.astype(np.uint8)
    return hsi_normed

def genConfig(seq_path, set_type):
    if set_type == 'RGBT234':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if os.path.splitext(p)[1] == '.jpg'])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'GTOT':
        ############################################  have to refine #############################################
        RGB_img_list = sorted([seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
        T_img_list = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])

        RGB_gt = np.loadtxt(seq_path + '/groundTruth_v.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')

        x_min = np.min(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(RGB_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(RGB_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(RGB_gt[:,[1,3]],axis=1)[:,None]
        RGB_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

        x_min = np.min(T_gt[:,[0,2]],axis=1)[:,None]
        y_min = np.min(T_gt[:,[1,3]],axis=1)[:,None]
        x_max = np.max(T_gt[:,[0,2]],axis=1)[:,None]
        y_max = np.max(T_gt[:,[1,3]],axis=1)[:,None]
        T_gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)
    
    elif set_type == 'LasHeR':
        RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')

    elif set_type == 'challenge2023':
        RGB_img_list = sorted([seq_path + '/HSI-FalseColor/' + p for p in os.listdir(seq_path + '/HSI-FalseColor') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/HSI/' + p for p in os.listdir(seq_path + '/HSI') if p.endswith(".png")])

        RGB_gt = np.loadtxt(seq_path + '/HSI-FalseColor' + '/groundtruth_rect.txt', delimiter=',')
        T_gt = np.loadtxt(seq_path + '/HSI' + '/groundtruth_rect.txt', delimiter=',')


    elif 'VTUAV' in set_type:
        RGB_img_list = sorted([seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if p.endswith(".jpg")])
        T_img_list = sorted([seq_path + '/ir/' + p for p in os.listdir(seq_path + '/ir') if p.endswith(".jpg")])

        RGB_gt = np.loadtxt(seq_path + '/rgb.txt', delimiter=' ')
        T_gt = np.loadtxt(seq_path + '/ir.txt', delimiter=' ')

    return RGB_img_list, T_img_list, RGB_gt, T_gt


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='ostrack'):
    if 'VTUAV' in dataset_name:
        seq_txt = seq_name.split('/')[1]
    else:
        seq_txt = seq_name
    # save_name = '{}_ep{}'.format(yaml_name, epoch)
    save_name = '{}'.format(yaml_name)
    save_path = f'./RGBT_workspace/results/{dataset_name}/' + save_name +  '/' + seq_txt + '.txt'
    save_folder = f'./RGBT_workspace/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    if script_name == 'ostrack':
        params = rgbt_prompt_params.parameters(yaml_name, epoch)
        mmtrack = OSTrack(params)  # "GTOT" # dataset_name
        tracker = OSTrack_RGBT(tracker=mmtrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    RGB_img_list, T_img_list, RGB_gt, T_gt = genConfig(seq_path, dataset_name)
    if len(RGB_img_list) == len(RGB_gt):
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(RGB_img_list), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    toc = 0
    for frame_idx, rgb_img in enumerate(RGB_img_list):
        tic = cv2.getTickCount()
        image = cv2.cvtColor(cv2.imread(rgb_img), cv2.COLOR_BGR2RGB)
        hsi_img = cv2.imread(T_img_list[frame_idx], -1)
        hsi_img_norm = X2Cube(hsi_img, B=[4, 4], skip = [4, 4], bandNumber=16)
        #if seq_name.startswith("HSI-RedNIR"):
        #    hsi_img_norm = hsi_img_norm[:,:,:-1]
        frame = np.concatenate((image, hsi_img_norm), axis=2)
        if frame_idx == 0:
            # initialization
            tracker.initialize(frame, RGB_gt[0].tolist())  # xywh
        elif frame_idx > 0:
            # track
            region = tracker.track(frame)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class OSTrack_RGBT(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        
        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        #pred_score = outputs['best_score']
        return pred_bbox#, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBT dataset.')
    parser.add_argument('--script_name', type=str, default='ostrack', help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='vitb_384_mae_ce_32x4_got10k_ep100', help='Name of tracking method.')  # vitb_256_mae_ce_32x4_ep300 vitb_256_mae_ce_32x4_ep60_prompt_i32v21_onlylasher_rgbt
    parser.add_argument('--dataset_name', type=str, default='challenge2023', help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT,challenge2023).')
    parser.add_argument('--threads', default=8, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=20, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='parallel', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    # path initialization
    seq_list = None
    if dataset_name == 'GTOT':
        seq_home = '/home/lz/Videos/GTOT'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'RGBT234':
        seq_home = '/media/jiawen/Datasets/Tracking/DATASET_TEST/RGBT234'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'LasHeR':
        seq_home = '/home/shiyanshi408/ZXZ/dataset/lasher/testingset'
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'challenge2023':
        seq_home = '/root/autodl-tmp/challenge2023/test'           #这里修改成自己的数据集地址
        seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    elif dataset_name == 'VTUAVST':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/short-term'
        with open(join(join(seq_home, 'VTUAV-ST.txt')), 'r') as f:
            seq_list = f.read().splitlines()
    elif dataset_name == 'VTUAVLT':
        seq_home = '/mnt/6196b16a-836e-45a4-b6f2-641dca0991d0/VTUAV/test/long-term'
        with open(join(seq_home, 'VTUAV-LT.txt'), 'r') as f:
            seq_list = f.read().splitlines()
    else:
        raise ValueError("Error dataset!")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
