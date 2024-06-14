import matplotlib.pyplot as plt

import open3d as o3d
# This source is based on https://github.com/AbnerHqC/GaitSet/blob/master/pretreatment.py
import argparse
import logging
import multiprocessing as mp
import os
import pickle as pkl
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
from deform2 import project_one_sample
# from natsort import natsorted


def align_img(img: np.ndarray, img_size: int = 64) -> np.ndarray:
    """Aligns the image to the center.
    Args:
        img (np.ndarray): Image to align.
        img_size (int, optional): Image resizing size. Defaults to 64.
    Returns:
        np.ndarray: Aligned image.
    """    
    if img.sum() <= 10000:
        y_top = 0
        y_btm = img.shape[0]
    else:
        # Get the upper and lower points
        # img.sum
        y_sum = img.sum(axis=2).sum(axis=1)
        y_top = (y_sum != 0).argmax(axis=0)
        y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)

    img = img[y_top: y_btm, :,:]

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = img.shape[1] / img.shape[0]
    img = cv2.resize(img, (int(img_size * ratio), img_size), interpolation=cv2.INTER_CUBIC)
    
    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = img.sum(axis=2).sum(axis=0).cumsum()
    x_center = img.shape[1] // 2
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break

    # if not x_center:
    #     logging.warning(f'{img_file} has no center.')
    #     continue

    # Get the left and right points
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        # _ = np.zeros((img.shape[0], half_width,3))
        # img = np.concatenate([_, img, _], axis=1)
    
    img = img[:, left: right,:].astype('uint8')
    return img


def lidar_to_2d_front_view(points,
                           saveto=None,
                           ):
    
    saveto = saveto.replace('.txt', '.png')
    img = project_one_sample(points)
    aligned_path = saveto.replace('offline','aligned')
    os.makedirs(os.path.dirname(aligned_path), exist_ok=True)
    cv2.imwrite(aligned_path, img)
    
    
def warp_cal(points_source, points_target, joints_source, joints_target):
    source_ori = np.mean(points_source, axis=0, keepdims=True)
    target_ori = np.mean(points_target, axis=0, keepdims=True)

    joints_source = joints_source - joints_source[0] + source_ori
    joints_target = joints_target - joints_target[0] + target_ori

    connects = [[0, 1], [0, 2], [0, 3],
                [1, 4],
                [2, 5],
                [3, 6],
                [4, 7],
                [5, 8],
                [6, 9],
                [7, 10],
                [8, 11],
                [9, 13], [9, 12], [9, 14],
                [12, 15],
                [13, 16],
                [14, 17],
                [16, 18],
                [17, 19],
                [18, 20],
                [19, 21],
                [20, 22],
                [21, 23]]

    # 求解骨骼的方向向量
    source_function = {}
    target_function = {}
    for indx in range(len(connects)):
        connection = connects[indx]
        dirrection_source = joints_source[connects[indx][1]] - joints_source[
            connects[indx][0]]
        dirrection_target = joints_target[connects[indx][1]] - joints_target[
            connects[indx][0]]
        source_function.update({f'd_{indx}': dirrection_source,
                                f'P_{indx}': joints_source[connects[indx][0]]})
        target_function.update({f'd_{indx}': dirrection_target,
                                f'P_{indx}': joints_target[connects[indx][0]]})

    # 求解点云到每一个骨骼的投影Q
    deforms = []
    # labels = []
    for point in points_source:
        closest_lenth = float('inf')
        closest_proj = None

        for i in range(23):
            v = source_function[f'd_{i}']  # (3,)
            w = point - source_function[f'P_{i}']  # (3)
            proj = np.dot(w, v) / np.dot(v, v)  # ()
            proj = np.clip(proj, 0, 1)
            closest_point = source_function[f'P_{i}'] + proj * v
            lenth = np.linalg.norm(point - closest_point)
            if lenth < closest_lenth:
                closest_lenth = lenth
                closest_proj = proj
                label = i
                deform = target_function[f'P_{i}'] + proj * target_function[f'd_{i}'] - \
                         source_function[f'P_{i}'] - proj * source_function[f'd_{i}']
        deforms.append(deform)
        # labels.append(label)

    deform_label = np.hstack((points_source,np.array(deforms)))
    return deform_label


def pcd2depth(img_groups: Tuple, output_path: Path, img_size: int = 64, verbose: bool = False, dataset='CASIAB') -> None:
    """Reads a group of images and saves the data in pickle format.
    Args:
        img_groups (Tuple): Tuple of (sid, seq, view) and list of image paths.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        verbose (bool, optional): Display debug info. Defaults to False.
    """    
    sinfo = img_groups[0]
    img_paths = img_groups[1]
    content = img_paths[0]
    joint_locations = img_paths[1]
    
    for indx in range(len(joint_locations)-1):
        point_source = content[indx]
        point_target = content[indx + 1]

        joints_source = joint_locations[indx]
        joints_target = joint_locations[indx + 1]
        warp = warp_cal(point_source, point_target, joints_source, joints_target)

        dst_path = os.path.join(output_path, *sinfo)
        os.makedirs(dst_path, exist_ok=True)
        dst_path = os.path.join(dst_path,f'{indx}.txt')

        try:
            lidar_to_2d_front_view(warp, saveto=dst_path)
        except:
            print(sinfo)


def pretreat(input_path: Path, output_path: Path, img_size: int = 64, workers: int = 4, verbose: bool = False, dataset: str = 'CASIAB') -> None:
    """Reads a dataset and saves the data in pickle format.
    Args:
        input_path (Path): Dataset root path.
        output_path (Path): Output path.
        img_size (int, optional): Image resizing size. Defaults to 64.
        workers (int, optional): Number of thread workers. Defaults to 4.
        verbose (bool, optional): Display debug info. Defaults to False.
    """
    img_groups = defaultdict(list)
    logging.info(f'Listing {input_path}')
    total_files = 0
    for sid in tqdm(sorted(os.listdir(input_path))):
        for seq in os.listdir(os.path.join(input_path,sid)):
            for view in os.listdir(os.path.join(input_path,sid,seq)):
                joint_pkl_path = os.path.join(input_path, sid, seq, view,f'05-{view}-Joints-Location.pkl')
                point_pkl_path = os.path.join(input_path, sid, seq, view,f'00-{view}-LiDAR-PCDs.pkl')
                with open(joint_pkl_path,'rb') as file:
                    joint_locations = pkl.load(file)
                file.close()
                with open(point_pkl_path, 'rb') as file1:
                    point_locations = pkl.load(file1)
                file1.close()
                # for img_path in os.listdir(os.path.join(input_path,sid,seq,view)):
                #     if img_path in ['raw','striaght']:
                #         continue
                #     img_groups[(sid, seq, view,'PCDs_offline_depths')].append(os.path.join(input_path,sid,seq,view,img_path))
                #     total_files += 1
                img_groups[(sid, seq, view, 'PCDs_offline_depths')].append(point_locations)
                img_groups[(sid, seq, view, 'PCDs_offline_depths')].append(joint_locations)

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(partial(pcd2depth, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset), img_groups.items()):
            progress.update(1)
    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', type=str, help='Root path of dataset.')
    parser.add_argument('-o', '--output_path', type=str, help='Output path of deformable dataset.')
    parser.add_argument('-l', '--log_file', default='./pretreatment.log', type=str, help='Log file path. Default: ./pretreatment.log')
    parser.add_argument('-n', '--n_workers', default=4, type=int, help='Number of thread workers. Default: 4')
    parser.add_argument('-r', '--img_size', default=64, type=int, help='Image resizing size. Default 64')
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Display debug info.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format='[%(asctime)s - %(levelname)s]: %(message)s')

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info('Verbose mode is on.')
        for k, v in args.__dict__.items():
            logging.debug(f'{k}: {v}')

    pretreat(input_path=Path(args.input_path), output_path=Path(args.output_path), img_size=args.img_size, workers=args.n_workers, verbose=args.verbose, dataset=args.dataset)
