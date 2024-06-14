import matplotlib.pyplot as plt

import open3d as o3d
# This source is based on https://github.com/AbnerHqC/GaitSet/blob/master/pretreatment.py
import argparse
import logging
import multiprocessing as mp
import os
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm
from projection2 import project_one_sample


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
                           v_res,
                           h_res,
                           v_fov,
                           val="depth",
                           cmap="jet",
                           saveto=None,
                           y_fudge=0.0
                           ):
    """ Takes points in 3D space from LIDAR data and projects them to a 2D
        "front view" image, and saves that image.

    Args:
        points: (np array)
            The numpy array containing the lidar points.
            The shape should be Nx4
            - Where N is the number of points, and
            - each point is specified by 4 values (x, y, z, reflectance)
        v_res: (float)
            vertical resolution of the lidar sensor used.
        h_res: (float)
            horizontal resolution of the lidar sensor used.
        v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
        val: (str)
            What value to use to encode the points that get plotted.
            One of {"depth", "height", "reflectance"}
        cmap: (str)
            Color map to use to color code the `val` values.
            NOTE: Must be a value accepted by matplotlib's scatter function
            Examples: "jet", "gray"
        saveto: (str or None)
            If a string is provided, it saves the image as this filename.
            If None, then it just shows the image.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical range do not match the actual data.

            For a Velodyne HDL 64E, set this value to 5.
    """

    # DUMMY PROOFING
    saveto = saveto.replace('.pcd', '.png')
    img = project_one_sample(points)
    aligned_path = saveto.replace('offline','aligned')
    os.makedirs(os.path.dirname(aligned_path), exist_ok=True)
    cv2.imwrite(aligned_path, img)


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
    for img_file in sorted(img_paths):
        pcd_name = img_file.split('/')[-1]
        pcd = o3d.io.read_point_cloud(img_file)
        points = np.asarray(pcd.points)
        HRES = 0.2      # horizontal resolution (assuming 20Hz setting)
        VRES = 0.2   
        VFOV = (-36.0, -11.0) # Field of view (-ve, +ve) along vertical axis
        Y_FUDGE = 0  # y fudge factor for velodyne HDL 64E
        dst_path = os.path.join(output_path, *sinfo)
        os.makedirs(dst_path, exist_ok=True)
        dst_path = os.path.join(dst_path,pcd_name)
        # if points.shape[0] <30:
        #     continue
        try:
            lidar_to_2d_front_view(points, v_res=VRES, h_res=HRES, v_fov=VFOV, val="depth",
                            saveto=dst_path, y_fudge=Y_FUDGE)
        except:
            continue
        # if len(points) == 0:
        #     print(img_file)
    #     to_pickle.append(points)
    # dst_path = os.path.join(output_path, *sinfo)
    # os.makedirs(dst_path, exist_ok=True)
    # pkl_path = os.path.join(dst_path, f'pcd-{sinfo[2]}.pkl')
    # pickle.dump(to_pickle, open(pkl_path, 'wb'))  
    # if len(to_pickle) < 5:
    #     logging.warning(f'{sinfo} has less than 5 valid data.')



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
                for img_path in os.listdir(os.path.join(input_path,sid,seq,view)):
                    if img_path in ['raw','striaght']:
                        continue
                    img_groups[(sid, seq, view,'PCDs_offline_depths')].append(os.path.join(input_path,sid,seq,view,img_path))
                    total_files += 1

    logging.info(f'Total files listed: {total_files}')

    progress = tqdm(total=len(img_groups), desc='Pretreating', unit='folder')

    with mp.Pool(workers) as pool:
        logging.info(f'Start pretreating {input_path}')
        for _ in pool.imap_unordered(partial(pcd2depth, output_path=output_path, img_size=img_size, verbose=verbose, dataset=dataset), img_groups.items()):
            progress.update(1)
    logging.info('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenGait dataset pretreatment module.')
    parser.add_argument('-i', '--input_path', default='', type=str, help='Root path of raw dataset.')
    parser.add_argument('-o', '--output_path', default='', type=str, help='Output path of pickled dataset.')
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
