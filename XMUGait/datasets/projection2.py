import numpy as np
import pickle
import math, cv2, os, h5py

# import configargparse
from tqdm import tqdm


def get_size(point,spacing):
    po1_x_min, po1_x_max = point[:, 0].min(), point[:, 0].max()
    po1_y_min, po1_y_max = point[:, 1].min(), point[:, 1].max()
    po1_z_min, po1_z_max = point[:, 2].min(), point[:, 2].max()

    # print (po1_x_min, po1_x_max)
    # print(po1_y_min, po1_y_max)
    # print(po1_z_min, po1_z_max)

    len_x, len_y, len_z = po1_x_max - po1_x_min, po1_y_max - po1_y_min, po1_z_max - po1_z_min
    size = np.array([len_x, len_y, len_z])
    #print (size)
    size = (size / spacing).astype(int)
    # size = np.array([math.ceil(len_x),math.ceil(len_y),math.ceil(len_z)])
    origin = np.array([po1_x_min, po1_y_min, po1_z_min])
    # mass or center?
    center = np.mean(point, axis=0)
    return size,origin,center


def _Crop(img):

    W, H = img.shape[0], img.shape[1]

    if H < W:
        dist_H = W - H
        if dist_H % 2 == 0:
            img = np.pad(img, ((0,0),(int((dist_H) / 2),int((dist_H) / 2))), 'constant', constant_values=0)
        else:
            img = np.pad(img, ((0, 0), (int((dist_H - 1) / 2) + 1, int((dist_H - 1) / 2))), 'constant',
                         constant_values=0)
            # img = np.pad(img, (int((dist_H - 1) / 2) + 1, int((dist_H - 1) / 2), 0, 0, 0, 0), 'constant',
            #             value=0)

    return img


def point2voxel(point,grid_size,origin,voxel_spacing,voxel_center):
    #grid_size = (size / voxel_spacing).astype(int)
    Mx,My,Mz = grid_size[0],grid_size[1],grid_size[2]
    voxel_3D = np.zeros(grid_size)
    norm_point = (point-origin)/voxel_spacing
    for i in range(norm_point.shape[0]):
        corrd = norm_point[i]
        coordx,coordy,coordz = math.floor(corrd[0]),math.floor(corrd[1]),math.floor(corrd[2])
        coordx = min(coordx, Mx-1)
        coordy = min(coordy, My-1)
        coordz = min(coordz, Mz-1)
        voxel_3D[coordx,coordy,coordz] += 1
    return voxel_3D


spacing = [0.02, 0.02, 0.02]


def project_one_sample(input_array):
    size, origin, center = get_size(input_array,spacing)
    voxel_3D = point2voxel(input_array, size, origin, spacing, center)

    x,y,z = np.where(voxel_3D != 0)
    project_size = voxel_3D.shape[1:]
    priect_image= np.zeros((project_size[0],project_size[1]))

    value = np.sqrt((x * spacing[0] + origin[0]) ** 2 + (y * spacing[1] + origin[1]) ** 2 + (z * spacing[2] + origin[2]) ** 2 )
    value = ((value-value.min())/(value.max()-value.min()))*255
    priect_image[y,z] = value
    priect_image = np.rot90(priect_image, 1, (0, 1))
    priect_image = _Crop(priect_image)
    priect_image = cv2.applyColorMap(priect_image.astype('uint8'), cv2.COLORMAP_JET)
    priect_image = cv2.resize(priect_image, (64, 64), interpolation=cv2.INTER_CUBIC)
    coord = np.where(priect_image == 128)
    priect_image[coord]=0
    return priect_image.astype('uint8')



