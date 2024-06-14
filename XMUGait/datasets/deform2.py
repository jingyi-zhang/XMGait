import numpy as np
import pickle
import math, cv2, os, h5py
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

    if W < H:
        dist_W = H - W
        if dist_W % 2 == 0:
            img = np.pad(img, ((int((dist_W) / 2), int((dist_W) / 2)), (0, 0), (0, 0)),
                         'constant', constant_values=0)
        else:
            img = np.pad(img, (
            (int((dist_W - 1) / 2) + 1, int((dist_W - 1) / 2)), (0, 0), (0, 0)),
                         'constant',
                         constant_values=0)
    if H < W:
        dist_H = W - H
        if dist_H % 2 == 0:
            img = np.pad(img, ((0, 0), (int((dist_H) / 2), int((dist_H) / 2)), (0, 0)),
                         'constant', constant_values=0)
        else:
            img = np.pad(img, (
            (0, 0), (int((dist_H - 1) / 2) + 1, int((dist_H - 1) / 2)), (0, 0)),
                         'constant',
                         constant_values=0)
            # img = np.pad(img, (int((dist_H - 1) / 2) + 1, int((dist_H - 1) / 2), 0, 0, 0, 0), 'constant', 
            #             value=0) 

    return img


def point2voxel(point,grid_size,origin,voxel_spacing,displacement_vectors):

    Mx,My,Mz = grid_size[0],grid_size[1],grid_size[2]
    voxel_disx = np.zeros((grid_size[1],grid_size[2]))
    voxel_disy = np.zeros((grid_size[1],grid_size[2]))
    voxel_disz = np.zeros((grid_size[1],grid_size[2]))

    #voxel_coordx = np.zeros((grid_size[1], grid_size[2]))
    #voxel_coordy = np.zeros((grid_size[1], grid_size[2]))
    #voxel_coordz = np.zeros((grid_size[1], grid_size[2]))

    voxel_count = np.ones((grid_size[1], grid_size[2]))

    norm_point = (point-origin)/voxel_spacing


    for i in range(norm_point.shape[0]):
        corrd = norm_point[i]
        coordx,coordy,coordz = math.floor(corrd[0]),math.floor(corrd[1]),math.floor(corrd[2])
        coordx = min(coordx, Mx-1)
        coordy = min(coordy, My-1)
        coordz = min(coordz, Mz-1)




        if voxel_disx[ coordy, coordz]==0:
            voxel_disx[coordy,  coordz] = displacement_vectors[i][0]
            voxel_disy[coordy,  coordz] = displacement_vectors[i][1]
            voxel_disz[coordy,  coordz] = displacement_vectors[i][2]

            #voxel_coordx[coordy, coordz] = coordx
            #voxel_coordy[coordy, coordz] = coordy
            #voxel_coordz[coordy, coordz] = coordz

        else:
            voxel_disx[coordy, coordz] =  voxel_disx[coordy, coordz] + displacement_vectors[i][0]
            voxel_disy[coordy, coordz] =  voxel_disy[coordy, coordz] + displacement_vectors[i][1]
            voxel_disz[coordy, coordz] =  voxel_disz[coordy, coordz] + displacement_vectors[i][2]
            #voxel_coordx[coordy, coordz] =  voxel_coordx[coordy, coordz]+coordx
            #voxel_coordy[coordy, coordz] = voxel_coordy[coordy, coordz]+coordy
            #voxel_coordz[coordy, coordz] = voxel_coordz[coordy, coordz]+coordz

            voxel_count[coordy, coordz] += 1

    voxel_disx = voxel_disx / voxel_count
    voxel_disy = voxel_disy / voxel_count
    voxel_disz = voxel_disz / voxel_count

    #voxel_coordx = voxel_coordx / voxel_count
    #voxel_coordy = voxel_coordy / voxel_count
    #voxel_coordz = voxel_coordz / voxel_count

    voxel_disx = np.expand_dims(voxel_disx, -1)
    voxel_disy = np.expand_dims(voxel_disy, -1)
    voxel_disz = np.expand_dims(voxel_disz, -1)

    #voxel_coordx = np.expand_dims(voxel_coordx, -1)
    #voxel_coordy = np.expand_dims(voxel_coordy, -1)
    #voxel_coordz = np.expand_dims(voxel_coordz, -1)

    final_dis = np.concatenate([voxel_disx, voxel_disy,voxel_disz], -1)
    #final_coord = np.concatenate([voxel_coordx, voxel_coordy, voxel_coordz], -1)

    #print (np.abs(displacement_vectors).max(),np.abs(displacement_vectors).min())

    return final_dis#,final_coord


spacing = [0.02, 0.02, 0.02]
Min = -0.4 
Max = 0.4 


def project_one_sample(input_array):
    source_points = input_array[:,:3]
    displacement_vectors = input_array[:, 3:]
    
    displacement_vectors = (displacement_vectors-Min)/(Max-Min)*255 

    size, origin, center = get_size(source_points, spacing) 
    
    priect_image = point2voxel(source_points, size, origin, spacing,displacement_vectors)
    priect_image = np.rot90(priect_image, 1, (0, 1)) 
    priect_image = _Crop(priect_image) 
    priect_image = cv2.resize(priect_image, (64, 64)) 
    
    return priect_image
