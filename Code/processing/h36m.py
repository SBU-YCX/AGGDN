"""
Author: Yingru Liu
Data pre-processing of RTDS data.
"""
import pickle
import h5py, glob, os
import scipy.io as sio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def readData(file_path, saveto, train_ratio=0.8, valid_ratio=0.9, spatial_p=0.6, temporal_p=0.4):
    # data.
    data = np.load(file_path)
    train_data, test_data = data['train_data'], data['test_data']
    # data
    segments = []
    for item in train_data:
        idx = 0
        while idx + 60 < item.shape[0]:
            segments.append(item[idx:idx + 60])
            idx += 60
    train_data = np.stack(segments, axis=0)
    segments = []
    for item in test_data:
        idx = 0
        while idx + 60 < item.shape[0]:
            segments.append(item[idx:idx + 60])
            idx += 60
    test_data = np.stack(segments, axis=0)
    train_data = np.concatenate([train_data, test_data], axis=0)
    # mask.
    masks = []
    data_idx = []
    # data_in_days = data_smooth
    for i in range(5):
        print(i)
        ms = []
        for n in range(train_data.shape[0]):
            '''
            mask = np.zeros(shape = train_data.shape)
            for t in range(1, train_data.shape[1]): 
                obs_idx = np.random.choice(np.arange(0, train_data.shape[2]), size = int(train_data.shape[2] * spatial_p), replace = False)
                mask[t, obs_idx, :] = 1.0
            mask[0, :, :] = 1.0
            mask = mask.astype(dtype=bool)
            '''
            temporal_mask = np.zeros(shape=(train_data.shape[1], 1, 1))
            t_obs_idx = np.random.choice(np.arange(1, train_data.shape[1]), size=min(int(train_data.shape[1] * temporal_p), train_data.shape[1] - 1), replace = False)
            temporal_mask[t_obs_idx] = 1.0
            #spatial_mask = np.random.binomial(1, spatial_p, size=train_data.shape[1:-1])
            #spatial_mask = np.expand_dims(spatial_mask, -1).repeat(3, -1)
            spatial_mask = np.zeros(shape=train_data.shape[1:])
            for t in range(0, train_data.shape[1]): 
                s_obs_idx = np.random.choice(np.arange(0, train_data.shape[2]), size = int(train_data.shape[2] * spatial_p), replace = False)
                spatial_mask[t, s_obs_idx, :] = 1.0
            mask = spatial_mask * temporal_mask
            #mask[0, :, :] = 1.0
            ms.append(mask)
            
            #ms.append(mask)
        m = np.stack(ms, 0)
        index = np.arange(0, train_data.shape[0])
        masks.append(m)
        data_idx.append(index)
    masks, data_idx = np.concatenate(masks, 0), np.concatenate(data_idx, 0)
    idx = np.arange(0, data_idx.shape[0])
    np.random.shuffle(idx)
    # adj.
    buff = (0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27)
    kin = np.array(
        [[0, 12], [12, 13], [13, 14], [15, 14], [13, 17], [17, 18], [18, 19], [13, 25], [25, 26], [26, 27], [0, 1], 
        [1, 2], [2, 3], [0, 6], [6, 7], [7, 8]]
    )
    buff_dict = {item: n for n, item in enumerate(buff)}
    adj = np.zeros((17, 17))
    for link in kin:
        adj[buff_dict[link[0]], buff_dict[link[1]]] = 1.0
    #
    adj = adj + adj.T
    with h5py.File(saveto, "w") as f:
        f.create_dataset('masks', data=masks.astype(bool), compression="gzip")
        f.create_dataset('data', data=train_data.astype(np.float32), compression="gzip")
        f.create_dataset('data_idx', data=data_idx, compression="gzip")
        f.create_dataset('adj', data=adj.astype(np.float32), compression="gzip")
        f.create_dataset('train_idx', data=idx[0:int(idx.shape[0]*train_ratio)], compression="gzip")
        f.create_dataset('valid_idx', data=idx[int(idx.shape[0] * train_ratio):int(idx.shape[0] * valid_ratio)])
        f.create_dataset('test_idx', data=idx[int(idx.shape[0] * valid_ratio):], compression="gzip")
    return

class Motion_data(Dataset):
    def __init__(self, split, data_path, delta_t, p_spatial, p_temporal):
        super(Motion_data, self).__init__()
        assert split in ['train', 'valid', 'test'], "mode should be in ['train', 'valid', 'test']."
        f = h5py.File(data_path, "r")
        self.adj = f['adj']
        self.data = f['data'][()]
        self.masks = f['masks']
        self.data_idx = f['data_idx']
        self.item_idx = f["{}_idx".format(split)]
        self.size = self.item_idx.shape[0]
        self.delta_t = delta_t
        self.upsampling = 0.01 / delta_t
        self.split = split
        self.p_spatial = p_spatial
        self.p_temporal = p_temporal
        return

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        adj = self.adj[()]
        item_idx = self.item_idx[idx]
        data_idx = self.data_idx[item_idx]
        X = self.data[data_idx]
        t = self.delta_t * np.arange(0, X.shape[0] * self.upsampling).astype(np.float32)
        if self.upsampling != 1:
            up_sampling_idx = np.arange(0, X.shape[0] * self.upsampling, self.upsampling).astype(np.int32)
            t = t[up_sampling_idx].astype(np.float32)
        mask = self.masks[data_idx].astype(np.float32)
        #
        return {'t': t, 'adjacent_matrices': adj, 'values': X, 'masks': mask,
                '#nodes': X.shape[-2]}


#readData("datasets/H36M/h36m.npz", "datasets/H36M/h36m_spatial6_temporal4.hdf5")
