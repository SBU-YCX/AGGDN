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


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """


    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def readData(file_path, pkl_path, saveto, train_ratio=0.8, valid_ratio=0.9, spatial_p=0.6, temporal_p=0.4):
    # adj.
    with open(pkl_path, 'rb') as f:
        adj = pickle.load(f, encoding='latin1')[-1]
    # data
    dimFrame  = 12
    dimDay = 24
    data = pd.read_hdf(file_path).values
    print(data.shape)
    return
    data_in_hours, indx = [],  0
    # for item in data.T:
    #     x = smooth(item)
    #     print()
    while indx < data.shape[0]:
        data_in_hours.append(data[indx:indx+dimFrame, :])
        indx += dimFrame
    data_in_hours = np.stack(data_in_hours, 0)
    data_in_days, indx = [], 0
    while indx + dimDay < data_in_hours.shape[0]:
        data_in_days.append(data_in_hours[indx:indx + dimDay, :, :])
        indx += dimDay
    data_in_days = np.stack(data_in_days, 0)
    data_in_days = np.reshape(data_in_days, (data_in_days.shape[0], -1, data_in_days.shape[-1]))
    data_smooth = np.zeros_like(data_in_days)
    # smoothing.
    for i in range(data_in_days.shape[0]):
        for j in range(data_in_days.shape[-1]):
            data_smooth[i, :, j] = smooth(data_in_days[i, :, j])[0:-10]
    # mask.
    masks = []
    data_idx = []
    v_sum, v_sum_square, num_sample = 0, 0, 0
    mean, std = data_in_days.mean(keepdims=True), data_in_days.std(keepdims=True)#.mean((0, 1), keepdims=True), data_in_days.std((0, 1), keepdims=True)
    data = []
    #print(mean.shape, data_in_days.shape)
    for item in data_in_days:
        idx = 0
        while idx + 72 < item.shape[0]:
            data.append(item[idx:idx+72])
            v_sum += item[idx:idx+72, :].sum(0, keepdims=True)
            v_sum_square += np.power(item[idx:idx+72, :], 2).sum(0, keepdims=True)
            num_sample += 72
            idx += 72
    data_in_days = np.expand_dims(np.stack(data, 0), -1)
    #print(data_in_days.shape, data_in_hours.shape)
    #return
    for i in range(10):
        print(i, data_in_days.shape)
        ms = []
        for n in range(data_in_days.shape[0]):
            temporal_mask = np.zeros(shape=(data_in_days.shape[1], 1, 1))
            obs_idx = np.random.choice(np.arange(1, data_in_days.shape[1]), size=int(data_in_days.shape[1] * temporal_p), replace = False)
            temporal_mask[obs_idx] = 1.0
            #spatial_mask = np.random.binomial(1, spatial_p, size=data_in_days.shape[1:])
            spatial_mask = np.zeros(shape=data_in_days.shape[1:])
            for t in range(0, data_in_days.shape[1]):
                s_obs_idx = np.random.choice(np.arange(0, data_in_days.shape[2]), size=int(data_in_days.shape[2] * spatial_p), replace=False)
                spatial_mask[t, s_obs_idx, :] = 1.0
            mask = spatial_mask * temporal_mask
            #mask[0, :, :] = 1.0
            ms.append(mask)
        m = np.stack(ms, 0)
        index = np.arange(0, data_in_days.shape[0])
        masks.append(m)
        data_idx.append(index)
    masks, data_idx = np.concatenate(masks, 0), np.concatenate(data_idx, 0)
    idx = np.arange(0, data_idx.shape[0])
    np.random.shuffle(idx)
    with h5py.File(saveto, "w") as f:
        f.create_dataset('masks', data=masks.astype(bool), compression="gzip")
        f.create_dataset('data', data=data_in_days.astype(np.float32), compression="gzip")
        f.create_dataset('data_idx', data=data_idx, compression="gzip")
        f.create_dataset('adj', data=adj.astype(np.float32), compression="gzip")
        #mean = v_sum / num_sample
        f.create_dataset('mean', data=mean.astype(np.float32))
        #std = np.sqrt(v_sum_square / num_sample - mean ** 2)
        f.create_dataset('std', data=std.astype(np.float32))
        f.create_dataset('train_idx', data=idx[0:int(idx.shape[0]*train_ratio)], compression="gzip")
        f.create_dataset('valid_idx', data=idx[int(idx.shape[0] * train_ratio):int(idx.shape[0] * valid_ratio)])
        f.create_dataset('test_idx', data=idx[int(idx.shape[0] * valid_ratio):], compression="gzip")
    return

class Traffic_data(Dataset):
    def __init__(self, split, data_path, delta_t, topK=207):
        super(Traffic_data, self).__init__()
        assert split in ['train', 'valid', 'test'], "mode should be in ['train', 'valid', 'test']."
        f = h5py.File(data_path, "r")
        self.adj = f['adj']
        self.data = f['data'][()]
        self.masks = f['masks']
        self.data_idx = f['data_idx']
        self.item_idx = f["{}_idx".format(split)]
        self.mean = np.expand_dims(f['mean'][()].squeeze(0), axis=2)
        self.std = np.expand_dims(f['std'][()].squeeze(0), axis=2)
        self.size = self.item_idx.shape[0]
        self.delta_t = delta_t
        #
        self.upsampling = 0.083 / delta_t
        self.topK = np.sort(np.argsort(self.data.mean((0, 1, -1)))[-topK:])
        return

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        adj = self.adj[()]
        item_idx = self.item_idx[idx]
        data_idx = self.data_idx[item_idx]
        X = (self.data[data_idx] - self.mean) / self.std
        mask = self.masks[data_idx].astype(np.float32)
        t = self.delta_t * np.arange(0, X.shape[0] * self.upsampling).astype(np.float32)
        if self.upsampling != 1:
            up_sampling_idx = np.arange(0, X.shape[0] * self.upsampling, self.upsampling).astype(np.int32)
            t = t[up_sampling_idx].astype(np.float32)
        X, mask = X[:, self.topK], mask[:, self.topK]
        adj = adj[self.topK, :][:, self.topK]
        #
        #print(X.dtype)
        return {'t': t, 'adjacent_matrices': adj, 'values': X, 'masks': mask,
                '#nodes': X.shape[-2], 'mean': self.mean, 'std': self.std}

# #
readData("datasets/Traffic/metr-la.h5",  "/home/yucxing/exp/C004/EvolveSDE/datasets/Traffic/adj_mx.pkl", 
          "datasets/Traffic/metr_spatial6_temporal4.hdf5")
readData("datasets/Traffic/pems-bay.h5",  "/home/yucxing/exp/C004/EvolveSDE/datasets/Traffic/adj_mx_bay.pkl", 
          "datasets/Traffic/pems_spatial6_temporal4.hdf5")