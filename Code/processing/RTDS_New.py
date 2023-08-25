"""
Author: Yingru Liu
Data pre-processing of RTDS data.
"""
import h5py, glob, os
import scipy.io as sio
import scipy.stats as stats
import numpy as np
import torch
from torch.utils.data import Dataset


def fetch_data(file_path, topo_path):
    """
    :param file_path:
    :return:
    """
    ts, vs = [], []
    with open(file_path) as f: 
        data = np.loadtxt(f, float, delimiter = ',', skiprows = 1)
    ts = data[:, 0]
    vs = data[:, 1:]
    with open(topo_path) as g:
        topo = np.loadtxt(g, int, delimiter = ',')
    return ts, vs, topo


def transform_rtds(segment_len=100, nf = 3, dr = 10, main_path = "datasets/RTDS_Real/Raw_Data/Bus_Voltage/", hdf5_path = "datasets/RTDS_Real/Processed_Data/Bus_Voltage/rtds_len100_p10.hdf5"):
    """
    :param main_path: the main path that saves the dataset.
    :return:
    """
    file_lists = [os.path.join(main_path, 'area{}_island_voltage_p{}.csv').format(n, dr) for n in range(1, 2)]
    topo_lists = [os.path.join(main_path, 'area{}_island_voltage_topo.csv').format(n) for n in range(1, 2)]
    with h5py.File(hdf5_path, 'w') as Dataset: 
        data_idx = 0
        topo_idx = 0
        node_values_sum = 0
        node_values_sum_square = 0
        node_num = 0
        for file_path, topo_path in zip(file_lists, topo_lists):
            ts, vs, topo = fetch_data(file_path, topo_path)
            Dataset.create_dataset('topologies_node_{}'.format(topo_idx), data = topo - 1, chunks = True)
            nt, nn = vs.shape[0], vs.shape[1] // nf
            start = 0
            Dataset.create_dataset('topologies_#node_{}'.format(topo_idx), data = topo.max())
            while start + segment_len <= nt: 
                end = start + segment_len
                node_values = vs[start : end, :].reshape(segment_len, nn, nf)
                node_values_sum += node_values.sum(0)
                node_values_sum_square += np.power(node_values, 2).sum(0)
                node_num += (end - start)
                Dataset.create_dataset('node_values_{}'.format(data_idx), data = node_values, chunks = True, compression = 'gzip')
                Dataset.create_dataset('topologies_{}'.format(data_idx), data = topo_idx)
                Dataset.create_dataset('ts_{}'.format(data_idx), data = ts[start : end])
                start += segment_len
                data_idx += 1
            topo_idx += 1
        Dataset.create_dataset('num_seg', data = data_idx)
        node_values_mean = node_values_sum / node_num
        node_values_std = np.sqrt(node_values_sum_square / node_num - node_values_mean ** 2)
        Dataset.create_dataset('values_mean', data = node_values_mean)
        Dataset.create_dataset('values_std', data = node_values_std)
    return


def transform_rtds_points(num_masks_per_seg=25, train_ratio=0.8, valid_ratio=0.9, p = 0.9, downsampling=1, 
                          hdf5_path="datasets/RTDS_Real/Processed_Data/Bus_Voltage/rtds_len100_p10_missing9_downsampling1.hdf5",
                          data_path="datasets/RTDS_Real/Processed_Data/Bus_Voltage/rtds_len100_p10.hdf5"):
    with h5py.File(hdf5_path, 'w') as Dataset:
        raw_data = h5py.File(data_path, 'r')
        num_segs = raw_data['num_seg'][()]
        node_value_shape = list(raw_data['node_values_{}'.format(0)][()].shape)
        node_value_shape[0] = (node_value_shape[0] - 1) // downsampling + 1
        # Create Dataset.
        Dataset.create_dataset('train/node_masks', (1, *node_value_shape), maxshape=(None, *node_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('valid/node_masks', (1, *node_value_shape), maxshape=(None, *node_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('test/node_masks', (1, *node_value_shape), maxshape=(None, *node_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('train/data_idx', (1, ), maxshape=(None, ), chunks=True, dtype=np.int32,
                               compression="gzip")
        Dataset.create_dataset('valid/data_idx', (1, ), maxshape=(None, ), chunks=True, dtype=np.int32,
                               compression="gzip")
        Dataset.create_dataset('test/data_idx', (1, ), maxshape=(None, ), chunks=True, dtype=np.int32,
                               compression="gzip")
        trainEND = 0
        validEND = 0
        testEND = 0
        #
        for data_idx in range(num_segs):
            for m in range(num_masks_per_seg):
                # generate node masks.
                seq_len = node_value_shape[0]
                '''
                node_spatial_mask = np.random.binomial(1, spatial_p, size=node_value_shape)
                temporal_mask = np.zeros(shape=(node_value_shape[0], 1, 1))
                obs_idx = np.random.choice(np.arange(1, node_value_shape[0]),
                                           size=min(int(node_value_shape[0] * temporal_p), node_value_shape[0] - 1), replace = False)
                temporal_mask[obs_idx] = 1.0
                mask = node_spatial_mask * temporal_mask
                '''
                mask = np.zeros(shape = node_value_shape)
                for t in range(1, seq_len): 
                    obs_idx = np.random.choice(np.arange(0, node_value_shape[1]), size = int(node_value_shape[1] * p), replace = False)
                    mask[t, obs_idx, :] = 1.0

                mask[0, :, :] = 1.0
                mask = mask.astype(dtype=np.bool)
                # split.
                rand = np.random.uniform(0, 1.0001)
                if rand < train_ratio:
                    # save to train.
                    Dataset['train/node_masks'].resize((trainEND + 1, *node_value_shape))
                    Dataset['train/node_masks'][trainEND:trainEND + 1, :, :] = np.expand_dims(mask, 0)
                    Dataset['train/data_idx'].resize((trainEND + 1, ))
                    Dataset['train/data_idx'][trainEND:trainEND + 1] = data_idx
                    trainEND += 1
                elif rand < valid_ratio:
                    # save to valid.
                    Dataset['valid/node_masks'].resize((validEND + 1, *node_value_shape))
                    Dataset['valid/node_masks'][validEND:validEND + 1, :, :] = np.expand_dims(mask, 0)
                    Dataset['valid/data_idx'].resize((validEND + 1,))
                    Dataset['valid/data_idx'][validEND:validEND + 1] = data_idx
                    validEND += 1
                else:
                    # save to test.
                    Dataset['test/node_masks'].resize((testEND + 1, *node_value_shape))
                    Dataset['test/node_masks'][testEND:testEND + 1, :, :] = np.expand_dims(mask, 0)
                    Dataset['test/data_idx'].resize((testEND + 1,))
                    Dataset['test/data_idx'][testEND:testEND + 1] = data_idx
                    testEND += 1
                print(data_idx, '\t', m)
    return



class RTDSReal_data(Dataset):
    def __init__(self, split, mask_path, downsampling=1, raw_data_path="datasets/RTDS_Real/Processed_Data/Bus_Voltage/rtds_len100.hdf5", delta_t=0.01, noise_ratio=0, 
                    return_Y=False, data_rate = 1):
        """
        :param split: train/valid/test
        :param mask_path:  the file path that saves the binary masks.
        :param downsampling:  the interval to downsampling the data trajectories (mask should be already downsampled)
        :param raw_data_path:  the file path that saves the raw data.
        :param delta_t: time step.
        :param return_Y:  whether return the DER controller signals (for supervised learning task).
        """
        super(RTDSReal_data, self).__init__()
        assert split in ['train', 'valid', 'test'], "mode should be in ['train', 'valid', 'test']."
        self.raw_data = h5py.File(raw_data_path.split('.')[0] + '_p{}'.format(data_rate) + '.hdf5', 'r')
        self.mean = self.raw_data['values_mean'][()]
        self.std = self.raw_data['values_std'][()]
        mask_data = h5py.File(mask_path, 'r')
        self.node_masks = mask_data['{}/node_masks'.format(split)]
        self.data_idx = mask_data['{}/data_idx'.format(split)]
        self.size = self.node_masks.shape[0]
        self.downsampling = downsampling
        self.upsampling = data_rate // delta_t
        self.split = split
        self.der = return_Y
        self.data_rate = data_rate * 2.5e-3
        self.delta_t = delta_t
        self.noise_ratio = noise_ratio
        return

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        train_data_index = self.data_idx[idx]
        X = (self.raw_data['node_values_{}'.format(train_data_index)][()] - self.mean) / self.std
        if self.noise_ratio != 0: 
            X_noise = np.random.normal(0, self.noise_ratio, size = X.shape)
            X = X + X_noise
        M_x = self.node_masks[idx].astype(np.float32)
        t = self.raw_data['ts_{}'.format(train_data_index)][()] * 100
        t = np.arange(t[0] * self.upsampling, t[-1] * self.upsampling + 1e-4 * self.data_rate, self.data_rate) * (1 / self.upsampling)
        # up sampling.
        if self.upsampling != 1: 
            up_sampling_idx = np.arange(0, t.shape[0], self.upsampling, dtype = np.int32)
            t = t[up_sampling_idx].astype(np.float32)
        # down sampling.
        down_sampling_idx = np.arange(0, X.shape[0], self.downsampling, dtype = np.int32)
        X, t = X[down_sampling_idx].astype(np.float32), t[down_sampling_idx].astype(np.float32)
        # numbers of nodes
        num_nodes = self.raw_data['topologies_#node_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        # construct the adjacent matrices.
        edges = self.raw_data['topologies_node_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        i = torch.LongTensor(edges)
        v = torch.FloatTensor(torch.ones(i.size(0)))
        adjacent_nodes = torch.sparse.FloatTensor(i.t(), v, torch.Size([num_nodes, num_nodes])).to_dense()
        adjacent_nodes += adjacent_nodes.t()
        return {'t': t, 'adjacent_matrices': adjacent_nodes, 'values': X, 'masks': M_x,
                        '#nodes': num_nodes}


#transform_rtds()
#transform_rtds_points()
# data = RTDS_data('train', "datasets/RTDS/rtds_data_missing_points_spatial6_temporal3_downsampling2.hdf5", return_der=True)
#
# for data_batch in data:
#     print()