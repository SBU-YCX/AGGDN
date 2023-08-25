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


def fetch_mat(file_path):
    """
    :param file_path:
    :return:
    """
    mat = sio.loadmat(file_path)
    topology = mat["Topology_MG"]
    trajSet = mat["TrajSet_MG"]
    ts, x_cDERs, x_iDERs, x_iLOADs, x_iLINEs, us = trajSet['t'][0, 0][0], trajSet['x_cDER'][0, 0][0], \
                                                   trajSet['x_iDER'][0, 0][0], trajSet['x_iLOAD'][0, 0][0], \
                                                   trajSet['x_iLINE'][0, 0][0], trajSet['u'][0, 0][0]
    #print(x_cDERs.shape, trajSet['t'][0, 0][1].shape)
    return ts, x_cDERs, x_iDERs, x_iLOADs, x_iLINEs, us, topology


def transform_rtds(segment_len=100, main_path = "datasets/RTDS", hdf5_path = "datasets/RTDS/rtds_noisy_len100.hdf5"):
    """
    :param main_path: the main path that saves the dataset.
    :return:
    """
    file_lists = [os.path.join(main_path, 'data1', 'NMG_4MG_topo{}', "data_Gaussian.mat").format(n) for n in range(1)]
    source_nodes = [0, 5, 12, 24, 32]
    load_nodes = []
    for n in range(33):
        if n not in source_nodes:
            load_nodes.append(n)
    with h5py.File(hdf5_path, 'w') as Dataset:
        data_idx = 0
        top_idx = 0
        current_min = float('inf')
        current_max = -float('inf')
        node_current_sum = 0
        node_current_sum_square = 0
        edge_current_sum = 0
        edge_current_sum_square = 0
        der_sum = 0
        der_sum_square = 0
        der_min = float('inf')
        der_max = -float('inf')
        num_nodes = 0
        num_edges = 0
        num_ders = 0
        max_num_edges = 0
        #
        for file_path in file_lists:
            max_num_edges = max(max_num_edges, fetch_mat(file_path)[-1].shape[0])
        #
        for file_path in file_lists:
            ts, x_cDERs, x_iDERs, x_iLOADs, x_iLINEs, us, topology = fetch_mat(file_path)
            Dataset.create_dataset('topologies_node_{}'.format(top_idx), data=topology - 1, chunks=True)
            # build edge graph.
            line_graph = []
            for i, edge_i in enumerate(topology):
                for j, edge_j in enumerate(topology):
                    if i >= j:
                        continue
                    share_node = bool(set(edge_i).intersection(set(edge_j)))
                    if share_node:
                        line_graph.append([i, j])
            line_graph = np.asarray(line_graph, dtype=np.uint8)
            Dataset.create_dataset('topologies_edge_{}'.format(top_idx), data=line_graph, chunks=True)
            Dataset.create_dataset('topologies_#node_{}'.format(top_idx), data=topology.max())
            Dataset.create_dataset('topologies_#edge_{}'.format(top_idx), data=topology.shape[0])
            for t, cDER, iDER, iLOAD, iLINE, u in zip(ts, x_cDERs, x_iDERs, x_iLOADs, x_iLINEs, us):
                #
                start = 0
                while start + segment_len <= t.shape[0]:
                    end = start + segment_len
                    node_currents = np.zeros((segment_len, 33, 2))
                    edge_currents = np.zeros((segment_len, topology.shape[0], 2))
                    #print(iDER.shape, iDER[start:end, :].shape)
                    node_currents[0:end - start, source_nodes, :] = iDER[start:end, :].reshape((-1, 5, 2))
                    node_currents[0:end - start, load_nodes, :] = iLOAD[start:end, :].reshape((-1, 28, 2))
                    edge_currents[0:end - start, :, :] = iLINE[start:end, :].reshape((-1, topology.shape[0], 2))
                    der = np.zeros((segment_len, 34))
                    der[0:end - start, :] = cDER[start:end, :]
                    # compute mean and std.
                    node_current_sum += node_currents[0:end - start, :, :].sum(0)
                    node_current_sum_square += np.power(node_currents[0:end - start, :, :], 2).sum(0)
                    num_nodes += (end - start)
                    #
                    temp = np.zeros((segment_len, max_num_edges, 2))
                    temp[0:end - start, 0:topology.shape[0], :] = edge_currents
                    edge_current_sum += temp[0:end - start, :, :].sum(0)
                    edge_current_sum_square += np.power(temp[0:end - start, :, :], 2).sum(0)
                    temp = np.zeros(max_num_edges)
                    temp[0:topology.shape[0]] = 1.0
                    num_edges += (end - start) * temp
                    current_max = np.maximum(current_max, edge_currents[0:end - start, :, :].max((0, 1)))
                    current_min = np.minimum(current_min, edge_currents[0:end - start, :, :].min((0, 1)))
                    #
                    der_sum += der[0:end - start, :].sum(0)
                    der_sum_square += np.power(der[0:end - start, :], 2).sum(0)
                    num_ders += (end - start)
                    der_max = np.maximum(der_max, der[0:end - start, :].max(0))
                    der_min = np.minimum(der_min, der[0:end - start, :].max(0))
                    # save.
                    Dataset.create_dataset('node_currents_{}'.format(data_idx), data=node_currents, chunks=True,
                                           compression="gzip")
                    Dataset.create_dataset('edge_currents_{}'.format(data_idx), data=edge_currents, chunks=True,
                                           compression="gzip")
                    Dataset.create_dataset('topologies_{}'.format(data_idx), data=top_idx)
                    Dataset.create_dataset('ts_{}'.format(data_idx), data=t[start:end, 0])
                    Dataset.create_dataset('der_{}'.format(data_idx), data=der, chunks=True,
                                           compression="gzip")
                    start += segment_len
                    data_idx += 1
                # if True:
                #     break
            print(file_path)
            top_idx += 1
        Dataset.create_dataset('num_seg', data=data_idx)
        Dataset.create_dataset('max_num_edges', data=max_num_edges)
        # compute mean and std.
        node_current_mean = node_current_sum / num_nodes
        node_current_std = np.sqrt(node_current_sum_square / num_nodes - node_current_mean ** 2)
        edge_current_mean = edge_current_sum / num_nodes
        edge_node_current_std = np.sqrt(edge_current_sum_square / num_nodes - edge_current_mean ** 2)
        Dataset.create_dataset('current_mean_v', data=node_current_mean)
        Dataset.create_dataset('current_std_v', data=node_current_std)
        Dataset.create_dataset('current_mean_e', data=edge_current_mean)
        Dataset.create_dataset('current_std_e', data=edge_node_current_std)
        #
        der_mean = der_sum /num_ders
        der_std = np.sqrt(der_sum_square / num_ders - der_mean ** 2)
        Dataset.create_dataset('der_mean', data=der_mean)
        Dataset.create_dataset('der_std', data=der_std)
        Dataset.create_dataset('der_max', data=der_max)
        Dataset.create_dataset('der_min', data=der_min)
        print(data_idx, max_num_edges)
    return


def transform_rtds_points(num_masks_per_seg=25, train_ratio=0.8, valid_ratio=0.9, spatial_p=1.0, temporal_p=1.0, downsampling=1,
                          hdf5_path="datasets/RTDS/rtds_noisy_len100_missing_points_spatial10_temporal10_downsampling1.hdf5",
                          data_path="datasets/RTDS/rtds_noisy_len100.hdf5"):
#def transform_rtds_points(num_masks_per_seg=50, train_ratio=0.8, valid_ratio=0.9, p = 0.7, downsampling=1, 
#                          hdf5_path="datasets/RTDS/rtds_noisy_len100_missing7_downsampling1.hdf5",
#                          data_path="datasets/RTDS/rtds_noisy_len100.hdf5"):
    with h5py.File(hdf5_path, 'w') as Dataset:
        raw_data = h5py.File(data_path, 'r')
        num_segs = raw_data['num_seg'][()]
        node_value_shape = list(raw_data['node_currents_{}'.format(0)][()].shape)
        node_value_shape[0] = (node_value_shape[0] - 1) // downsampling + 1
        max_num_edges = raw_data['max_num_edges'][()]
        edge_value_shape = [node_value_shape[0], max_num_edges, node_value_shape[2]]
        # Create Dataset.
        Dataset.create_dataset('train/node_masks', (1, *node_value_shape), maxshape=(None, *node_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('valid/node_masks', (1, *node_value_shape), maxshape=(None, *node_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('test/node_masks', (1, *node_value_shape), maxshape=(None, *node_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('train/edge_masks', (1, *edge_value_shape), maxshape=(None, *edge_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('valid/edge_masks', (1, *edge_value_shape), maxshape=(None, *edge_value_shape),
                               dtype=np.bool, chunks=True, compression="gzip")
        Dataset.create_dataset('test/edge_masks', (1, *edge_value_shape), maxshape=(None, *edge_value_shape),
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
                
                #node_spatial_mask = np.random.binomial(1, spatial_p, size=node_value_shape)
                temporal_mask = np.zeros(shape=(node_value_shape[0], 1, 1))
                obs_idx = np.random.choice(np.arange(0, node_value_shape[0]),
                                           size=min(int(node_value_shape[0] * temporal_p), node_value_shape[0] - 1), replace = False)
                temporal_mask[obs_idx] = 1.0
                node_spatial_mask = np.zeros(shape=node_value_shape)
                for t in range(0, seq_len):
                    s_obs_idx = np.random.choice(np.arange(0, node_value_shape[1]), size=int(node_value_shape[1] * spatial_p), replace = False)
                    node_spatial_mask[t, s_obs_idx, :] = 1.0
                mask = node_spatial_mask * temporal_mask

                '''
                mask = np.zeros(shape = node_value_shape)
                for t in range(1, seq_len): 
                    obs_idx = np.random.choice(np.arange(0, node_value_shape[1]), size = int(node_value_shape[1] * p), replace = False)
                    mask[t, obs_idx, :] = 1.0
                '''
                mask[0, :, :] = 1.0
                mask = mask.astype(dtype=np.bool)
                
                #print(node_value_shape, node_spatial_mask.shape, temporal_mask.shape, mask.shape)
                # generate edge masks.
                true_edge_value_shape = list(raw_data['edge_currents_{}'.format(data_idx)][()].shape)
                true_edge_value_shape[0] = (true_edge_value_shape[0] - 1) // downsampling + 1
                

                edge_spatial_mask = np.zeros(shape=edge_value_shape)
                for t in range(0, seq_len): 
                    se_obs_idx = np.random.choice(np.arange(0, edge_value_shape[1]), size=int(edge_value_shape[1] * spatial_p), replace = False)
                    edge_spatial_mask[t, se_obs_idx, :] = 1.0
                #edge_spatial_mask[:, 0:true_edge_value_shape[1], :] = np.random.binomial(1, spatial_p, size=true_edge_value_shape)
                edge_mask = edge_spatial_mask * temporal_mask
                '''
                edge_mask = np.zeros(shape = edge_value_shape)
                for t in range(1, seq_len): 
                    obs_idx = np.random.choice(np.arange(0, edge_value_shape[1]), size = int(edge_value_shape[1] * p), replace = False)
                    edge_mask[t, obs_idx, :] = 1.0
                '''
                edge_mask[0, :, :] = 1.0
                edge_mask = edge_mask.astype(dtype=np.bool)
                # split.
                rand = np.random.uniform(0, 1.0001)
                if rand < train_ratio:
                    # save to train.
                    Dataset['train/node_masks'].resize((trainEND + 1, *node_value_shape))
                    Dataset['train/node_masks'][trainEND:trainEND + 1, :, :] = np.expand_dims(mask, 0)
                    Dataset['train/edge_masks'].resize((trainEND + 1, *edge_value_shape))
                    Dataset['train/edge_masks'][trainEND:trainEND + 1, :, :] = np.expand_dims(edge_mask, 0)
                    Dataset['train/data_idx'].resize((trainEND + 1, ))
                    Dataset['train/data_idx'][trainEND:trainEND + 1] = data_idx
                    trainEND += 1
                elif rand < valid_ratio:
                    # save to valid.
                    Dataset['valid/node_masks'].resize((validEND + 1, *node_value_shape))
                    Dataset['valid/node_masks'][validEND:validEND + 1, :, :] = np.expand_dims(mask, 0)
                    Dataset['valid/edge_masks'].resize((validEND + 1, *edge_value_shape))
                    Dataset['valid/edge_masks'][validEND:validEND + 1, :, :] = np.expand_dims(edge_mask, 0)
                    Dataset['valid/data_idx'].resize((validEND + 1,))
                    Dataset['valid/data_idx'][validEND:validEND + 1] = data_idx
                    validEND += 1
                else:
                    # save to test.
                    Dataset['test/node_masks'].resize((testEND + 1, *node_value_shape))
                    Dataset['test/node_masks'][testEND:testEND + 1, :, :] = np.expand_dims(mask, 0)
                    Dataset['test/edge_masks'].resize((testEND + 1, *edge_value_shape))
                    Dataset['test/edge_masks'][testEND:testEND + 1, :, :] = np.expand_dims(edge_mask, 0)
                    Dataset['test/data_idx'].resize((testEND + 1,))
                    Dataset['test/data_idx'][testEND:testEND + 1] = data_idx
                    testEND += 1
                print(data_idx, '\t', m)
    return



class RTDS_data(Dataset):
    def __init__(self, split, mask_path, downsampling=1, raw_data_path="datasets/RTDS/rtds_noisy_len100.hdf5", delta_t=0.01, noise_ratio=0, 
                    return_Y=False, return_Edge=False):
        """
        :param split: train/valid/test
        :param mask_path:  the file path that saves the binary masks.
        :param downsampling:  the interval to downsampling the data trajectories (mask should be already downsampled)
        :param raw_data_path:  the file path that saves the raw data.
        :param delta_t: time step.
        :param return_Y:  whether return the DER controller signals (for supervised learning task).
        """
        super(RTDS_data, self).__init__()
        assert split in ['train', 'valid', 'test'], "mode should be in ['train', 'valid', 'test']."
        self.raw_data = h5py.File(raw_data_path, 'r')
        self.mean_v = self.raw_data['current_mean_v'][()]
        self.std_v = self.raw_data['current_std_v'][()]
        self.mean_e = self.raw_data['current_mean_e'][()]
        self.std_e = self.raw_data['current_std_e'][()]
        mask_data = h5py.File(mask_path, 'r')
        self.node_masks = mask_data['{}/node_masks'.format(split)]
        self.edge_masks = mask_data['{}/edge_masks'.format(split)]
        self.data_idx = mask_data['{}/data_idx'.format(split)]
        self.size = self.node_masks.shape[0]
        self.downsampling = downsampling
        self.upsampling = 0.01 / delta_t
        self.split = split
        self.der = return_Y
        self.return_edge = return_Edge
        self.delta_t = delta_t
        self.noise_ratio = noise_ratio
        return

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        train_data_index = self.data_idx[idx]
        max_num_edge = self.raw_data['max_num_edges'][()]
        X = (self.raw_data['node_currents_{}'.format(train_data_index)][()] - self.mean_v) / self.std_v
        #X = self.raw_data['node_currents_{}'.format(train_data_index)][()]
        if self.noise_ratio != 0: 
            #noise_scale = self.noise_ratio * (X.max() - X.min()) / 2
            X_noise = np.random.normal(0, self.noise_ratio, size = X.shape)
            #X_noise = stats.truncnorm(X.min() * self.noise_ratio, X.max() * self.noise_ratio, loc=0, scale=1, size=X.shape)
            X = X + X_noise
        E_t = self.raw_data['edge_currents_{}'.format(train_data_index)][()]
        E = np.zeros(shape=(X.shape[0], max_num_edge, X.shape[-1]))
        E[:, 0:E_t.shape[1], :] = E_t
        E = (E - self.mean_e) / (self.std_e)
        M_x = self.node_masks[idx].astype(np.float32)
        M_e = self.edge_masks[idx].astype(np.float32)
        t = self.raw_data['ts_{}'.format(train_data_index)][()]
        t = np.arange(0, X.shape[0] * self.upsampling) * self.delta_t
        # up sampling.
        if self.upsampling != 1: 
            up_sampling_idx = np.arange(0, X.shape[0] * self.upsampling, self.upsampling).astype(np.int32)
            t = t[up_sampling_idx].astype(np.float32)
            #newX = [], newE = [], idx = 0
            #for n, dt in enumerate(t): 
            #    if n == up_sampling_idx[idx]: 
            #        newX.append(X[idx])
            #        newE.append(E[idx])
            #        idx += 1
            #    else:
            #        newX.append(torch.zeros_like(X[0]))
            #        newE.append(torch.zeros_like(E[0]))
            #X = torch.stack(newX, 0).astype(np.float32)
            #E = torch.stack(newE, 0).astype(np.float32)
        # down sampling.
        down_sampling_idx = np.arange(0, X.shape[0], self.downsampling)
        X, E, t = X[down_sampling_idx].astype(np.float32), E[down_sampling_idx].astype(np.float32), \
                  t[down_sampling_idx].astype(np.float32)
        # construct the adjacent matrices.
        edges = self.raw_data['topologies_node_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        #print(edges.shape)
        i = torch.LongTensor(edges)
        v = torch.FloatTensor(torch.ones(i.size(0)))
        adjacent_nodes = torch.sparse.FloatTensor(i.t(), v, torch.Size([33, 33])).to_dense()
        adjacent_nodes += adjacent_nodes.t()
        # Normalized Laplacian Matrix
        '''
        adjacent_nodes += torch.eye(adjacent_nodes.size(0))
        degree_nodes = torch.FloatTensor(adjacent_nodes.sum(1))
        degree_nodes = torch.diag(torch.pow(degree_nodes, -0.5))
        adjacent_nodes = degree_nodes.mm(adjacent_nodes).mm(degree_nodes)
        '''
        #
        nodes = self.raw_data['topologies_edge_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        #print(nodes.shape)
        #print(nodes.max())
        i = torch.LongTensor(nodes)
        v = torch.FloatTensor(torch.ones(i.size(0)))
        adjacent_edges = torch.sparse.FloatTensor(i.t(), v, torch.Size([max_num_edge, max_num_edge])).to_dense()
        adjacent_edges += adjacent_edges.t()
        # Normalized Laplacian Matrix
        '''
        adjacent_edges += torch.eye(adjacent_edges.size(0))
        degree_edges = torch.FloatTensor(adjacent_edges.sum(1))
        degree_edges = torch.diag(torch.pow(degree_edges, -0.5))
        adjacent_edges = degree_edges.mm(adjacent_edges).mm(degree_edges)
        '''
        # transformation matrix.
        edge_node_pairs = np.asarray([[[i, v[0]], [i, v[1]]] for i, v in enumerate(edges)], dtype=np.uint8).reshape((-1, 2))
        i = torch.LongTensor(edge_node_pairs)
        v = torch.FloatTensor(torch.ones(i.size(0)))
        T = torch.sparse.FloatTensor(i.t(), v, torch.Size([max_num_edge, 33])).to_dense()
        # numbers of nodes and edges.
        num_nodes = self.raw_data['topologies_#node_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        num_edges = self.raw_data['topologies_#edge_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        if self.der:
            der = self.raw_data['der_{}'.format(train_data_index)][()]
            mean, std = self.raw_data['der_mean'][()], self.raw_data['der_std'][()]
            der = (der[down_sampling_idx] - mean) / std
            if self.return_edge:
                return {'t': t, 'adjacent_matrices': adjacent_edges, 'values': E, 'masks': M_e,
                        '#nodes': num_edges, 'Ys': der, 'ori_values': E}
            else:
                return {'t': t, 'adjacent_matrices': adjacent_nodes, 'values': X, 'masks': M_x,
                        '#nodes': num_nodes, 'Ys': der, 'ori_values': X}
        else:
            if self.return_edge:
                return {'t': t, 'adjacent_matrices': adjacent_edges, 'values': E, 'masks': M_e,
                        '#nodes': num_edges, 'mean': self.mean_v, 'std': self.std_v, 'ori_values': E}
            else:
                return {'t': t, 'adjacent_matrices': adjacent_nodes, 'values': X, 'masks': M_x,
                        '#nodes': num_nodes, 'mean': self.mean_v, 'std': self.std_v, 'ori_values': X}


#transform_rtds()
#transform_rtds_points()
# data = RTDS_data('train', "datasets/RTDS/rtds_data_missing_points_spatial6_temporal3_downsampling2.hdf5", return_der=True)
#
# for data_batch in data:
#     print()