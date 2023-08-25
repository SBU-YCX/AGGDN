import h5py, glob, os, csv, openpyxl, torch
import scipy.io as sio
import scipy.stats as stats
import numpy as np
from torch.utils.data import Dataset

'''
def cleanData(data_path_prefix = "/home/yucxing/exp/C004/EvolveSDE/datasets/RTDS_Real/Raw_Data/BCM_SS2/Load/Raw/BCM_SS2_LOAD", 
            raw_data_num = 10, 
            map_path_prefix = "/home/yucxing/exp/C004/EvolveSDE/datasets/RTDS_Real/Raw_Data/BCM_SS2/Load/BCM_SS2_LOAD", 
            map_path_suffix = ["C", "V"], 
            root_path = "datasets/RTDS_Real/Raw_Data/BCM_SS2/Load/Clean", 
            adj_path = "/home/yucxing/exp/C004/EvolveSDE/datasets/RTDS_Real/Raw_Data/BCM_SS2/Load/BCM_SS2_LOAD_ADJ.xlsx"): 
	with open(data_path, 'r') as data_file: 
		reader = csv.reader(data_file)
		for	line in reader: 
			break
	full_var_lst = [var.lower() for var in line]
	node_lst, need_var_lst = [], []
	map_file = openpyxl.load_workbook(map_path)["Sheet1"]
	for j in range(1, map_file.max_column): 
		node_lst.append(map_file.cell(1, j + 1).value)
	for i in range(1, map_file.max_row): 
		need_var_lst.append(str(map_file.cell(i + 1, 1).value).lower())
	idx_lst = []
	for j in range(1, map_file.max_column): 
		for i in range(1, map_file.max_row): 
			if map_file.cell(i + 1, j + 1).value == 1: 
				idx_lst.append(full_var_lst.index(need_var_lst[i - 1] + 'a'))
				idx_lst.append(full_var_lst.index(need_var_lst[i - 1] + 'b'))
				idx_lst.append(full_var_lst.index(need_var_lst[i - 1] + 'c'))
	print(node_lst, len(node_lst), len(idx_lst))
	with open(data_path, 'r') as data_file: 
		data = np.loadtxt(data_file, float, delimiter = ',', skiprows = 1, usecols = idx_lst)
	batch = 0
	maxrows = data.shape[0]
	while (batch * 30000 < maxrows):
		np.savetxt(os.path.join(root_path, 'clean_data_{}.csv'.format(batch)), data[batch * 30000: min(maxrows, (batch + 1) * 30000), :], delimiter = ',')
		batch += 1
		print(batch)
	full_node_lst = []
	adj_file = openpyxl.load_workbook(map_path)["Sheet1"]
	for j in range(1, adj_file.max_column): 
		full_node_lst.append(adj_file.cell(1, j + 1).value)
	adj_lst = []
	for node in node_lst: 
		i = full_node_lst.index(node)
		for j in range(i + 1, adj_file.max_column): 
			if adj_file.cell(i + 2, j + 1).value == 1:
				adj_lst.append([i, j - 1])
	print(adj_lst)
	np.savetxt(os.path.join(root_path, 'topology.csv'), np.array(adj_lst, dtype = int), delimiter = ',')
	return
'''


def cleanData(data_path_prefix = "/home/yucxing/exp/C004/EvolveSDE/datasets/RTDS_Real/Raw_Data/BCM_CONNECTED/Raw/DATA1", 
            raw_data_num = 5, 
            map_path_prefix = "/home/yucxing/exp/C004/EvolveSDE/datasets/RTDS_Real/Raw_Data/BCM_CONNECTED/BCM_LOAD", 
            map_path_suffix = ["C", "V", "P", "Q"], 
            root_path = "datasets/RTDS_Real/Raw_Data/BCM_CONNECTED/Clean/", 
            adj_path = "/home/yucxing/exp/C004/EvolveSDE/datasets/RTDS_Real/Raw_Data/BCM_CONNECTED/BCM_LOAD_ADJ_SIMP.xlsx"): 
    map_node_var = {}
    for suffix in map_path_suffix: 
        map_file = openpyxl.load_workbook(map_path_prefix + "_" + suffix + ".xlsx")["Sheet1"]
        for j in range(1, map_file.max_column):
            if str(map_file.cell(1, j + 1).value) not in map_node_var.keys(): 
                map_node_var[str(map_file.cell(1, j + 1).value)] = []
            for i in range(1, map_file.max_row): 
                if map_file.cell(i + 1, j + 1).value == 1:
                    map_node_var[str(map_file.cell(1, j + 1).value)].append(str(map_file.cell(i + 1, 1).value))
    full_node_lst, adj_lst = [], []
    adj_file = openpyxl.load_workbook(adj_path)["Sheet1"]
    for j in range(1, adj_file.max_column): 
        full_node_lst.append(str(adj_file.cell(1, j + 1).value))
    for node in map_node_var.keys(): 
        if node not in full_node_lst:
            continue
        i = full_node_lst.index(node)
        for j in range(i + 1, adj_file.max_column): 
            if adj_file.cell(i + 2, j + 1).value == 1: 
                adj_lst.append([i, j - 1])
    np.savetxt(os.path.join(root_path, 'TOPO.csv'), np.array(adj_lst, dtype = int), delimiter = ',')
    print(map_node_var)
    for k in range(1, raw_data_num + 1): 
        data_path = data_path_prefix + "_" + str(k) + ".csv"
        with open(data_path, 'r') as data_file: 
            reader = csv.reader(data_file)
            for line in reader: 
                break
        full_var_lst = [var.lower().split('|')[-1].split(' ')[-1] for var in line]
        idx_lst = []
        for node in full_node_lst: 
            for var in map_node_var[node]: 
                idx_lst.append(full_var_lst.index(var.lower()))
                print(var, full_var_lst.index(var.lower()))
        print(len(full_node_lst), len(idx_lst), idx_lst)
        with open(data_path, 'r') as data_file: 
            data = np.loadtxt(data_file, float, delimiter = ',', skiprows = 1, usecols = idx_lst)
            np.savetxt(os.path.join(root_path, 'BCM_LOAD_CLEAN_{}.csv'.format(k)), data, delimiter = ',')
        print(k)
    return


def fetch_data(file_path, topo_path):
    vs = []
    with open(file_path) as f: 
        data = np.loadtxt(f, float, delimiter = ',')
    vs = data
    with open(topo_path) as g:
        topo = np.loadtxt(g, float, delimiter = ',').astype(int)
    return vs, topo


def transform_bcm(segment_len=100, nf = 12, dr = 8, main_path = "datasets/RTDS_Real/Raw_Data/BCM_CONNECTED/Clean/", hdf5_path = "datasets/RTDS_Real/Processed_Data/BCM_CONNECTED/bcm_len100_p8.hdf5"):
    """
    :param main_path: the main path that saves the dataset.
    :return:
    """
    file_lists = [os.path.join(main_path, 'BCM_LOAD_CLEAN_{}.csv'.format(n)) for n in range(1, 6)]
    topo_path = os.path.join(main_path, 'TOPO.csv')
    with h5py.File(hdf5_path, 'w') as Dataset: 
        data_idx = 0
        topo_idx = 0
        node_values_sum = 0
        node_values_sum_square = 0
        node_num = 0
        for file_path in file_lists:
            vs, topo = fetch_data(file_path, topo_path)
            Dataset.create_dataset('topologies_node_{}'.format(topo_idx), data = topo, chunks = True)
            nt, nn = vs.shape[0], vs.shape[1] // nf
            start = 0
            Dataset.create_dataset('topologies_#node_{}'.format(topo_idx), data = nn)
            while start + segment_len <= nt: 
                end = start + segment_len
                node_values = vs[start : end, :].reshape(segment_len, nn, nf)
                node_values_sum += node_values.sum(0)
                node_values_sum_square += np.power(node_values, 2).sum(0)
                node_num += (end - start)
                Dataset.create_dataset('node_values_{}'.format(data_idx), data = node_values, chunks = True, compression = 'gzip')
                Dataset.create_dataset('topologies_{}'.format(data_idx), data = topo_idx)
                start += segment_len
                data_idx += 1
            topo_idx += 1
        Dataset.create_dataset('num_seg', data = data_idx)
        node_values_mean = node_values_sum / node_num
        node_values_std = np.sqrt(node_values_sum_square / node_num - node_values_mean ** 2)
        Dataset.create_dataset('values_mean', data = node_values_mean)
        Dataset.create_dataset('values_std', data = node_values_std)
    return


def transform_bcm_points(num_masks_per_seg=25, train_ratio=0.8, valid_ratio=0.9, p = 0.7, downsampling=1, 
                          hdf5_path="datasets/RTDS_Real/Processed_Data/BCM_CONNECTED/bcm_len100_p8_missing7_downsampling1.hdf5",
                          data_path="datasets/RTDS_Real/Processed_Data/BCM_CONNECTED/bcm_len100_p8.hdf5"):
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



class BCM_data(Dataset):
    def __init__(self, split, mask_path, downsampling=1, raw_data_path="/home/yucxing/exp/C004/EvolveSDE/datasets/RTDS_Real/Processed_Data/BCM_SS2/Load/bcm_len100.hdf5", delta_t=0.01, noise_ratio=0, 
                    return_Y=False, data_rate = 8):
        """
        :param split: train/valid/test
        :param mask_path:  the file path that saves the binary masks.
        :param downsampling:  the interval to downsampling the data trajectories (mask should be already downsampled)
        :param raw_data_path:  the file path that saves the raw data.
        :param delta_t: time step.
        :param return_Y:  whether return the DER controller signals (for supervised learning task).
        """
        super(BCM_data, self).__init__()
        assert split in ['train', 'valid', 'test'], "mode should be in ['train', 'valid', 'test']."
        self.raw_data = h5py.File(raw_data_path.split('.')[0] + '_p{}'.format(data_rate) + '.hdf5', 'r')
        self.mean = self.raw_data['values_mean'][()]
        self.std = self.raw_data['values_std'][()]
        mask_data = h5py.File(mask_path, 'r')
        self.node_masks = mask_data['{}/node_masks'.format(split)]
        self.data_idx = mask_data['{}/data_idx'.format(split)]
        self.size = self.node_masks.shape[0]
        self.downsampling = downsampling
        self.upsampling = 0.01 / delta_t
        self.split = split
        self.der = return_Y
        self.data_rate = data_rate
        self.delta_t = delta_t
        self.noise_ratio = noise_ratio
        return

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        train_data_index = self.data_idx[idx]
        #X = self.raw_data['node_values_{}'.format(train_data_index)][()]
        #print(X.min().item(), X.max().item())
        X = (self.raw_data['node_values_{}'.format(train_data_index)][()] - self.mean) / (self.std + 1e-16)
        if self.noise_ratio != 0: 
            rms_X = np.sqrt((X * X).sum((0)) / X.shape[0])
            X_noise = np.random.randn(X.shape[0], X.shape[1], X.shape[2]) * self.noise_ratio * rms_X
            X_ori = X + X_noise
        M_x = self.node_masks[idx].astype(np.float32)
        t = np.arange(0, X.shape[0] * self.upsampling) * self.delta_t
        # up sampling.
        if self.upsampling != 1: 
            up_sampling_idx = np.arange(0, X.shape[0] * self.upsampling, self.upsampling).astype(np.int32)
            t = t[up_sampling_idx].astype(np.float32)
        # down sampling.
        down_sampling_idx = np.arange(0, X.shape[0], self.downsampling)
        X, t = X[down_sampling_idx].astype(np.float32), t[down_sampling_idx].astype(np.float32)
        if self.noise_ratio != 0: 
            X_ori = X_ori[down_sampling_idx].astype(np.float32)
        # numbers of nodes
        num_nodes = self.raw_data['topologies_#node_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        # construct the adjacent matrices.
        edges = self.raw_data['topologies_node_{}'.format(self.raw_data['topologies_{}'.format(train_data_index)][()])][()]
        i = torch.LongTensor(edges)
        v = torch.FloatTensor(torch.ones(i.size(0)))
        adjacent_nodes = torch.sparse.FloatTensor(i.t(), v, torch.Size([num_nodes, num_nodes])).to_dense()
        adjacent_nodes += adjacent_nodes.t()
        #print(X.min().item(), X.max().item())
        if self.noise_ratio != 0:
            return {'t': t, 'adjacent_matrices': adjacent_nodes, 'values': X_ori, 'masks': M_x,
                        '#nodes': num_nodes, 'ori_values': X, 'mean': self.mean, 'std': self.std}
        else: 
            return {'t': t, 'adjacent_matrices': adjacent_nodes, 'values': X, 'masks': M_x,
                        '#nodes': num_nodes, 'ori_values': X, 'mean': self.mean, 'std': self.std}


#cleanData()
#transform_bcm()
#transform_bcm_points()
#"datasets/RTDS_Real/Processed_Data/BCM_CONNECTED/bcm_len100.hdf5"