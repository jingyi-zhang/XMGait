import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr
import numpy as np


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        
        # self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_data[0])

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        index = 0
        for idx, data in enumerate(data_list):
            if len(data) > len(data_list[-1]):
                data_list[index] = data[:len(data_list[-1])]
                index = index+1
                # raise ValueError(
                #     'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        data_list = []
        # if not self.cache:
        #     data_list = self.__loader__(self.seqs_info[idx][-1])
        # elif self.seqs_data[idx] is None:
        #     data_list = self.__loader__(self.seqs_info[idx][-1])
        #     self.seqs_data[idx] = data_list
        # else:
        #     data_list = self.seqs_data[idx]
        for data_in in self.seqs_data:
            data_list.append(data_in[idx])
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None
        try:
            seq_len = data_config['seq_len']  # [n], true or false
        except:
            seq_len = None
        assert seq_len is not None
            
        with open(data_config['dataset_partition'], "rb") as f: #split dataset json
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            height_info = np.load(
                'datasets/height_and_gender.npy',
                allow_pickle=True).item()
            seqs_info_list = []
            use_num = sum(data_in_use)
            seqs_data = [[] for i in range(use_num)]
            for lab in label_set:
                keys_ = '/hdd24T/zjy/jingyi/gait/xmu_gait/lidarcap/human/' + str(lab)
                height = height_info[keys_]['height']
                gender_ = height_info[keys_]['gender']
                if gender_ == '男':
                    gender=1
                else:
                    gender=0
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie,height, gender]
                        seq_path = osp.join(dataset_root, *seq_info[:3])
                        seq_dirs = sorted(os.listdir(seq_path))
                        # if seq_dirs != []:
                        if f'50-{vie}-LiDAR-warp.pkl' in seq_dirs:
                            seq_dirs = [osp.join(seq_path, f'00-{vie}-LiDAR-PCDs.pkl'),
                                        osp.join(seq_path, f'00-{vie}-LiDAR-PCDs_depths.pkl'),
                                        osp.join(seq_path, f'50-{vie}-LiDAR-warp.pkl')]
                            # if data_in_use is not None:
                            #     seq_dirs = [dir for dir, use_bl in zip(
                            #         seq_dirs, data_in_use) if use_bl]
                            data_list_pkl = self.__loader__(seq_dirs)
                            for index in range(len(data_list_pkl[0])//seq_len):
                                l = index*seq_len
                                r = l + seq_len
                                for i in range(use_num):
                                    seqs_data[i].append(data_list_pkl[i][l:r])
                                
                                seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list, seqs_data

        self.seqs_info, self.seqs_data = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
