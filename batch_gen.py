#!/usr/bin/python2.7

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):    # 对数据进行shuffle
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    # 所有的数据列表
    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    # batch_size为1，这个难道是直接对整个视频操作？
    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        # 对一个batch中的每个视频处理
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')  # 视频对应的特征 
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))  # 这个gt里应该是每一帧的类别
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]  # 更新每一帧所属的类别
            batch_input .append(features[:, ::self.sample_rate])   # (N,C,T)，因为论文是对时序卷积，因此空间维度倍移除了
            batch_target.append(classes[::self.sample_rate])   

        length_of_sequences = map(len, batch_target)   # 获取每一个视频的序列长度
        
        # 总视频数，通道数，帧数
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        # (视频数，帧数)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        # (视频数，类别，帧数)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])  # 全部置为1

        return batch_input_tensor, batch_target_tensor, mask
