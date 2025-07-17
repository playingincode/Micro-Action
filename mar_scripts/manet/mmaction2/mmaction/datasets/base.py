# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from torch.utils.data import Dataset
import os
import csv
from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)
from .pipelines import Compose
import pickle
# import tools
import sys
sys.path.insert(0, '/data/stars/user/npoddar/UNIK_with_skeleton/')
from feeders import tools


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.该成员变量就是所有模块化数据预处理的集合。
                                                                        该成员变量用于后续数据读取过程。
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length
        # self.sample_name, self.label = pickle.load(f)
        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))
        
        self.numpy_data_for_unik_train=np.load("/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/UNIK_delete/data/newFilesma/xsub/train_data_joint.npy")
        self.numpy_data_for_unik_val=np.load("/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/UNIK_delete/data/newFilesma/xsub/val_data_joint.npy")
        with open("/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/UNIK_delete/data/newFilesma/xsub/train_label.pkl", 'rb') as f:
            self.sample_name_train, self.label_name_train = pickle.load(f, encoding='latin1')
            
        
        with open("/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/UNIK_delete/data/newFilesma/xsub/val_label.pkl", 'rb') as f:
            self.sample_name_val, self.label_name_val = pickle.load(f, encoding='latin1')
        
        self.window_size_unik=150
        # self.normalization=False
        self.random_shift_unik=False
        self.random_choose_unik=True
        self.random_move_unik=True
        
        
        self.reverse_mapping = {}  # video.mp4 → skeleton_file
        
        with open("/data/stars/user/npoddar/Micro-Action/mar_scripts/manet/mmaction2/mmaction/datasets/file_mapping_train_val.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                json_file = row["original_json_file"]
                skeleton_file = row["ntu_skeleton_file"]

                # Process json_file to get video filename
                if json_file.startswith("train_"):
                    trimmed = json_file[len("train_"):]
                elif json_file.startswith("val_"):
                    trimmed = json_file[len("val_"):]
                else:
                    trimmed = json_file
              
                video_file = trimmed.split("_class")[0] + ".mp4"
                # print(video_file)
                self.reverse_mapping[video_file] = skeleton_file

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def get_skeleton_from_rgb(self, rgb_filename,parts):
        """
        Given an RGB filename (e.g., 'video123.mp4'), return the corresponding skeleton numpy array
        from self.data using reverse lookup via file_mapping_train_val.csv
        """
        # Step 1: Build reverse map (once)
        # print(rgb_filename)
        

        # Step 2: Lookup skeleton file for given RGB
        if rgb_filename not in self.reverse_mapping:
            raise ValueError(f"RGB file {rgb_filename} not found in mapping.")

        skeleton_file = self.reverse_mapping[rgb_filename]

        # Step 3: Find index of skeleton_file in sample_name
        if "train" in parts:
            try:
                idx = self.sample_name_train.index(skeleton_file)
            except ValueError:
                raise ValueError(f"Skeleton file {skeleton_file} not found in sample_name.")

            # Step 4: Return skeleton data
            
            skeleton_numpy = self.numpy_data_for_unik_train[idx]
            return skeleton_numpy, idx
        
        
        else:
            # if "train" in parts:
            try:
                idx = self.sample_name_val.index(skeleton_file)
            except ValueError:
                raise ValueError(f"Skeleton file {skeleton_file} not found in sample_name.")

            # Step 4: Return skeleton data
            
            skeleton_numpy = self.numpy_data_for_unik_val[idx]
            return skeleton_numpy, idx
    
    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results,
                                                      gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def pad_or_truncate(self,features, target_len=10):
        """
        Pad or truncate a tensor to have shape [target_len, 1, 1408].

        Args:
            features (Tensor): Input tensor of shape [T, 1, 1408]
            target_len (int): Desired length along the first dimension

        Returns:
            Tensor of shape [target_len, 1, 1408]
        """
        T, _, D = features.shape
        if T == target_len:
            return features
        elif T < target_len:
            # Pad with zeros
            pad_len = target_len - T
            pad_tensor = torch.zeros(pad_len, 1, D, dtype=features.dtype, device=features.device)
            return torch.cat([features, pad_tensor], dim=0)
        else:
            # Truncate
            return features[:target_len]
    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # print("results",results)
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        video_path = results['filename']
        parts = video_path.strip(os.sep).split(os.sep)
        # print("Video name",parts[-1])
        relative_path = os.path.join(parts[-2], parts[-1])
        # print("parts",parts[-2])
        # print("Video path",)
        complete_path_videomaev2_features=os.path.join("/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/features_ma52_RGB/",relative_path)
        # print("Video mae v2",complete_path_videomaev2_features)
        npy_path = os.path.splitext(complete_path_videomaev2_features)[0] + '.npy'
        
        # unik_path=""
        
        # print(numpy_data)
        # print("Video path",video_path)
        skeleton_data,_=self.get_skeleton_from_rgb(parts[-1],parts[-2])
        skeleton_data = np.concatenate((skeleton_data[:,:,:,:], np.zeros((skeleton_data.shape))), axis=3)
        skeleton_data=self.unik_data_maker(skeleton_data)
        # print(skeleton_data)
        numpy_data=skeleton_data
        # print("Type:", type(numpy_data))
        # if isinstance(numpy_data, np.ndarray):
        #     print("Shape:", numpy_data.shape)
        #     print("Dtype:", numpy_data.dtype)
        # print(skeleton_data.dtype)

# Load the .npy file
        if os.path.exists(npy_path):
            features = np.load(npy_path)
        else:
            print("Hi")
            # print(features)

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
        
        features_tensor = torch.tensor(features) 
        features_tensor=self.pad_or_truncate(features_tensor)
        # print("Pipeline ",type(self.pipeline(results)['imgs']))
        # exit()
        # print(f"[DEBUG] idx={idx}, features shape: {features_tensor.shape}")
        data = self.pipeline(results)
        data['imgs'] = features_tensor
        data['skeleton_data']=numpy_data
        return data
    
    
    def unik_data_maker(self,data_numpy):
        
       
        # if self.normalization:
        #     data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift_unik:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose_unik:
            data_numpy = tools.random_choose(data_numpy, self.window_size_unik)
        elif self.window_size_unik > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size_unik)
        if self.random_move_unik:
            data_numpy = tools.random_move(data_numpy)
        
        return data_numpy
       

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        # print("results",results)
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        video_path = results['filename']
        parts = video_path.strip(os.sep).split(os.sep)
        # print("Video name",parts[-1])
        relative_path = os.path.join(parts[-2], parts[-1])
        # print("parts",parts[-2])
        # print("Video path",)
        complete_path_videomaev2_features=os.path.join("/data/stars/user/areka/MULTIMEDIA_CONFERANCE_2025/features_ma52_RGB/",relative_path)
        # print("Video mae v2",complete_path_videomaev2_features)
        npy_path = os.path.splitext(complete_path_videomaev2_features)[0] + '.npy'
        
        # unik_path=""
        
        # print(numpy_data)
        # print("Video path",video_path)
        skeleton_data,_=self.get_skeleton_from_rgb(parts[-1],parts[-2])
        skeleton_data = np.concatenate((skeleton_data[:,:,:,:], np.zeros((skeleton_data.shape))), axis=3)
        skeleton_data=self.unik_data_maker(skeleton_data)
        # print(skeleton_data)
        numpy_data=skeleton_data
        # print("Type:", type(numpy_data))
        # if isinstance(numpy_data, np.ndarray):
        #     print("Shape:", numpy_data.shape)
        #     print("Dtype:", numpy_data.dtype)
        # print(skeleton_data.dtype)

# Load the .npy file
        if os.path.exists(npy_path):
            features = np.load(npy_path)
        else:
            print("Hi")
            # print(features)

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot
        
        features_tensor = torch.tensor(features) 
        features_tensor=self.pad_or_truncate(features_tensor)
        # print("Pipeline ",type(self.pipeline(results)['imgs']))
        # exit()
        # print(f"[DEBUG] idx={idx}, features shape: {features_tensor.shape}")
        data = self.pipeline(results)
        data['imgs'] = features_tensor
        data['skeleton_data']=numpy_data
        return data
    
        # return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        # print("Ask a good question?")
        if self.test_mode:
            # print("Entering this")
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)
