# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base import BaseDataset
from .builder import DATASETS
import numpy as np

@DATASETS.register_module()
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        self.embeddings=np.load("./manet/1214_new_mean_Vectors.npy")
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        


    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                    if 0 <= label <= 10:
                        label=0
                        emb=self.embeddings[0]
                    elif 11 <= label <= 23:
                        label=1
                        emb=self.embeddings[11]
                    elif 24 <= label <= 31:
                        label=2
                        emb=self.embeddings[24]
                    elif 32 <= label <= 37:
                        label=3
                        emb=self.embeddings[32]
                    elif 38 <= label <= 47:
                        label=4
                        emb=self.embeddings[38]
                    elif 48 <= label <= 51:
                        label=5
                        emb=self.embeddings[48]
                    # else:
                    #     continue
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                # emb=self.embeddings[label]
                video_infos.append(dict(filename=filename, label=label,emb=emb))
        return video_infos
