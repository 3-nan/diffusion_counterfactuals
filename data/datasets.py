from typing import Any
from torchvision import datasets, transforms
from data.imagenet_classnames import name_map, folder_label_map
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from pathlib import Path
import random
import pickle
import linecache


class ImageNet(datasets.ImageFolder):
    classes = [name_map[i] for i in range(1000)]
    name_map = name_map

    def __init__(
            self, 
            root:str, 
            split:str="val", 
            transform=None, 
            target_transform=None, 
            class_idcs=None, 
            start_sample: float = 0., 
            end_sample: int = 50000//1000,
            return_tgt_cls: bool = False,
            idx_to_tgt_cls_path = None,
            restart_idx: int = 0, 
            **kwargs
    ):
        _ = kwargs  # Just for consistency with other datasets.
        print(f"Loading ImageNet with start_sample={start_sample}, end_sample={end_sample} ")
        assert split in ["train", "val"]
        assert start_sample < end_sample and start_sample >= 0 and end_sample <= 50000//1000
        self.start_sample = start_sample

        assert 0 <= restart_idx < 50000
        self.restart_idx = restart_idx

        path = root if root[-3:] == "val" or root[-5:] == "train" else os.path.join(root, split)
        super().__init__(path, transform=transform, target_transform=target_transform)
        
        with open(idx_to_tgt_cls_path, 'r') as file:
            idx_to_tgt_cls = yaml.safe_load(file)
            if isinstance(idx_to_tgt_cls, dict):
                idx_to_tgt_cls = [idx_to_tgt_cls[i] for i in range(len(idx_to_tgt_cls))]
        self.idx_to_tgt_cls = idx_to_tgt_cls

        self.return_tgt_cls = return_tgt_cls

        if class_idcs is not None:
            class_idcs = list(sorted(class_idcs))
            tgt_to_tgt_map = {c: i for i, c in enumerate(class_idcs)}
            self.classes = [self.classes[c] for c in class_idcs]
            samples = []
            idx_to_tgt_cls = []
            for i, (p, t) in enumerate(self.samples):
                if t in tgt_to_tgt_map:
                    samples.append((p, tgt_to_tgt_map[t]))
                    idx_to_tgt_cls.append(self.idx_to_tgt_cls[i])
            
            self.idx_to_tgt_cls = idx_to_tgt_cls
            #self.samples = [(p, tgt_to_tgt_map[t]) for i, (p, t) in enumerate(self.samples) if t in tgt_to_tgt_map]
            self.class_to_idx = {k: tgt_to_tgt_map[v] for k, v in self.class_to_idx.items() if v in tgt_to_tgt_map}

        if "val" == split: # reorder
            new_samples = []
            idx_to_tgt_cls = []
            for idx in range(50000//1000):
                new_samples.extend(self.samples[idx::50000//1000])
                idx_to_tgt_cls.extend(self.idx_to_tgt_cls[idx::50000//1000])
            self.samples = new_samples[int(start_sample*1000):end_sample*1000]
            self.idx_to_tgt_cls = idx_to_tgt_cls[int(start_sample*1000):end_sample*1000]

        else:
            raise NotImplementedError
        
        if self.restart_idx > 0:
            self.samples = self.samples[self.restart_idx:]
            self.idx_to_tgt_cls = self.idx_to_tgt_cls[self.restart_idx:]

        self.class_labels = {i: folder_label_map[folder] for i, folder in enumerate(self.classes)}
        self.targets = np.array(self.samples)[:, 1]
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.return_tgt_cls:
            return *sample, self.idx_to_tgt_cls[index], index + self.start_sample*1000 + self.restart_idx
        else:
            # return sample, index + self.start_sample*1000 + self.restart_idx
            return sample
