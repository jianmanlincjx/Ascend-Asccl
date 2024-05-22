import os
import torch
import random
import torchvision
import pickle
import cv2
import json
import time
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F


import sys
sys.path.append(os.getcwd())


class SpatialDataloader(Dataset):
    def __init__(self, mode="train") -> None:
        super(SpatialDataloader).__init__()
        self.size = 224
        self.root = "/path/to/MEAD"
        with open("./dataloader/aligned_path36.json", "r") as json_file:
            self.data_file = json.load(json_file)
        if mode == "train":
            self.vid_list = sorted(list(self.data_file.keys()))
            self.vid_list.remove("W021")
        else:
            self.vid_list = ["W021"]
        
        self._img_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((self.size, self.size), antialias=True),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.num_img = self.calculate_length()
        print(f"{mode} The dataset size is： {self.num_img}")
        
        
    def calculate_length(self):

        len_pair = 0
        for vid in self.vid_list:
            emotion = self.data_file[vid].keys()
            for em in emotion:
                vid_sub = sorted(self.data_file[vid][em])
                for vid_sub_sub in vid_sub:
                    len_pair += len(self.data_file[vid][em][vid_sub_sub][0])
        return len_pair
            
    
    def __len__(self):
        return self.num_img
    
    def __getitem__(self, index):
        vid = random.choice(self.vid_list)
        emotion_list = list(self.data_file[vid].keys())
        emn = random.choice(emotion_list)
        vid_sub = list(self.data_file[vid][emn].keys())
        sub = random.choice(vid_sub)
        vid_len = len(self.data_file[vid][emn][sub][0])
        assert len(self.data_file[vid][emn][sub][0]) == len(self.data_file[vid][emn][sub][1])
        idx = random.choice(range(0, vid_len))
        source_vid = sub.split("_")[0]
        target_vid = sub.split("_")[1]
        source_img_idx = self.data_file[vid][emn][sub][0][idx]
        target_img_idx = self.data_file[vid][emn][sub][1][idx]
        source_img = cv2.imread(os.path.join(self.root, vid, "align_img", "neutral", source_vid, str(source_img_idx).zfill(6)+".jpg"))        
        target_img = cv2.imread(os.path.join(self.root, vid, "align_img", emn, target_vid, str(target_img_idx).zfill(6)+".jpg"))
        source_img = self._img_transform(source_img)
        target_img = self._img_transform(target_img)
        return {"source_img": source_img,
                "target_img": target_img}
        
