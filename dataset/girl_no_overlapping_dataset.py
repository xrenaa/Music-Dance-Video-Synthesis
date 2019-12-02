import os
import os.path
import math
import threading
import torch
import torch.utils.data
import numpy as np
import bisect
import json

class DanceDataset(torch.utils.data.Dataset):
    def __init__(self, train="train"):
        if train=="train":
            self.dict_path="/mnt/external4/xuanchi/korean_small_dataset/kpop_sequence/girl_revised_pose_pairs.json"
#         elif train=="diff":
#             self.dict_path="/mnt/external4/xuanchi/small_data/coco_val.json"
#         elif train =="one":
#             self.dict_path="/mnt/external4/xuanchi/small_data/one_val.json"
        dict=read_from_json(self.dict_path)
        self.x=np.array(dict["x"])
        self.label=np.array(dict["label"])
    
    def __getitem__(self, idx):
        input=self.x[idx]
        target=self.label[idx]                 
        return input, target

    def __len__(self):
        N,T,V=self.x.shape
        return N
       
def read_from_json(target_dir):
    f = open(target_dir,'r')
    data = json.load(f)
    data = json.loads(data)
    f.close()
    return data 