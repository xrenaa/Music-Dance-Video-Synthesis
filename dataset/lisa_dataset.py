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
    def __init__(self, opt, train=True):
        file_location=opt.data
        pose_dict=read_from_json(file_location)
        
        length=0
        keys=sorted(pose_dict.keys())
        for key in keys:
            #index = str("%03d" % i)
            sub_keys=sorted(pose_dict[str(key)].keys())
            if key=="046":
                break
            for sub_key in sub_keys:
                temp_pose=np.array(pose_dict[str(key)][str(sub_key)]["joint_coors"])
                if(temp_pose.shape==(100,)):
                    print("girl"+key+" "+sub_key+" is wrong")
                    continue
                length+=1
        self.length=2*length
        print(self.length)
        
        target=torch.FloatTensor(2*length,50,1600).zero_()
        label=torch.FloatTensor(2*length,50,18,2).zero_()
        index=0
        
        keys=sorted(pose_dict.keys())
        #keys=["017","018"]
        for key in keys:
            #index = str("%03d" % i)
            sub_keys=sorted(pose_dict[str(key)].keys())
            if key=="046":
                break
            for sub_key in sub_keys:

                print(key+" "+sub_key)
                temp_audio=np.array(pose_dict[str(key)][str(sub_key)]['audio_sequence'])
                
                temp_pose=np.array(pose_dict[str(key)][str(sub_key)]["joint_coors"])
                if(temp_pose.shape==(100,)):
                    continue
                x_coor=(temp_pose[:,:,0]/320)-1
                y_coor=(temp_pose[:,:,1]/180)-1
                temp=np.zeros((100,18,2))
                temp[:,:,0]=x_coor
                temp[:,:,1]=y_coor
                temp_pose=temp

                d = torch.from_numpy(temp_audio).type(torch.LongTensor)
               
                slices1=d[0:80000].view(50,1600)
                slices2=d[80000:160000].view(50,1600)
                target[index]=slices1
                target[index+1]=slices2
                
                label[index]=torch.from_numpy(temp_pose[0:50,:,:])
                label[index+1]=torch.from_numpy(temp_pose[50:100,:,:])
                index+=2
        
        self.audio=target
        self.label=label
        #self.audio_label=audio_label
        
        #10s
        self._length = 80000
        
        self.train = train
        print("load the json file to dictionary (10s raw data)" )
        # assign every *test_stride*th item to the test set


    def __getitem__(self, idx):
        #print("idx:",idx)
        one_hot=self.audio[idx]
        target=self.label[idx]          
        return one_hot, target

    def __len__(self):
        return self.length


def quantize_data(data, classes):
    mu_x = mu_law_encoding(data, classes)
    bins = np.linspace(-1, 1, classes)
    quantized = np.digitize(mu_x, bins) - 1
    return quantized


def mu_law_encoding(data, mu):
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj) 
    
def save_to_json(dic,target_dir):
    dumped = json.dumps(dic, cls=NumpyEncoder)  
    file = open(target_dir, 'w')  
    json.dump(dumped, file)
    file.close()
    
def read_from_json(target_dir):
    f = open(target_dir,'r')
    data = json.load(f)
    data = json.loads(data)
    f.close()
    return data 