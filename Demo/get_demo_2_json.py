import sys
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
#from model.frameD import frame_discriminator#(50,3,360,640)
from model.pose_generator_norm import Generator#input 50,1,1600
#from model.sequenceD import seq_discriminator#image input 50*3*360*640

#from dataset.small_dataset import DanceDataset #audio input 50*1*1600
#from dataset.boy_test_dataset import DanceDataset
from dataset.lisa_dataset_test import DanceDataset
# from dataset.dance_dataset import DanceDataset #audio input 50*1*1600
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.utils import save_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "8"
import numpy as np
import math
import itertools
import time
import datetime
from matplotlib import pyplot as plt
#import cv2
from dataset.output_helper import save_2_batch_images
import argparse
from scipy.io.wavfile import write
import json
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
    
parser = argparse.ArgumentParser()
parser.add_argument(
        "--model",
        default="",
        metavar="FILE",
        help="path to pth file",
        type=str,
    )
parser.add_argument("--count", type=int, default=50)
parser.add_argument(
        "--output",
        default="/home/xuanchi/August/train_for_boy/pose2vid/result/fake",
        metavar="FILE",
        help="path to output",
        type=str,
    )
args = parser.parse_args()

file_path=args.model
counter=args.count
output_dir=args.output

Tensor = torch.cuda.FloatTensor
generator = Generator(1)
generator.load_state_dict(torch.load(file_path))
generator.cuda()
data=DanceDataset()
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=8,
                                         pin_memory=False)
criterion_pixelwise = torch.nn.L1Loss()
count = 0
total_loss=0.0
img_orig = np.ones((360,640,3), np.uint8) * 255
dict = {}
dict["real"]=[]
dict["fake"]=[]

for i, (x,target) in enumerate(dataloader):
            audio_out=x.view(-1) #80000
            scaled=np.int16(audio_out)
            audio = Variable(x.type(Tensor).transpose(1,0))#50,1,1600
            pose = Variable(target.type(Tensor))#1,50,18,2
            pose=pose.view(1,50,36)
            # Adversarial ground truths
#             frame_valid = Variable(Tensor(np.ones((1,50))),requires_grad=False)                
#             frame_fake_gt = Variable(Tensor(np.zeros((1,50))),requires_grad=False)
#             seq_valid = Variable(Tensor(np.ones((1,1))),requires_grad=False)                
#             seq_fake_gt = Variable(Tensor(np.zeros((1,1))),requires_grad=False)
            
        # ------------------
        #  Train Generators
        # ------------------
            generator.eval()
            #optimizer_G.zero_grad()

            # GAN loss
            fake = generator(audio)
            loss_pixel = criterion_pixelwise(fake, pose)
            total_loss+=loss_pixel.item()
            
            fake = fake.contiguous().cpu().detach().numpy()#1,50,36 
            fake = fake.reshape([50,36])
            
            
            if(count <= counter):
                write(output_dir+"/audio/{}.wav".format(i),16000,scaled)
                real_coors = pose.cpu().numpy()
                #print(real_coors.shape)
                fake_coors = fake
                real_coors = real_coors.reshape([-1,18,2])
                fake_coors = fake_coors.reshape([-1,18,2])
                
                real_coors[:,:,0] = (real_coors[:,:,0]+1) * 320
                real_coors[:,:,1] = (real_coors[:,:,1]+1 ) * 180
                real_coors = real_coors.astype(int)
                dict["real"].append(real_coors)
                
                fake_coors[:,:,0] = (fake_coors[:,:,0]+1) * 320
                fake_coors[:,:,1] = (fake_coors[:,:,1]+1 ) * 180
                fake_coors = fake_coors.astype(int)
                dict["fake"].append(fake_coors)
            count += 1
save_to_json(dict,"/home/xuanchi/August/train_for_boy/pose2vid/pose.json")

final_loss=total_loss/count
print("final_loss:",final_loss)