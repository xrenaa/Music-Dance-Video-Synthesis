import sys
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
#from model.frameD import frame_discriminator#(50,3,360,640)
from model.demo_generator import Generator#input 50,1,1600
#from model.sequenceD import seq_discriminator#image input 50*3*360*640
#from dataset.lisa_dataset_test import DanceDataset #audio input 50*1*1600
#from dataset.test_dataset import DanceDataset #audio input 50*1*1600
#from dataset.small_dataset import DanceDataset #audio input 50*1*1600
from dataset.new_lisa import DanceDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
import numpy as np
import math
import itertools
import time
import datetime
import sys
from matplotlib import pyplot as plt
#import cv2
from dataset.output_helper import save_batch_images_long
import argparse
from scipy.io.wavfile import write

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
        default="'/mnt/external4/output_demo'",
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
generator.eval()
generator.load_state_dict(torch.load(file_path))
generator.cuda()
data=DanceDataset()
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=6,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=False,
                                         drop_last=True)
# dataloader = torch.utils.data.DataLoader(data,
#                                          batch_size=1,
#                                          shuffle=False,
#                                          num_workers=8,
#                                          pin_memory=False)
criterion_pixelwise = torch.nn.L1Loss()
count = 0
total_loss=0.0
img_orig = np.ones((360,640,3), np.uint8) * 255
# for i, (x,target) in enumerate(dataloader):
for i, (target,x) in enumerate(dataloader):
            audio_out=x.view(-1) #80000
            scaled=np.int16(audio_out)
            x = x.contiguous().view(1,300,1600)
            audio = Variable(x.type(Tensor).transpose(1,0))#50,1,1600
            pose = Variable(target.type(Tensor))#1,50,18,2
            pose=pose.view(1,300,36)
        # ------------------
        #  Train Generators
        # ------------------
            #generator.eval()
            #optimizer_G.zero_grad()

            # GAN loss
            fake = generator(audio)
            loss_pixel = criterion_pixelwise(fake, pose)
            total_loss+=loss_pixel.item()
            
            fake = fake.contiguous().cpu().detach().numpy()#1,50,36 
            fake = fake.reshape([300,36])
            
            if(count <= counter):
                write(output_dir+"/audio/{}.wav".format(i),16000,scaled)
                real_coors = pose.cpu().numpy()
                #print(real_coors.shape)
#                 fake_coors = fake
                real_coors = real_coors.reshape([-1,18,2])
#                 fake_coors = fake_coors.reshape([-1,18,2])
                real_coors[:,:,0] = (real_coors[:,:,0]+1) * 320
                real_coors[:,:,1] = (real_coors[:,:,1]+1 ) * 180 + 30
                real_coors = real_coors.astype(int)
                
#                 fake_coors[:,:,0] = (fake_coors[:,:,0]+1) * 320
#                 fake_coors[:,:,1] = (fake_coors[:,:,1]+1 ) * 180
#                 fake_coors = fake_coors.astype(int)
                
                save_batch_images_long(real_coors,batch_num=count,save_dir_start=output_dir)
            count += 1

final_loss=total_loss/count
print("final_loss:",final_loss)