import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import model.utils as utils
import torchvision
import os

class HCN(nn.Module):
    '''
    Input shape:
    Input shape should be (N, C, T, V, M)
    where N is the number of samples,
          C is the number of input channels,
          T is the length of the sequence,
          V is the number of joints
      and M is the number of people.
    '''
    def __init__(self,
                 in_channel=2,
                 num_joint=18,
                 num_person=1,
                 out_channel=64,
                 window_size=32,
                 num_class = 256,
                 ):
        super(HCN, self).__init__()
        self.num_person = num_person
        self.num_class = num_class
        # position
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=32, kernel_size=(3,1), stride=1, padding=0)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))
        # motion
        self.conv1m = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        self.conv2m = nn.Conv2d(in_channels=out_channel, out_channels=window_size, kernel_size=(3,1), stride=1, padding=0)

        self.conv3m = nn.Sequential(
            nn.Conv2d(in_channels=num_joint, out_channels=out_channel//2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2))
        self.conv4m = nn.Sequential(
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2))

        # concatenate motion & position
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel*4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2)
        )

        self.fc7= nn.Sequential(
            nn.Linear(1536,256*2), # 4*4 for window=64; 8*8 for window=128
            nn.ReLU(),
            nn.Dropout2d(p=0.5))
        self.fc8 = nn.Linear(256*2,num_class)

        # initial weight
        utils.initial_model_weight(layers = list(self.children()))
        print('weight initial finished!')


    def forward(self, x,target=None):
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        x=x.view(N,C,T,V)
        motion=motion.view(N,C,T,V)
            # position
            # N0,C1,T2,V3 point-level
        out = self.conv1(x)

        out = self.conv2(out)
            # N0,V1,T2,C3, global level
        out = out.permute(0,3,2,1).contiguous()
        out = self.conv3(out)
        out_p = self.conv4(out)


            # motion
            # N0,T1,V2,C3 point-level
        out = self.conv1m(motion)
        out = self.conv2m(out)
            # N0,V1,T2,C3, global level
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv3m(out)
        out_m = self.conv4m(out)

            # concat
        out = torch.cat((out_p,out_m),dim=1)
        out = self.conv5(out)
        out = self.conv6(out)

        # max out logits
        out = out.view(out.size(0), -1)
        #print("out",out.shape)
        out = self.fc7(out)
        out = self.fc8(out)

        t = out
        assert not ((t != t).any())# find out nan in tensor
        assert not (t.abs().sum() == 0) # find out 0 tensor

        return out
    
    def extract_feature(self, x,target=None):
        outs = []
        N, C, T, V, M = x.size()  # N0, C1, T2, V3, M4
        motion = x[:,:,1::,:,:]-x[:,:,0:-1,:,:]
        motion = motion.permute(0,1,4,2,3).contiguous().view(N,C*M,T-1,V)
        motion = F.upsample(motion, size=(T,V), mode='bilinear',align_corners=False).contiguous().view(N,C,M,T,V).permute(0,1,3,4,2)

        x=x.view(N,C,T,V)
        motion=motion.view(N,C,T,V)
            # position
            # N0,C1,T2,V3 point-level
        out = self.conv1(x)
        outs.append(out)
        out = self.conv2(out)
        outs.append(out)
            # N0,V1,T2,C3, global level
        out = out.permute(0,3,2,1).contiguous()
        out = self.conv3(out)
        outs.append(out)
        out_p = self.conv4(out)
        outs.append(out)

            # motion
            # N0,T1,V2,C3 point-level
        out = self.conv1m(motion)
        out = self.conv2m(out)
            # N0,V1,T2,C3, global level
        out = out.permute(0, 3, 2, 1).contiguous()
        out = self.conv3m(out)
        out_m = self.conv4m(out)

            # concat
        out = torch.cat((out_p,out_m),dim=1)
        out = self.conv5(out)
        outs.append(out)
        out = self.conv6(out)
        outs.append(out)

        return outs

