import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import torch
import torch.nn as nn
from torch import autograd
import numpy as np
from model.HCN_D import seq_discriminator
from model.local_HCN_frame_D import HCN
from model.pose_generator_norm import Generator#input 50,1,1600
from dataset.girl_no_overlapping_dataset import DanceDataset #audio input 50*1*1600
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import numpy as np
import math
import itertools
import time
import datetime
import sys
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import cv2

Tensor = torch.cuda.FloatTensor
batch_size = 100
log_dir = "local_GCN_perceptual_D_Feature_girl"
weight=200
gap=1

writer = SummaryWriter(log_dir='/home/xuanchi/self_attention_model/log/{}'.format(log_dir))

generator = Generator(batch_size)
frame_discriminator = HCN()
seq_discriminator=seq_discriminator(batch_size) #output

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0003)
optimizer_D1 = torch.optim.Adam(frame_discriminator.parameters(), lr=0.0003)
optimizer_D2 = torch.optim.Adam(seq_discriminator.parameters(), lr=0.0005)

generator.cuda()
frame_discriminator.cuda()
seq_discriminator.cuda()

from net.st_gcn_perceptual import Model
class GCNLoss(nn.Module):
    def __init__(self,dict_path="/home/xuanchi/August/gcn_dance/log/dropout/generator_799.pth"):
        super(GCNLoss, self).__init__()
        graph_args={"layout": 'openpose',"strategy": 'spatial'}
        self.gcn = Model(2,16,graph_args,edge_importance_weighting=True).cuda()
        self.gcn.load_state_dict(torch.load(dict_path))
        self.gcn.eval()
        self.criterion = nn.L1Loss()
        self.weights = [20.0 ,5.0 ,1.0 ,1.0 ,1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  #10 output      

    def forward(self, x, y):              
        x_gcn, y_gcn = self.gcn.extract_feature(x), self.gcn.extract_feature(y)
        loss = 0
        for i in range(len(x_gcn)):
            loss_state = self.weights[i] * self.criterion(x_gcn[i], y_gcn[i].detach())  
            print("VGG_loss "+ str(i),loss_state.item())
            loss += loss_state       
        return loss

class HCNLoss(nn.Module):
    def __init__(self):
        super(HCNLoss, self).__init__()
#         graph_args={"layout": 'openpose',"strategy": 'spatial'}
#         self.gcn = Model(2,16,graph_args,edge_importance_weighting=True).cuda()
#         self.gcn.load_state_dict(torch.load(dict_path))
#         self.gcn.eval()
        self.criterion = nn.L1Loss()
        #self.weights = [16.0, 16.0 ,16.0 ,8.0, 8.0 ,4.0, 2.0]  #7 output      
        #self.weights = [64.0 ,32.0 ,16.0 ,8.0, 8.0 ,4.0, 4.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        #self.weights = [8.0, 4.0, 4.0, 4.0, 4.0, 2.0, 1.0]
    def forward(self,D, x, y):   
        D.eval()
        x_gcn, y_gcn = D.extract_feature(x), D.extract_feature(y)
        loss = 0
        for i in range(len(x_gcn)):
            loss_state = self.weights[i] * self.criterion(x_gcn[i], y_gcn[i].detach())  
            print("VGG_loss "+ str(i),loss_state.item())
            loss += loss_state       
        return loss

    
data=DanceDataset()
dataloader = torch.utils.data.DataLoader(data,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=16,
                                         pin_memory=False,
                                         drop_last=True
                                        )
print("data ok")
def save_models(epoch):
    epoch = "%04d" % (epoch+1)
    torch.save(generator.state_dict(), "/home/xuanchi/self_attention_model/log/{}/generator_{}.pth".format(log_dir,epoch))
    torch.save(frame_discriminator.state_dict(), "/home/xuanchi/self_attention_model/log/{}/frame_{}.pth".format(log_dir,epoch))
    torch.save(seq_discriminator.state_dict(), "/home/xuanchi/self_attention_model/log/{}/sequence_{}.pth".format(log_dir,epoch))
    print("Chekcpoint saved") 
    
def compute_gradient_penalty_sequence(D, real_samples, fake_samples,audio):
    """Calculates the gradient penalty loss for WGAN GP"""
    #16,50,36
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    audio_input=audio.detach()
    audio_input.requires_grad_(True)
    d_interpolates = D(interpolates,audio_input)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=(interpolates,audio_input),
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_gradient_penalty_frame(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    #16,50,36
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 16).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs= interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
    
def train(epoch):
    adversarial_loss = torch.nn.BCELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    VGGLoss = GCNLoss()
    D_Feature = HCNLoss()
    index=0
    for epoch in range(epoch):
        batches_done=0
        total_loss1 = 0.0
        total_loss2 = 0.0
        total_loss3 = 0.0
        total_loss4 = 0.0
        for i, (x,target) in enumerate(dataloader):
            audio = Variable(x.type(Tensor).transpose(1,0))#50,1,1600
            pose = Variable(target.type(Tensor))#1,50,18,2
            #print(pose.shape)
            pose=pose.view(batch_size,50,36)
            # Adversarial ground truths
            frame_valid = Variable(Tensor(np.ones((batch_size,16))),requires_grad=False)                
            frame_fake_gt = Variable(Tensor(np.zeros((batch_size,16))),requires_grad=False)
            seq_valid = Variable(Tensor(np.ones((batch_size,1))),requires_grad=False)                
            seq_fake_gt = Variable(Tensor(np.zeros((batch_size,1))),requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
            generator.train()
            optimizer_G.zero_grad()
            
            # GAN loss
            fake = generator(audio).contiguous()#1,50,36
            frame_fake = frame_discriminator(fake)#1,50
            seq_fake=seq_discriminator(fake,audio)#1
            loss_frame = adversarial_loss(frame_fake, frame_valid)
            loss_seq= adversarial_loss(seq_fake,seq_valid)
            loss_pixel = criterion_pixelwise(fake, pose)
            loss_GCN = VGGLoss(fake,pose)
            loss_Frame_D = D_Feature(seq_discriminator, fake, pose)
            #print("loss_pixel:", loss_pixel.item())
            # Total loss
            loss_G = loss_frame + loss_seq + weight*loss_pixel + loss_GCN + loss_Frame_D
            loss_G.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator frame
        # ---------------------
            frame_discriminator.train()
            seq_discriminator.train()
            if batches_done%gap==0:
                optimizer_D1.zero_grad()
            # Real loss
                pred_real_frame = frame_discriminator(pose)# input bsz,50,36
                loss_real_frame = adversarial_loss(pred_real_frame, frame_valid)

            # Fake loss
                pred_fake_frame = frame_discriminator(fake.detach())
                loss_fake_frame = adversarial_loss(pred_fake_frame, frame_fake_gt)
                
                #GP_frame=compute_gradient_penalty_frame(frame_discriminator,pose,fake.detach())

            # Total loss
                D_loss_frame = 0.5 * (loss_real_frame + loss_fake_frame)
                loss_D1 = D_loss_frame
                loss_D1.backward()
                optimizer_D1.step()
        # ---------------------
        #  Train Discriminator seq
        # ---------------------
                optimizer_D2.zero_grad()
            # Real loss
                pred_real_seq = seq_discriminator(pose,audio)
                loss_real_seq = adversarial_loss(pred_real_seq, seq_valid)

            # Fake loss
                pred_fake_seq = seq_discriminator(fake.detach(),audio)
                loss_fake_seq = adversarial_loss(pred_fake_seq, seq_fake_gt)
                
                GP_seq=compute_gradient_penalty_sequence(seq_discriminator,pose,fake.detach(),audio)

            # Total loss
                D_loss_seq = 0.5 * (loss_real_seq + loss_fake_seq)
                loss_D2 = D_loss_seq + GP_seq
                loss_D2.backward()
                optimizer_D2.step()
        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left

            batches_done+=1
            index+=1
            batches_now = epoch * len(dataloader) + i
            total_loss1 += loss_G.item()
            total_loss2 += loss_pixel.item()
            total_loss3 += loss_D1.item()
            total_loss4 += loss_D2.item()
            writer.add_scalar('iteration/gan_loss', loss_G.item(), batches_now)
            writer.add_scalar('iteration/frame_loss', loss_D1.item(), batches_now)
            writer.add_scalar('iteration/real', loss_real_frame.item(), batches_now)
            writer.add_scalar('iteration/fake', loss_fake_seq.item(), batches_now)
            writer.add_scalar('iteration/seq_loss', loss_D2.item(), batches_now)
            writer.add_scalar('iteration/L1loss', loss_pixel.item(), batches_now)
            writer.add_scalar('iteration/VGGLoss', loss_GCN.item(), batches_now)
            writer.add_scalar('iteration/D_Feature_Loss', loss_Frame_D.item(), batches_now)
            print("Epoch {} {}, GLoss: {}, L1Loss: {}, D_Feature_Loss {}, VGG_Loss {}, D1Loss: {}, D2Loss: {}  ".format(epoch , batches_done , loss_G.item(),loss_pixel.item(),loss_Frame_D.item(),loss_GCN.item(),loss_D1.item(),loss_D2.item()))
                
#        if (epoch+1)%20==0:
#            #save_models(epoch)
        total_loss1 /= batches_done
        total_loss2 /= batches_done
        total_loss3 /= batches_done
        total_loss4 /= batches_done
        writer.add_scalar('epoch/gan_loss', total_loss1, epoch)
        writer.add_scalar('epoch/L1_loss', total_loss2, epoch)
        writer.add_scalar('epoch/frame_loss', total_loss3, epoch)
        writer.add_scalar('epoch/seq_loss', total_loss4, epoch)
        
    
    writer.close()   
    
train(401)