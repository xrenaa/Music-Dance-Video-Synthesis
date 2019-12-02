from model.audio_encoder import RNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HCN_encoder import HCN

class seq_discriminator(nn.Module):
    def __init__(self,batch):
        super(seq_discriminator,self).__init__()
        self.audio_encoder=RNN(batch)
        #self.image_encoder=image_encoder()#1,50,256
        self.pose_encoder=HCN()#input (batch,2,50,18,1)
        self.pose_rnn = nn.GRU(bidirectional=True,hidden_size=256, input_size=256,num_layers= 2, batch_first=True)
        self.pose_fc = nn.Linear(512,256)
        self.conv1d = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=2)
        #self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(255,1)
        #self.fc2=nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.1)
        self.batch = batch
    
    def forward(self,image,audio):
        #image input batch*50*36
        #audio input 50*1*1600
        #pose=image.view(1,50,18,2).permute()
        pose=image.contiguous().view(self.batch,50,18,2,1).permute(0,3,1,2,4)#(batch,2,50,18,1) N, C, T, V, M
        pose_out=self.pose_encoder(pose).contiguous().view(self.batch,1,256)#batch,1,256
        #print("pose_out",pose_out.shape)
        
        tran_audio=audio.contiguous().view(-1,1,1600)
        audio_out=self.audio_encoder(tran_audio)#1, 50, 256
        audio_out=audio_out.view(50,self.batch,-1).transpose(0,1) #batch,50,256
        
        #pose_out,h0=self.pose_rnn(pose_out)#1,50,512
        #pose_out=self.pose_fc(pose_out)#1,50,256
        output=torch.cat([audio_out[:,-1:,:],pose_out], 1)#batch,2,256
        output=self.conv1d(output)#batch,1,255
        #output=self.fc1(output)
        #output=self.lrelu(output)
        output=self.fc2(output)#batch,1,1
        #output=self.lrelu(output)#batch,1
        output=self.sigmoid(output).view(self.batch,1)
        return output.contiguous()