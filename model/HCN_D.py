from model.audio_encoder import RNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HCN_encoder import HCN

class SelfAttentiveEncoder(nn.Module):
    def __init__(self):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.ws1 = nn.Linear(256, 20, bias=False)
        self.ws2 = nn.Linear(20, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
#        self.init_weights()
        self.attention_hops = 1

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, outp):
        size = outp.size()  # [bsz, len, nhid] 50,25,128
        compressed_embeddings = outp.contiguous().view(-1, size[2])  # [bsz*len, nhid*2]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).contiguous().view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.contiguous().view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp) #bsz,hop,nhid

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)

class seq_discriminator(nn.Module):
    def __init__(self,batch):
        super(seq_discriminator,self).__init__()
        self.audio_encoder=RNN(batch)
        self.pose_encoder=HCN()#input (batch,2,50,18,1)
        self.attention = SelfAttentiveEncoder()
        self.conv1d = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=2)
        self.fc2=nn.Linear(255,1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.1)
        self.batch = batch
    
    def forward(self,image,audio):
        #image input batch*50*36
        #audio input 50*1*1600
        #pose=image.view(1,50,18,2).permute()
        pose=image.contiguous().view(self.batch,50,18,2,1).permute(0,3,1,2,4)#(batch,2,50,18,1) N, C, T, V, M
        pose_out=self.pose_encoder(pose).contiguous().view(self.batch,1,256)#batch,1,256
        tran_audio=audio.contiguous().view(-1,1,1600)
        audio_out=self.audio_encoder(tran_audio)#1, 50, 256
        audio_out=audio_out.contiguous().view(50,self.batch,-1).transpose(0,1) #batch,50,256
        audio_out=self.attention(audio_out)
        output=torch.cat([audio_out,pose_out], 1)#batch,2,256
        output=self.conv1d(output)#batch,1,255
        output=self.fc2(output)#batch,1,1
        #output=self.lrelu(output)#batch,1
        output=self.sigmoid(output).view(self.batch,1)
        return output.contiguous()
    
    def extract_feature(self,image):
        pose=image.contiguous().view(self.batch,50,18,2,1).permute(0,3,1,2,4)#(batch,2,50,18,1) N, C, T, V, M
        outs = self.pose_encoder.extract_feature(pose)
        return outs
    