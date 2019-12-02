import torch
import torch.nn as nn
import math
from math import ceil

def prime_factors(number):
    factor = 2
    factors = []
    while factor * factor <= number:
        if number % factor:
            factor += 1
        else:
            number //= factor
            factors.append(int(factor))
    if number > 1:
        factors.append(int(number))
    return factors


def calculate_padding(kernel_size, stride=1, in_size=0):
    out_size = ceil(float(in_size) / float(stride))
    return int((out_size - 1) * stride + kernel_size - in_size)


def calculate_output_size(in_size, kernel_size, stride, padding):
    return int((in_size + padding - kernel_size) / stride) + 1


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    # def __init__(self,in_dim,activation):
    def __init__(self, in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        
        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                # x : input feature maps( B X C X W X H)
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature 
                # attention: B X N X N (N is Width*Height)
                attention: B X N X N
        """
        # m_batchsize,C,width ,height = x.size()
        m_batchsize, C, N = x.size()
        # proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_query  = self.query_conv(x).view(m_batchsize, -1, N).permute(0,2,1)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, N) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        # proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        proj_value = self.value_conv(x).view(m_batchsize, -1, N)

        out = torch.bmm(proj_value,attention.permute(0, 2, 1) )
        # out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma * out + x
        return out

class Encoder(nn.Module):
    def __init__(self, code_size=256, rate=16000, feat_length=0.1, init_kernel=0.005, init_stride=0.001, num_feature_maps=16,
                 increasing_stride=True):
        super(Encoder, self).__init__()

        self.code_size = code_size
        self.cl = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.strides = []
        self.kernels = []

        features = feat_length * rate
        strides = prime_factors(features)
        kernels = [2 * s for s in strides]

        if init_kernel is not None and init_stride is not None:
            self.strides.append(int(init_stride * rate))
            self.kernels.append(int(init_kernel * rate))
            padding = calculate_padding(init_kernel * rate, stride=init_stride * rate, in_size=features)
            init_features = calculate_output_size(features, init_kernel * rate, stride=init_stride * rate,
                                                        padding=padding)
            strides = prime_factors(init_features)
            kernels = [2 * s for s in strides]

        if not increasing_stride:
            strides.reverse()
            kernels.reverse()

        self.strides.extend(strides)
        self.kernels.extend(kernels)

        for i in range(len(self.strides) - 1):
            padding = calculate_padding(self.kernels[i], stride=self.strides[i], in_size=features)
            features = calculate_output_size(features, self.kernels[i], stride=self.strides[i], padding=padding)
            pad = int(math.ceil(padding / 2.0))

            if i == 0:
                self.cl.append(
                    nn.Conv1d(1, num_feature_maps, self.kernels[i], stride=self.strides[i], padding=pad))
                self.activations.append(nn.Sequential(nn.BatchNorm1d(num_feature_maps), nn.ReLU(True)))
                
            else:
                self.cl.append(nn.Conv1d(num_feature_maps, 2 * num_feature_maps, self.kernels[i],
                                         stride=self.strides[i], padding=pad))
                self.activations.append(nn.Sequential(nn.BatchNorm1d(2 * num_feature_maps), nn.ReLU(True)))

                num_feature_maps *= 2
                
               

        self.cl.append(nn.Conv1d(num_feature_maps, self.code_size, features))
        self.activations.append(nn.Tanh())
        
    def forward(self, x):
        for i in range(len(self.strides)):
            x = self.cl[i](x)
            x = self.activations[i](x)

        return x.squeeze()
    
    
class RNN(nn.Module):
    def __init__(self,batch):
        super(RNN, self).__init__()
        self.encoder = Encoder()
        self.rnn = nn.GRU(bidirectional=True,hidden_size=256, input_size=256,num_layers= 2, batch_first=True)
        #self.rnn = nn.GRU(hidden_size=256, input_size=256,num_layers= 2, batch_first=True)
        self.fc = nn.Linear(512, 256)
        self.batch=batch
        
    def forward(self, x):
        #x should be (50,bsz,1600)
        length,bsz,sample_rate = x.shape
        tran_x = x.contiguous().view(-1,1,1600)
        output = self.encoder(tran_x)
        output = output.view(length,bsz,-1).transpose(0,1)
        output, _ = self.rnn(output)
        output = output.contiguous().view(self.batch,output.shape[1],-1)
        output = self.fc(output)
        return output.contiguous()
    

    
class res_linear_layer(nn.Module):
    
    def __init__(self, linear_hidden = 1024,time=1024):
        super(res_linear_layer,self).__init__()
        self.layer = nn.Sequential(        
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU(),
            nn.Linear(linear_hidden, linear_hidden),
            nn.BatchNorm1d(time),
            nn.ReLU()
                      
        )
    def forward(self,input):
        output = self.layer(input)
        return output
        
class hr_pose_generator(nn.Module):
    def __init__(self,batch,hidden_channel_num=64,input_c = 266,linear_hidden = 1024):
        super(hr_pose_generator,self).__init__()
        self.batch=batch
        self.rnn_noise = nn.GRU(10, 10, batch_first=True)
        self.rnn_noise_squashing = nn.Tanh()
        self.layer0 = nn.Linear(266,linear_hidden)
        self.layer1 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer2 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer3 = res_linear_layer(linear_hidden = linear_hidden)
        self.dropout =  nn.Dropout(p=0.5)
        self.final_linear = nn.Linear(linear_hidden,36)
        
    def forward(self,input):
        bsz,length,feature = input.shape
        noise = torch.FloatTensor(bsz, length, 10).normal_(0, 0.33).cuda()
        aux, h = self.rnn_noise(noise)
        aux = self.rnn_noise_squashing(aux)
        input = torch.cat([input, aux], 2)
        #print(input.shape)
        input = input.view(-1,266)
        output = self.layer0(input)
        #output = self.relu(output)
        #output = self.bn(output)
        output = self.layer1(output) + output
        output = self.layer2(output) + output
        output = self.layer3(output) + output
        output = self.dropout(output)
        output = self.final_linear(output)#,36
        output = output.view(self.batch,length,36)
        output = self.rnn_noise_squashing(output)
        return output
    
class Generator(nn.Module):
    def __init__(self,batch):
        super(Generator,self).__init__()    
        self.audio_encoder=RNN(batch)
        self.pose_generator=hr_pose_generator(batch)
        self.batch=batch

    def forward(self,input):
        output=self.audio_encoder(input)#input 50,1,1600
        output=self.pose_generator(output)#1，50，36
        return output#1,50,36