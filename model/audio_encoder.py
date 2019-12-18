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
        #x should be (50,1,1600)
        #x=x.view(-1,1,1600)
        tran_x=x.contiguous().view(-1,1,1600)
        output=self.encoder(tran_x)
        output=output.view(50,self.batch,-1).transpose(0,1)
        output, _ = self.rnn(output)
        output = output.contiguous().view(self.batch,output.shape[1],-1)
        output = self.fc(output)
        return output.contiguous()
        
        
        
        
        
        
    
    
    
    
    
    
    
    

    
    
    

