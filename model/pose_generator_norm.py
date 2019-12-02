import torch
import torch.nn as nn
from model.audio_encoder import RNN

#model for generator decoder

    
# use standard conv-relu-pool approach
'''
haoran modified version
'''
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
        #self.relu = nn.ReLU()
        #self.decoder = nn.GRU(bidirectional=True,hidden_size=36, input_size=266,num_layers= 3, batch_first=True)
        #self.fc=nn.Linear(72,36)
        self.rnn_noise = nn.GRU( 10, 10, batch_first=True)
        self.rnn_noise_squashing = nn.Tanh()
        # state size. hidden_channel_num*8 x 360 x 640
        self.layer0 = nn.Linear(266,linear_hidden)
        #self.relu = nn.ReLU()
        #self.bn=nn.BatchNorm1d(50)
        self.layer1 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer2 = res_linear_layer(linear_hidden = linear_hidden)
        self.layer3 = res_linear_layer(linear_hidden = linear_hidden)
        self.dropout =  nn.Dropout(p=0.5)
        self.final_linear = nn.Linear(linear_hidden,36)
        
    def forward(self,input):
        noise = torch.FloatTensor(self.batch, 50, 10).normal_(0, 0.33).cuda()
        aux, h = self.rnn_noise(noise)
        aux = self.rnn_noise_squashing(aux)
        input = torch.cat([input, aux], 2)
        #print(input.shape)
        #input=input.squeeze().view(1,50,266)
        #input=input.squeeze().view(50,266)
        input = input.view(-1,266)
        output = self.layer0(input)
        #output = self.relu(output)
        #output = self.bn(output)
        output = self.layer1(output) + output
        output = self.layer2(output) + output
        output = self.layer3(output) + output
        output = self.dropout(output)
        output = self.final_linear(output)#,36
        output = output.view(self.batch,50,36)
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
        