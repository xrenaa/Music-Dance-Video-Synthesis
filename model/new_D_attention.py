from model.audio_encoder import RNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from model.GRU import GRUModel

class SelfAttentiveEncoder(nn.Module):
    def __init__(self):
        super(SelfAttentiveEncoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.ws1 = nn.Linear(256, 40, bias=False)
        self.ws2 = nn.Linear(40, 1, bias=False)
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

class image_encoder(nn.Module):
    def __init__(self):
        super(image_encoder,self).__init__()
        self.rnn = nn.GRU(bidirectional=True, hidden_size=256, input_size=36,num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, 256)
        
    def forward(self,input):#1,50,36
        output,__=self.rnn(input)#1,50,512
        output=self.fc(output)#1,50,256
        return output.contiguous()
        

class Pose_encoder(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        #x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A

    
graph_args={"layout": 'openpose',"strategy": 'spatial'}
class seq_discriminator(nn.Module):
    def __init__(self,batch):
        super(seq_discriminator,self).__init__()
        self.audio_encoder=RNN(batch)
        #self.image_encoder=image_encoder()#1,50,256
        self.pose_encoder=Pose_encoder(2,256,graph_args,edge_importance_weighting=True)#input (1,2,50,18,1)
        #self.pose_rnn = nn.GRU(bidirectional=True,hidden_size=256, input_size=256,num_layers= 2, batch_first=True)
        #self.pose_rnn = nn.GRU(hidden_size=256, input_size=256,num_layers= 2, batch_first=True)
        self.pose_rnn = GRUModel(input_dim=256, hidden_dim=256, layer_dim=2, output_dim=256)
        #self.pose_fc = nn.Linear(512,256)
        self.conv1d = nn.Conv1d(in_channels=2,out_channels=1,kernel_size=1)
        #self.fc1=nn.Linear(512,256)
        self.fc2=nn.Linear(256,1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.1)
        self.batch = batch
        self.pose_attention = SelfAttentiveEncoder()
        self.audio_attention = SelfAttentiveEncoder()
    
    def forward(self,image,audio):
        #image input 1*50*36
        #audio input 50*1*1600
        #pose=image.view(1,50,18,2).permute()
        pose=image.contiguous().view(1,self.batch*50,18,2,1).permute(1,3,0,2,4)#(1,2,50,18,1)->(50,2,1,18,1) N, C, T, V, M
        pose_out=self.pose_encoder(pose).contiguous().view(self.batch,50,256)#1,50,256
        #print("pose_out",pose_out.shape)
        
        tran_audio=audio.contiguous().view(-1,1,1600)
        audio_out=self.audio_encoder(tran_audio)#1, 50, 256
        audio_out=audio_out.view(50,self.batch,-1).transpose(0,1)
        audio_out=self.audio_attention(audio_out)#bsz,1,512
        
        pose_out=self.pose_rnn(pose_out)#1,50,512
        #print("pose_out",pose_out.shape)
        pose_out = self.pose_attention(pose_out)
#       pose_out=self.pose_fc(pose_out)#1,50,256
        output=torch.cat([audio_out,pose_out], 1)#bsz,2,256
        #print("output",output.shape)
        output=self.conv1d(output)#1,1,256
        #output=self.fc1(output)
        #output=self.lrelu(output)
        output=self.fc2(output)#bsz,1,1
        output=self.sigmoid(output).contiguous().view(self.batch,1)
        return output.contiguous()

