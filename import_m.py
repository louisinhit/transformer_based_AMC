from torch import nn
import torch.nn.functional as F
import sys,os
from torch import nn
#base_path = os.path.dirname(os.path.realpath(__file__))
#sys.path.append(os.path.join(base_path,'../models'))
from models import RTransformer 


class RT(nn.Module):
    def __init__(self, input_size, d_model, output_size, h, rnn_type, ksize, n_level, n, dropout=0.01, emb_dropout=0.01):
        super(RT, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x):

        x = self.encoder(x)
        x = self.rt(x)
        x = x.transpose(-2,-1)
        
        o = self.linear(x[:, :, -1])
        
        return F.relu(o)
        