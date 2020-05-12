#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'LSTM model'

__author__ = 'David'

import sys
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layer=2, dropout=0.3):
        super(LSTM,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer, dropout=dropout)
        self.layer2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x