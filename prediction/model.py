import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.autograd import Variable 
from utils import get_time_series_data,get_mean_std,get_normalized_data
from sklearn.model_selection import KFold

class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length,training=False):
        super(LSTM_model, self).__init__()
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #number of hidden "nodes"
        self.seq_length = seq_length #sequence length
        self.train_flag = training #Flag to indicate if model is training model, set to True to turn on feature forcing

        self.cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.fc_1 = nn.Linear(hidden_size,seq_length) #fully connected 1
        self.fc = nn.Linear(seq_length,1) #fully connected last layer

        self.relu = nn.ReLU()

    def forward(self,x,y):
        h = Variable(torch.zeros(x.shape[0], self.hidden_size)) #hidden state
        c = Variable(torch.zeros(x.shape[0], self.hidden_size)) #internal state

        # print("h_0: ",h_0.shape)
        # print("c_0: ",c_0.shape)

        x_loss = 0.0
        # print(f"x: {x.shape}")
        for t in range(x.shape[1]):
            # print(f"x_t: {x[:,t,:].shape}")
            cur_x = x[:, t, :]
            # print("Inner loop x shape:",cur_x.shape)
            h, c = self.cell(cur_x, (h, c)) #lstm with input, hidden, and internal state
            # print("hn: ",hn.shape)
            # print("cn: ",cn.shape)

            # Linear layer
            yp = h.view(-1, self.hidden_size) #reshaping the data for Dense layer next
            yp = self.relu(yp)
            yp = self.fc_1(yp) #first Dense
            yp = self.relu(yp) #relu
            yp = self.fc(yp) #Final Output
            
            # Calculate loss
            temp_loss = torch.mean(torch.pow(y[t] - yp, 2), dim=1)
            print(f"temp: {temp_loss}")
            x_loss += torch.mean(temp_loss, dim=0)
        # print(f"x_loss: {x_loss.shape}")
        return yp, x_loss