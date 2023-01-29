import torch as pt
import torch.nn as nn
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
        h_0 = Variable(pt.zeros(self.seq_length,self.hidden_size)) #hidden state
        c_0 = Variable(pt.zeros(self.seq_length,self.hidden_size)) #internal state

        # print("h_0: ",h_0.shape)
        # print("c_0: ",c_0.shape)

        x_loss = 0.0

        for t in range(x.size(0)):
            cur_x = x[t]
            # print("Inner loop x shape:",cur_x.shape)
            hn,cn = self.cell(cur_x, (h_0, c_0)) #lstm with input, hidden, and internal state
            # print("hn: ",hn.shape)
            # print("cn: ",cn.shape)

            # Linear layer
            yp = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
            yp = self.relu(yp)
            yp = self.fc_1(yp) #first Dense
            yp = self.relu(yp) #relu
            yp = self.fc(yp) #Final Output
            
            # Calculate loss
            x_loss += pt.sum(pt.pow(y[t] - yp,2)) / yp.shape[0] + 1e-5

            h_0 = hn

        return yp,x_loss

# Take loss from each layer, then final loss is average of all losses.
# LSTMCell, manually 

# If training do the feature forcing

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Hyper parameters
    num_epochs = 1000

    # Load, process, and split raw data
    raw_data = pd.read_csv("data/00_all_ClimateIndices_and_precip.csv",na_values=0)
    X,Y = get_time_series_data(raw_data)
    # Set sequence length given from get_time_series_data
    seq_length = X.shape[1]

    print(f"X shape: ",X.shape)
    print("(Year, month, feature)")
    print(f"Y shape: ",Y.shape)
    print("(Year, month)")

    mean,std = get_mean_std(X)
    X_norm = get_normalized_data(X,mean,std)

    # Derrive meta statistics
    feature_num = X_norm.shape[2] #calculated on data
    hidden_size = seq_length
    FOLDS = 5

    # Define K-fold utility
    kfold = KFold(n_splits=FOLDS)

    # Start training

    print("Training Shape: ",X_norm.shape)
    
    # Create tensors for later
    X_tensors = Variable(pt.Tensor(X_norm))
    Y_tensors = Variable(pt.Tensor(Y))

    X_tensors = pt.nan_to_num(X_tensors)
    Y_tensors = pt.nan_to_num(Y_tensors)

    # Initialize model
    model = LSTM_model(feature_num,hidden_size,seq_length)

    for epoch in range(num_epochs):

        outputs,loss = model.forward(X_tensors,Y_tensors)

        loss.backward()

        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss))

    # out,hidden = lstm(pt.from_numpy(X_norm),hidden)
    # print("out: ",out)
    # print("hidden: ",hidden)

    # for i,year in enumerate(X_norm):
    #     print(year.shape)
    #     # print(year.view(0,1).dim())
    #     out,hidden = lstm(pt.from_numpy(year),hidden)