import torch as pt
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable 
from utils import get_time_series_data,get_mean_std,get_normalized_data
from sklearn.model_selection import KFold

class LSTM_model(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM_model, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()

    def forward(self,x):
        h_0 = Variable(pt.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(pt.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        print("hn: ", hn)
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Hyper parameters
    num_epochs = 1000
    learning_rate = 0.001

    # Load, process, and split raw data
    raw_data = pd.read_csv("data/00_all_ClimateIndices_and_precip.csv",na_values=0)
    X,Y = get_time_series_data(raw_data)

    print(f"X shape: ",X.shape)
    print("(Year, month, feature)")
    print(f"Y shape: ",Y.shape)
    print("(Year, month)")

    mean,std = get_mean_std(X)
    X_norm = get_normalized_data(X,mean,std)

    # Derrive meta statistics
    feature_num = X_norm.shape[2] #calculated on data
    LAYERS = 1 # Number of layers to run, then blend
    hidden_size = feature_num * LAYERS
    FOLDS = 5

    # Define loss function
    loss = nn.MSELoss()

    # Define K-fold utility
    kfold = KFold(n_splits=FOLDS)

    # initialize hidden container
    hidden = ((pt.randn(1,7,hidden_size)),(pt.randn(1,7,hidden_size)))

    # Start training
    print("---Starting Job---")
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X_norm)):
        print(f"Starting fold # {fold}")

        # Split out training data
        cur_X_train = X_norm[train_ids]
        cur_Y_train = Y[train_ids]

        # Split out test data
        cur_X_test = X_norm[test_ids]
        cur_Y_test = Y[test_ids]

        print("Training Shape: ",cur_X_train.shape)
        print("Testing Shape: ",cur_X_test.shape)
        
        # Create tensors for later
        X_train_tensors = Variable(pt.Tensor(cur_X_train))
        X_test_tensors = Variable(pt.Tensor(cur_X_test))

        Y_train_tensors = Variable(pt.Tensor(cur_Y_train))
        Y_test_tensors = Variable(pt.Tensor(cur_Y_test))
    
        # print(X_train_tensors.shape,Y_train_tensors.shape)
        # print(X_test_tensors.shape,Y_test_tensors.shape)
        # Initialize model
        model = LSTM_model(11,feature_num,hidden_size,LAYERS,X_train_tensors.shape[1])
        optimizer = pt.optim.Adam(model.parameters(), lr=learning_rate) 

        for epoch in range(num_epochs):
            outputs = model.forward(X_train_tensors)
            optimizer.zero_grad()
            
            # print("inputs for loss function: ")
            # print("outputs: ",outputs)
            # print("Y_train_tensors: ",Y_train_tensors)

            cur_loss = loss(outputs,Y_train_tensors)

            cur_loss.backward()

            optimizer.step()

            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, cur_loss.item()))

        # out,hidden = lstm(pt.from_numpy(X_norm),hidden)
        # print("out: ",out)
        # print("hidden: ",hidden)

    # for i,year in enumerate(X_norm):
    #     print(year.shape)
    #     # print(year.view(0,1).dim())
    #     out,hidden = lstm(pt.from_numpy(year),hidden)