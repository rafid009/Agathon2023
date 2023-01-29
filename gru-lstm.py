import numpy as np
import pandas as pd
from utils import *

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
    X, Y = get_time_series_data(raw_data)
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

    lr = 1e-3

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    p1 = int(0.75 * num_epochs)
    p2 = int(0.9 * num_epochs)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs,loss = model.forward(X_tensors,Y_tensors)

        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss))

    # out,hidden = lstm(pt.from_numpy(X_norm),hidden)
    # print("out: ",out)
    # print("hidden: ",hidden)

    # for i,year in enumerate(X_norm):
    #     print(year.shape)
    #     # print(year.view(0,1).dim())
    #     out,hidden = lstm(pt.from_numpy(year),hidden)