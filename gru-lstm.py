import numpy as np
import pandas as pd
from utils import *
import torch
import torch.optim as optim
from torch.optim import Adam
from prediction.model import LSTM_model
from torch.utils.data import DataLoader
from prediction.pred_utils import *
from prediction.dataset_pr import *
from tqdm import tqdm
import os
from imputation import train_imputation_model, impute

# Take loss from each layer, then final loss is average of all losses.
# LSTMCell, manually 

# If training do the feature forcing

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Hyper parameters
    num_epochs = 1000

    # Load, process, and split raw data
    raw_data = pd.read_csv("../data/00_all_ClimateIndices_and_precip.csv")

    X, Y, mean, std = preprocess_and_normalize(raw_data)

    print(f"X: {X}\nmean: {mean}\nstd: {std}")
    # Set sequence length given from get_time_series_data
    seq_length = X.shape[1]

    print(f"X shape: ",X.shape)
    print("(seasons, month, feature)")
    print(f"Y shape: ",Y.shape)
    print("(seasons, month)")

    
    # Derrive meta statistics
    feature_num = X.shape[2] #calculated on data
    hidden_size = 128
    batch_size = 32
    FOLDS = 5

    # Define K-fold utility
    # kfold = KFold(n_splits=FOLDS)

    # Start training
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_idx=-2)

    # train_imputation_model(X_train)

    X_train = impute(X_train)
    X_test = impute(X_test)

    print("Training Shape: ", X.shape)
    
    # Create tensors for later
    # X_tensors = Variable(pt.Tensor(X_norm))
    # Y_tensors = Variable(pt.Tensor(Y))

    # X_tensors = pt.nan_to_num(X_tensors)
    # Y_tensors = pt.nan_to_num(Y_tensors)

    train_dataset = Data_Precipitation(X_train, Y_train, mean, std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Data_Precipitation(X_test, Y_test, mean, std)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # # Initialize model
    model = LSTM_model(feature_num, hidden_size, seq_length)

    lr = 1e-3

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    p1 = int(0.75 * num_epochs)
    p2 = int(0.9 * num_epochs)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    avg_loss = 0
    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                outputs,loss = model(train_batch['X'], train_batch['Y'])
                print(f"loss: {loss}")
                loss.backward()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "train_loss": avg_loss / batch_no,
                        "epoch": epoch,
                    },
                    refresh=False
                )
            lr_scheduler.step()
        break
    #     # model.eval()
    #     # with torch.no_grad():
    #     #     with tqdm(test_loader) as it:
    #     #         for batch_no, test_batch in enumerate(it, start=1):
    #     #             outputs,loss = model(train_batch['X'], train_batch['Y'])
    #     #             avg_loss += loss
    #     #             it.set_postfix(
    #     #                 ordered_dict={
    #     #                     "avg_valid_loss": avg_loss / batch_no,
    #     #                     "epoch": epoch,
    #     #                 },
    #     #                 refresh=False
    #     #             )
                
    #     #     print(f"Epoch: {epoch} avg_loss: {avg_loss/batch_no}")
    # model_folder = "../saved_model"
    # if not os.path.isdir(model_folder):
    #     os.makedirs(model_folder)
    # torch.save(model.state_dict(), f"{model_folder}/lstm_model.pth")


    # out,hidden = lstm(pt.from_numpy(X_norm),hidden)
    # print("out: ",out)
    # print("hidden: ",hidden)

    # for i,year in enumerate(X_norm):
    #     print(year.shape)
    #     # print(year.view(0,1).dim())
    #     out,hidden = lstm(pt.from_numpy(year),hidden)