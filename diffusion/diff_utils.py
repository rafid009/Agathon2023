import numpy as np
import torch
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
import pickle
import json
from json import JSONEncoder
from diffusion.dataset_precip import *
import matplotlib.pyplot as plt
import matplotlib

def get_features(df: pd.DataFrame):
    features = df.columns.values.tolist()
    features.remove("date")
    return features

def get_mean_std(X):
    mean = []
    std = []
    data = X.reshape((-1, X.shape[2]))
    for feature in range(X.shape[2]):
        season_npy = data[feature]
        idx = np.where(~np.isnan(season_npy))
        mean.append(np.mean(season_npy[idx]))
        std.append(np.std(season_npy[idx]))
    mean = np.array(mean)
    std = np.array(std)
    return mean, std

def get_X_mean_std(df: pd.DataFrame, test_idx=-1):
    X = []
    features = get_features(df)
    first = False
    x = []
    for index, row in df.iterrows():
        date = row['date']
        month = int(date.split('-')[1])
        
        if month == 3:
            if not first:
                x = []
                first = True
                continue
            x.append(row[features])
            X.append(x)
            x = []
        else:
            x.append(row[features])
    if len(x) < 12:
        pads = 12 - len(x)
        for j in range(pads):
            x.append([0 for i in range(len(features))])
        X.append(x)
        
    X = np.array(X, dtype=np.float32)
    if test_idx == 0:
        X_train, X_test = X[1:], X[0]
    elif test_idx == -1 or test_idx == len(X) - 1:
        X_train, X_test = X[:-1], X[-1]
    else:
        X_test = X[test_idx]
        X_train_1 = X[0:test_idx]
        X_train_2 = X[test_idx+1:]
        X_train = np.concatenate((X_train_1, X_train_2), axis=0)
    X_test = np.expand_dims(X_test, axis=0)
    mean, std = get_mean_std(X_train)
    # print(f"X: {X.shape}\nmean: {mean.shape}\nstd: {std.shape}")
    return X_train, X_test, mean, std

def get_X_test(df, test_idx=-1):
    X = []
    features = get_features(df)
    first = False
    x = []
    for index, row in df.iterrows():
        date = row['date']
        month = int(date.split('-')[1])
        
        if month == 3:
            if not first:
                x = []
                first = True
                continue
            x.append(row[features])
            X.append(x)
            x = []
        else:
            x.append(row[features])
    if len(x) < 12:
        pads = 12 - len(x)
        for j in range(pads):
            x.append([0 for i in range(len(features))])
        X.append(x)
        
    X = np.array(X, dtype=np.float32)
    X_test = np.expand_dims(X[test_idx], axis=0)
    return X_test

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    filename=""
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + f"/{filename if len(filename) != 0 else 'model_csdi.pth'}"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "train_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
                        

                print(
                    "\n avg loss is now ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)