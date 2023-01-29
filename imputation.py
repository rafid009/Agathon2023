import numpy as np
import pandas as pd
import torch
import pickle
from pypots.data import mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mse
from utils import *
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def train_imputation_model(X, model_file_dir='./saved_model/'):
    if not os.path.isdir(model_file_dir):
        os.makedirs(model_file_dir)
    model_file_path = f"{model_file_dir}/saits.pkl"
    saits = SAITS(n_steps=X.shape[1], n_features=X.shape[2], n_layers=3, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=3000, patience=200, device=device)
    saits.fit(X)
    pickle.dump(saits, open(model_file_path, 'wb'))

def impute(X, model_file_path='./saved_model/saits.pkl'):
    saits = pickle.load(open(model_file_path, 'rb'))
    return saits.impute(X)

if __name__=="__main__":
    filepath = "../data/00_all_ClimateIndices_and_precip.csv"
    df = pd.read_csv(filepath)
    X, Y, mean, std = preprocess_and_normalize(df)
    # train_imputation_model(X)
    print(f"Missing in original X: {np.isnan(X).sum()}")
    imputed_X = impute(X)
    print(f"Missing in X: {np.isnan(imputed_X).sum()}")