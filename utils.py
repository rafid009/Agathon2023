import numpy as np
import pandas as pd


def get_XY(df: pd.DataFrame, label, features):
    X = []
    Y = []
    y_toggle = True
    x = []
    y = []
    first = False
    for index, row in df.iterrows():
        date = row['date']
        month = int(date.split('-')[1])
        # print(f"month: {month}")
        
        if month >= 4 and month <= 10:
            if not first:
                first = True
            x.append(row[features].to_numpy())
        else:
            if len(x) != 0:
                X.append(x)
            x = []
        if y_toggle:
            if len(y) != 0:
                # print(f"y_len: {len(y)}")
                Y.append(y)
            y = []
            y_toggle = False
        else:
            if first:
                y.append(row[label])
        if month == 3:
            y_toggle = True

    if y_toggle:
        Y.append(y)
    else:
        pads = 11 - len(y)
        for pad in range(pads):
            y.append(0.0)
        Y.append(y)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def get_time_series_data(df: pd.DataFrame):
    label = 'precipitation'
    features = get_features(df, label)
    X, Y = get_XY(df, label=label, features=features)
    return X, Y

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

def get_normalized_data(X, mean, std):
    return (X - mean) / std

def get_features(df, label):
    features = df.columns.values.tolist()
    features.remove(label)
    features.remove("date")
    return features

def preprocess_and_normalize(df: pd.DataFrame):
    X, Y = get_time_series_data(df)
    mean, std = get_mean_std(X)
    X = get_normalized_data(X, mean, std)
    return X, Y, mean, std
