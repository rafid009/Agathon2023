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
                print(f"y_len: {len(y)}")
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
    return np.array(X), np.array(Y)

def get_time_series_data(df: pd.DataFrame):
    label = 'precipitation'
    features = df.columns.values.tolist()
    features.remove(label)
    X, Y = get_XY(df, label=label, features=features)
    return X, Y