import numpy as np

def train_test_split(X, Y, test_idx=-1):
    if test_idx == 0:
        X_train, X_test = X[1:], X[0]
        Y_train, Y_test = Y[1:], Y[0]
    elif test_idx == -1 or test_idx == len(X) - 1:
        X_train, X_test = X[:-1], X[-1]
        Y_train, Y_test = Y[:-1], Y[-1]
    else:
        X_test = X[test_idx]
        Y_test = Y[test_idx]
        X_train_1 = X[0:test_idx]
        Y_train_1 = Y[0:test_idx]
        X_train_2 = X[test_idx+1:]
        Y_train_2 = Y[test_idx+1:]
        X_train = np.concatenate((X_train_1, X_train_2), axis=0)
        Y_train = np.concatenate((Y_train_1, Y_train_2), axis=0)
    X_test = np.expand_dims(X_test, axis=0)
    Y_test = np.expand_dims(Y_test, axis=0)
    return X_train, X_test, Y_train, Y_test