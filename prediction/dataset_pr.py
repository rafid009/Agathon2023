import numpy as np
import torch
from torch.utils.data import Dataset


class Data_Precipitation(Dataset):
    def __init__(self, X, Y, mean, std, is_train=True) -> None:
        super().__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.mean = mean
        self.std = std
        print(f"X: {self.X}\n\nY: {self.Y}")

    def __getitem__(self, index):
        sample = {'X': self.X[index], 'Y': self.Y[index]}
        return sample
    
    def __len__(self):
        return len(self.X)

    
