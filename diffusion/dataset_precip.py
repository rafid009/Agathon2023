import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
# from utils import *
# from diffusion.diff_utils import *

def parse_data(sample, rate, is_test=False):
    """
        Get mask of random points (missing at random) across channels based on k,
        where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
        as per ts imputers
    """
    if isinstance(sample, torch.Tensor):
        sample = sample.numpy()
    obs_mask = ~np.isnan(sample)
    
    if not is_test:
        shp = sample.shape
        evals = sample.reshape(-1).copy()
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, int(len(indices) * rate))
        values = evals.copy()
        values[indices] = np.nan
        mask = ~np.isnan(values)
        mask = mask.reshape(shp)
        # obs_data_intact = values.reshape(shp).copy()
        obs_data = np.nan_to_num(evals, copy=True)
        obs_data = obs_data.reshape(shp)
        obs_intact = evals.reshape(shp)
    else:
        values = sample.copy()
        values[7:] = np.nan

        shp = sample.shape
        evals = sample.reshape(-1).copy()

        mask = ~np.isnan(values)
        obs_intact = evals.reshape(-1).copy()
        obs_data = np.nan_to_num(obs_intact, copy=True)
        obs_data = obs_data.reshape(shp)
        obs_intact = obs_intact.reshape(shp)
        # obs_intact = np.nan_to_num(obs_intact, copy=True)
    return obs_data, obs_mask, mask, sample, obs_intact

class Precipitation_Dataset(Dataset):
    def __init__(self, X, mean, std, rate=0.2, is_test=False) -> None:
        super().__init__()
        
        self.eval_length = X.shape[1]
        self.observed_values = []
        self.obs_data_intact = []
        self.observed_masks = []
        self.gt_masks = []
        self.gt_intact = []
        # X, mean, std = create_synthetic_data(n_steps, num_seasons, seed=seed)
        self.mean = mean
        self.std = std

        for i in range(X.shape[0]):
            obs_val, obs_mask, mask, sample, obs_intact = parse_data(X[i], rate, is_test)
            self.observed_values.append(obs_val)
            self.observed_masks.append(obs_mask)
            self.gt_masks.append(mask)
            self.obs_data_intact.append(sample)
            self.gt_intact.append(obs_intact)
        self.gt_masks = torch.tensor(np.array(self.gt_masks), dtype=torch.float32)
        self.observed_values = torch.tensor(np.array(self.observed_values), dtype=torch.float32)
        self.obs_data_intact = np.array(self.obs_data_intact)
        self.gt_intact = np.array(self.gt_intact)
        self.observed_masks = torch.tensor(np.array(self.observed_masks), dtype=torch.float32)
        self.observed_values = ((self.observed_values - self.mean) / self.std) * self.observed_masks
        self.obs_data_intact = ((self.obs_data_intact - self.mean) / self.std) * self.observed_masks.numpy()
        self.gt_intact = ((self.gt_intact - self.mean) / self.std) * self.gt_masks.numpy()
        
    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            # "gt_mask": self.gt_masks[index],
            "obs_data_intact": self.obs_data_intact[index],
            "timepoints": np.arange(self.eval_length),
            "gt_intact": self.gt_intact
        }
        if len(self.gt_masks) == 0:
            s["gt_mask"] = None
        else:
            s["gt_mask"] = self.gt_masks[index]
        return s
    
    def __len__(self):
        return len(self.observed_values)


def get_dataloader(X_train, X_test, mean, std, batch_size=16, missing_ratio=0.2, seed=10):
    train_dataset = Precipitation_Dataset(X_train, mean, std, rate=missing_ratio)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Precipitation_Dataset(X_test, mean, std, rate=missing_ratio, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # test_dataset = Precipitation_Dataset(df, 2, rate=missing_ratio, seed=seed)
    # if is_test:
    #     test_loader = DataLoader(test_dataset, batch_size=1)
    # else:
    # test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    return train_loader, test_loader

def get_testloader(X_test, mean, std):
    test_dataset = Precipitation_Dataset(X_test, mean, std, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    return test_loader