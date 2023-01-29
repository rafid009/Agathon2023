import torch
import random
import pandas as pd
import sys
import os
from pypots.imputation import SAITS
import matplotlib.pyplot as plt
import matplotlib
import pickle
# from synthetic_data import create_synthetic_data
import json
from json import JSONEncoder
import numpy as np
from diffusion.diff_utils import *
import utils
from diffusion.main_model import CSDI_Precipitation
from diffusion.dataset_precip import get_dataloader, get_testloader


np.set_printoptions(threshold=sys.maxsize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Evaluate for Generative Model
def evaluate_imputation(df, mean, std, models, mse_folder, test_idx=-1, trials=30, given_features=None):
    # given_features = given_features = ['sin', 'cos2', 'harmonic', 'weight', 'inv'] 
    nsample = 50
    # trials = 30
    season_avg_mse = {}
    # exclude_features = ['MEAN_AT', 'MIN_AT', 'AVG_AT', 'MAX_AT']
    results = {}
    models['DiffSAITS'].eval()

    mse_diff_saits_total = {}

    X_test = get_X_test(df, test_idx=test_idx)
    test_loader = get_testloader(X_test, mean, std)
    for j, test_batch in enumerate(test_loader, start=1):

        output_diff_saits = models['DiffSAITS'].evaluate(test_batch, nsample)
        samples_diff_saits, c_target, eval_points, observed_points, observed_time, obs_intact, gt_intact = output_diff_saits
        samples_diff_saits = samples_diff_saits.permute(0, 1, 3, 2)
        c_target = c_target.permute(0, 2, 1)  # (B,L,K)
        eval_points = eval_points.permute(0, 2, 1)
        observed_points = observed_points.permute(0, 2, 1)
        samples_diff_saits_mean = samples_diff_saits.mean(dim=1)
        print()
        if trials == 1:
            results[j] = {
                'target mask': eval_points[0, :, :].cpu().numpy(),
                'target': c_target[0, :, :].cpu().numpy(),
                'diff_saits_mean': samples_diff_saits_mean[0, :, :].cpu().numpy(),
                'diff_saits_samples': samples_diff_saits[0].cpu().numpy(),
            }
        else:

            for feature in given_features:
                feature_idx = given_features.index(feature)
                if eval_points[0, :, feature_idx].sum().item() == 0:
                    continue

                mse_diff_saits = ((samples_diff_saits_mean[0, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                mse_diff_saits = mse_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                if feature not in mse_diff_saits_total.keys():
                    mse_diff_saits_total[feature] = {"mean": mse_diff_saits}
                else:
                    mse_diff_saits_total[feature]["mean"] += mse_diff_saits

                # for k in range(samples_diff_saits.shape[1]):
                #     mse_diff_saits = ((samples_diff_saits[0, k, :, feature_idx] - c_target[0, :, feature_idx]) * eval_points[0, :, feature_idx]) ** 2
                #     mse_diff_saits = mse_diff_saits.sum().item() / eval_points[0, :, feature_idx].sum().item()
                #     if feature not in mse_diff_saits_total.keys():
                #         mse_diff_saits_total[feature] = {str(k): mse_diff_saits}
                #     else:
                #         if str(k) not in mse_diff_saits_total[feature].keys():
                #             mse_diff_saits_total[feature][str(k)] = mse_diff_saits
                #         else:
                #             mse_diff_saits_total[feature][str(k)] += mse_diff_saits
                
    for feature in given_features:
        print(f"\n\tFor feature = {feature}\n\tDiffSAITS mse: {mse_diff_saits_total[feature]['mean']}")

    season_avg_mse = {
        'DiffSAITS': mse_diff_saits_total
    }

    if not os.path.isdir(mse_folder):
        os.makedirs(mse_folder)
    if trials == 1:
        fp = open(f"{mse_folder}/all-sample-results.json", "w")
        json.dump(results, fp=fp, indent=4, cls=NumpyArrayEncoder)
        fp.close()
    else:
        out_file = open(f"{mse_folder}/model_mse.json", "w")
        json.dump(season_avg_mse, out_file, indent = 4)
        out_file.close()

#Evaluate for Predictive Model
def evaluate_imputation_predictive(df, mean, std, models, mse_folder, test_idx=-1):
    pass    

def run_train_test (df, config_dict, mse_folder, model_folder, n_test=5, seed=0):

    # Set random seed for reproducibility
    random.seed(seed)

    # Get number of instances
    X, Y = utils.get_time_series_data(df)
    #print(X.shape)
    n_instances = X.shape[0]

    # Randomly select n_test indices
    test_indices = random.sample(list(range(n_instances)), n_test)
    print(test_indices)


    # Loop through each randomly selected test indices
    for test_idx in test_indices:
        
        

            
        X_train, X_test, mean, std  = get_X_mean_std(df, test_idx=test_idx)
            
        config_dict_diffsaits = config_dict

        train_loader, valid_loader = get_dataloader(
            X_train,
            X_test,
            mean,
            std,
            seed=seed,
            batch_size=config_dict_diffsaits["train"]["batch_size"],
            missing_ratio=0.2,
        )
            
        model_diff_saits = CSDI_Precipitation(config_dict_diffsaits, device, target_dim=len(real_features)).to(device)    
        #model_diff_saits.load_state_dict(torch.load(f"{model_folder}/model_diffsaits.pth"))
        models = {
            'DiffSAITS': model_diff_saits
        }            
        train(
            model_diff_saits,
            config_dict_diffsaits["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=model_folder,
            filename="model_diffsaits.pth"
        )
            
        evaluate_imputation(df, mean, std, models, mse_folder, test_idx=test_idx, given_features=['precipitation'])


data_file = '../../DATA/waterchallenge/MainData/00_all_ClimateIndices_and_precip.csv'
df = pd.read_csv(data_file)


mse_folder = "../results_mse_precip"
model_folder = "./saved_model"



real_features = get_features(df)

given_features = ['precipitation']

config_dict_diffsaits = {
    'train': {
        'epochs': 1500,
        'batch_size': 32,
        'lr': 0.0009
    },      
    'diffusion': {
        'layers': 4, 
        'channels': 64,
        'nheads': 8,
        'diffusion_embedding_dim': 128,
        'beta_start': 0.0001,
        'beta_end': 0.5,
        'num_steps': 100,
        'schedule': "quad"
    },
    'model': {
        'is_unconditional': 0,
        'timeemb': 128,
        'featureemb': 16,
        'target_strategy': "mix",
        'type': 'SAITS',
        'n_layers': 3, 
        'd_time': 12,
        'n_feature': len(real_features),
        'd_model': 256,
        'd_inner': 128,
        'n_head': 4,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1,
        'diagonal_attention_mask': True
    }
}



seed = 0
n_test = 5

# Generative model
run_train_test(df, config_dict_diffsaits, mse_folder, model_folder, n_test=n_test, seed=seed)

