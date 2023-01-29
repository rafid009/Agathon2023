import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from diffusion.diff_utils import *

data_file = '../../DATA/waterchallenge/MainData/00_all_ClimateIndices_and_precip.csv'

df = pd.read_csv(data_file)
features = get_features(df)
seasons = ['1']
input_folder = '../results_mse_precip'
input_folder = '/home/rafid/results_mse_precip/' # Comment out this line 
output_folder = 'data_plots'

filename = open(f'{input_folder}/all-sample-results.json', 'r')
json_data = json.load(filename)
filename.close()
for season in seasons:
    # filename.close()
    # features = ['sin', 'cos2', 'harmonic', 'weight', 'lin_comb', 'non_lin_comb']
    # given_features = ['sin', 'cos2', 'harmonic', 'weight', 'lin_comb', 'non_lin_comb']
    given_features = features
    model_name = 'DiffSAITS'
    key_name = 'diff_saits'
    # season = '2020-2021'
    # type = "agaid_X_eps_v2_ctarget_2_hum"
    d_time = 12
    folder = f"{output_folder}_all"
    if not os.path.isdir(folder):
        os.makedirs(folder)

    for feature in given_features:

        print(f'Plotting {feature}')
        
        gt = np.array(json_data[season]['target'])
        gt = np.squeeze(gt[:, features.index(feature)])
        # print(gt)
        
        df = pd.DataFrame()
        df['gt'] = gt

        diff_saits_median = np.array(json_data[season][f'{key_name}_mean'])
        diff_saits_median = np.squeeze(diff_saits_median[:, features.index(feature)])
        df['diff_saits_median'] = diff_saits_median

        # csdi_median = np.array(json_data[season][f'csdi_median'])
        # csdi_median = np.squeeze(csdi_median[:, features.index(feature)])
        diff_saits_samples = np.array(json_data[season][f'{key_name}_samples'])
        diff_saits_samples = np.squeeze(diff_saits_samples[:50, :, features.index(feature)])#np.expand_dims(np.squeeze(diff_saits_samples[:1, :, features.index(feature)]), axis=0)#
       

        df_sample = pd.DataFrame()
        for i in range(len(diff_saits_samples)):
            df_sample[str(i)]=diff_saits_samples[i]
        
        x = np.linspace(1, d_time, d_time)
        df['x'] = x
        df_sample['x'] = x
       
        df_sample = pd.melt(df_sample, id_vars='x', value_vars=[str(i) for i in range(50)])

       # print(x)
        # y = 3.0 * x
        #some confidence interval
        # ci = 1.96 * np.std(y)/np.sqrt(len(x))
        plt.figure(figsize=(8, 4))
        plt.title(f'Time Series Prediction Plot\n Season = {season}, for {feature}')
        
        # fig, ax = plt.subplots()
        sns.lineplot(data=df_sample, x='x', y='value', errorbar='sd', err_style='band', label=f'{model_name} samples', color='green')
        sns.lineplot(data=df, x='x', y='diff_saits_median', label=f'{model_name} mean', color='blue')
        sns.lineplot(data=df, x='x', y='gt', label='Ground Truth', color='orange')
        plt.xlabel("Time (Month)")
        plt.ylabel(feature)
        # ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
       

        #for i in range(len(diff_saits_samples)):
        #    plt.plot(x, diff_saits_samples[i], label=f'{model_name} sample {i}')
        plt.legend(loc='upper left')

        plt.savefig(f"{folder}/agathon_season={season}_{feature}.png", dpi=300)
        plt.close()
