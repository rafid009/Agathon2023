import numpy as np
import matplotlib.pyplot as plt
import json
import os
from diffusion.diff_utils import *

data_file = '../../DATA/waterchallenge/MainData/00_all_ClimateIndices_and_precip.csv'
df = pd.read_csv(data_file)
features = get_features(df)
seasons = ['1']
input_folder = '../results_mse_precip'
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

        gt = np.array(json_data[season]['target'])
        gt = np.squeeze(gt[:, features.index(feature)])
        # print(gt)

        diff_saits_median = np.array(json_data[season][f'{key_name}_mean'])
        diff_saits_median = np.squeeze(diff_saits_median[:, features.index(feature)])

        # csdi_median = np.array(json_data[season][f'csdi_median'])
        # csdi_median = np.squeeze(csdi_median[:, features.index(feature)])
        diff_saits_samples = np.array(json_data[season][f'{key_name}_samples'])
        diff_saits_samples = np.squeeze(diff_saits_samples[:50, :, features.index(feature)])#np.expand_dims(np.squeeze(diff_saits_samples[:1, :, features.index(feature)]), axis=0)#
        
        
        x = np.linspace(1, d_time, d_time)
        # print(x)
        # y = 3.0 * x
        #some confidence interval
        # ci = 1.96 * np.std(y)/np.sqrt(len(x))
        plt.figure(figsize=(16, 9))
        plt.title(f'Plots for season = {season} for feature {feature}')
        # fig, ax = plt.subplots()
        plt.plot(x, gt, label='Ground Truth')
        plt.plot(x, diff_saits_median, label=f'{model_name} mean')
        # ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)
        
        for i in range(len(diff_saits_samples)):
            plt.plot(x, diff_saits_samples[i], label=f'{model_name} sample {i}')
        # plt.legend()

        plt.savefig(f"{folder}/agathon_{feature}.png", dpi=300)
        plt.close()
