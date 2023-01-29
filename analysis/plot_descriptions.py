

# Most missing values are for Nov to March
# Need to decide on K-fold cross validation versus shuffle validation.
 

import pandas as pd
# import os

# print(os.getcwd())
# print(os.listdir('./AgAthon2023_data/MainData/'))

df = pd.read_csv('./AgAthon2023_data/MainData/00_all_ClimateIndices_and_precip.csv')


import seaborn as sns

import matplotlib.pyplot as plt

date_time = df['precipitation']
date_time = pd.to_datetime(date_time)
# data = [1, 2, 3]

# DF = pd.DataFrame()
# DF['precipitation'] = df['precipitation']
# DF = DF.set_index(date_time)
# plt.plot(DF)    
# plt.gcf().autofmt_xdate()



sns.pointplot(data=df, x='date', y='precipitation',)
plt.show()