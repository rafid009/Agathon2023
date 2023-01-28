import numpy as np
import pandas as pd
import os
from pandas_profiling import ProfileReport


filepath = "./00_all_ClimateIndices_and_precip.csv"

df = pd.read_csv(filepath)

features = df.columns.values

print(features)
print("\n\n")
for col in features:
    if col == "date":
        continue
    missing = df[col].isna().sum()
    print(f"For Feature: {col} data rate of missing data: {(missing*100)/len(df)}")

print(f"\n\n{df.describe()}")

profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_file("data_report.html")