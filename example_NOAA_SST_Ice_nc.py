#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan. 10, 2023
for processing NOAA data sets (monthly mean)

https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/
NOAA Optimum Interpolation (OI) SST V2

Temporal Coverage

    Weekly means from 1981/10/29 to 1989/12/28.
    Weekly means from 1989/12/31 to 2022/10/30
    Monthly means from 1981/12 to 2022/10

icec.mnmean.nc: Monthly Means of Ice Concentration 
icec.wkmean.1990-present.nc: Weekly Means of Ice Concentration

sst.mnmean.nc: Monthly Mean of Sea Surface Temperature
sst.wkmean.1990-present.nc: Weekly Mean of Sea Surface Temperature


@author: liuming
"""

import xarray
import matplotlib.pyplot as plt

#relative path to the data folder
datapath = "../../NOAA_Ice_SST"

#You can test either SST or ice cover
vars = ['sst','icec']
figindex = 0
for var in vars:
    if var == 'sst':
        #monthlymean SST
        ncfile = "sst.mnmean.nc"
    elif var == 'icec':
        #monthlymean Ice Cover
        ncfile = "icec.mnmean.nc"

    #read netcdf file into xarray
    ds = xarray.open_dataset(datapath + "/" + ncfile)

    #plot time series

    time_series_mean = ds[var].mean(dim=['lat','lon'])
    plt.figure(figindex)
    time_series_mean.plot()
    del time_series_mean
    figindex += 1

    #plot multiple year mean
    spatial_mean = ds[var].mean(dim=['time'])
    plt.figure(figindex)
    spatial_mean.plot()
    del spatial_mean
    figindex += 1
    del ds

