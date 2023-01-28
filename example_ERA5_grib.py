#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan. 10, 2023
for processing ERA5 data sets 

https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

adaptor.mars.internal_SST_SoilVWC_1980_2021_Jan_Dec.grib
Product type:Monthly averaged reanalysis
Variable:Sea surface temperature, Volumetric soil water layer 1
'sst','swvl1'

adaptor.mars.internal_SeaIceCover_1980_2021_Jan_Dec.grib
Product type:Monthly averaged reanalysis
Variable:Sea-ice cover
'siconc'

Format:GRIB

Whole available region:
Format:GRIB

@author: liuming
"""

import xarray as xr
import matplotlib.pyplot as plt

#relative path to the data folder
datapath = "../../ERA5_monthly_gridded"


vars = ['sst','swvl1','swvl2','swvl3','swvl4','siconc']
#vars = ['sst']
figindex = 0
for var in vars:
    if var in ['sst','swvl1','swvl2','swvl3','swvl4']:
        #monthlymean SST
        gribfile = "adaptor.mars.internal_SST_SoilVWC_1980_2021_Jan_Dec.grib"
    elif var == 'siconc':
        #monthlymean Ice Cover
        gribfile = "adaptor.mars.internal_SeaIceCover_1980_2021_Jan_Dec.grib"
        
    ds = xr.open_dataset(datapath + "/" + gribfile, engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': var}})
    #time series
    time_series_mean = ds[var].mean(dim=['latitude','longitude'])
    plt.figure(figindex)
    time_series_mean.plot()
    del time_series_mean
    figindex += 1
    #map
    spatial_mean = ds[var].mean(dim=['time'])
    plt.figure(figindex)
    spatial_mean.plot()
    del spatial_mean
    figindex += 1
    del ds

