import xarray as xr


#relative path to the data folder
datapath = "./AgAthon2023_data/ERA5_monthly_gridded/"
preprocessed_datapath = "./preprocessed_data/"


vars = ['sst','swvl1','swvl2','swvl3','swvl4','siconc']
vars = ['sst']
# figindex = 0
for var in vars:
    if var in ['sst','swvl1','swvl2','swvl3','swvl4']:
        #monthlymean SST
        gribfile = "adaptor.mars.internal_SST_SoilVWC_1980_2021_Jan_Dec.grib"
    elif var == 'siconc':
        #monthlymean Ice Cover
        gribfile = "adaptor.mars.internal_SeaIceCover_1980_2021_Jan_Dec.grib"
        
    ds = xr.open_dataset(f"{datapath}{gribfile}", engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': var}})
    
    del ds