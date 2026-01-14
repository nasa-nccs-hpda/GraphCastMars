## The script is used to generate the input data for GraphCast by merging the MCD data with the ERA5 data.
import os
import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe
from datetime import datetime, timedelta

def extend_time_dim(ds, n_steps=1):
    time = ds.time
    dt = (time[1] - time[0])

    new_steps = []
    for i in range(n_steps):
        template = ds.isel(time=i) 
        new_time = time[-1] + (i+1) * dt
        new_step = template.assign_coords(time=new_time)
        new_steps.append(new_step)

    ds_extended = xr.concat([ds] + new_steps, dim='time')
    return ds_extended

def load_mcd_data(mcd_file):
    """Load MCD data from a NetCDF file."""
    return xr.open_mfdataset(mcd_file, engine='netcdf4')

def load_era5_data(era5_file):
    """Load ERA5 data from a NetCDF file."""
    return xr.open_dataset(era5_file)

def regrid_mcd_data(mcd_ds, res=1.0):
    target_lat = np.arange(-90, 90 + res, res)
    target_lon = np.arange(0, 360, res)  # 0–360
    target_grid = xr.Dataset({'lat': (['lat'], target_lat),
                              'lon': (['lon'], target_lon)})
    
    regridder = xe.Regridder(mcd_ds, target_grid, 'bilinear',
                             periodic=True, ignore_degenerate=True)

    regridded_vars = {}
    for var in mcd_ds.data_vars:
        dims = mcd_ds[var].dims
        out_list = []
        for t in mcd_ds.time:
            sub = mcd_ds[var].sel(time=t)
            if dims == ("time", "lat", "lon"):
                regridded = regridder(sub)
            else:
                other_dims = [d for d in sub.dims if d not in ["lat", "lon"]]
                stacked = sub.stack(z=other_dims)
                regridded = regridder(stacked)
                regridded = regridded.unstack("z")
                regridded = regridded.transpose(*other_dims, "lat", "lon")
            out_list.append(regridded)
        regridded_vars[var] = xr.concat(out_list, dim="time")
        
    ds_out = xr.Dataset(regridded_vars, coords={"lat": target_lat, "lon": target_lon})
    return ds_out

def preprocess_mcd_data(mcd_ds):
    """Preprocess MCD data to match ERA5 resoltuion."""
    # Longitude adjustment -180~180 to 0~360
    if mcd_ds.lon.min() < 0:
        mcd_ds = mcd_ds.assign_coords(lon=(((mcd_ds.lon + 360) % 360)))
        mcd_ds = mcd_ds.sortby(mcd_ds.lon)

    # Regrid MCD data to 1-degree resolution
    mcd_ds_regridded = regrid_mcd_data(mcd_ds, res=1.0)
    return mcd_ds_regridded

def scale_mcd_data(mcd_ds, era5_ds, var_name):
    era5_sub = era5_ds[var_name].isel(time=slice(0, 6), batch=0)
    orig_min = mcd_ds[var_name].min(dim=("lat", "lon"))
    orig_max = mcd_ds[var_name].max(dim=("lat", "lon"))
    target_min = era5_sub.min(dim=("lat", "lon"))
    target_max = era5_sub.max(dim=("lat", "lon"))
    # Then index explicitly per timestep
    scaled_list = []
    for t in range(mcd_ds.sizes["time"]):
        scaled_list.append(
            (mcd_ds[var_name].isel(time=t) - orig_min[t]) /
            (orig_max[t] - orig_min[t]) *
            (target_max[t] - target_min[t]) +
            target_min[t]
        )
    scaled_data = xr.concat(scaled_list, dim="time")
    return scaled_data

def set_constants(era5_ds, var_name, value):
    if var_name in era5_ds.data_vars:
        era5_ds[var_name].values[:] = value
    return era5_ds

def constants_to_era5(era5_ds, var_names=None, var_values=None):
    for var, value in zip(var_names or [], var_values or []):
        era5_ds = set_constants(era5_ds, var, value)
    # era5_ds = set_constants(era5_ds, 'land_sea_mask', 1)
    # era5_ds = set_constants(era5_ds, 'total_precipitation_6hr', 0.0)
    # era5_ds = set_constants(era5_ds, 'geopotantial_at_surface', 25.0)
    return era5_ds

def era5_to_mean(era5_ds, var_names=None):
    mean_file = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/stats_mean_by_level.nc"
    mean_ds = xr.open_dataset(mean_file)
    for var in var_names:
    #    if var in ["2m_temperature", "temperature", "toa_incident_solar_radiation"]:
    #        continue

        da = era5_ds[var]
        mean_da = mean_ds[var]

        if "level" in da.dims:
            # Ensure levels match
            common_levels = set(da.level.values).intersection(mean_da.level.values)
            for lev in common_levels:
                era5_ds[var].loc[dict(level=lev)] = mean_da.sel(level=lev).values
        else:
            # No level: assign scalar mean value
            era5_ds[var].values[:] = mean_da.values

    return era5_ds

def main():
    exp = "mcd_Temp_wohr"
    graph_root = "/discover/nobackup/projects/QEFM/data/FMGenCast/6hr/samples"
    #mcd_root = "/discover/nobackup/projects/nccs_interns/mvu2/jli/data/revz"
    mcd_root = "/discover/nobackup/projects/nccs_interns/mvu2/jli/data/hrkey0/temperature/climatology"
    #era5_file = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/graphcast_dataset_source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
    #output_root = "/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/source-era5-mcdv3_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
    output_root = f"/explore/nobackup/projects/ilab/data/qefm/graphcast/{exp}"
    os.makedirs(output_root, exist_ok=True)

    n  = 361 
    start_date = datetime(2022, 3, 20)
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(n)]

    #series = np.arange(285, 361, 5).tolist() + np.arange(0, 286, 5).tolist()
    for idx, dstr in enumerate(dates):
        # Load MCD data
        mcd_scheme = os.path.join(mcd_root, f"mcd_temperature_{dstr}-hr00.nc")

        hrs = ["00" , "06", "12", "18"]
        mcd_files = [mcd_scheme.replace("hr00", f"hr{hr}") for hr in hrs]
        mcd_ds = load_mcd_data(mcd_files)
        mcd_ds = extend_time_dim(mcd_ds, n_steps=2)


        # Load ERA5 data
        era5_file = os.path.join(graph_root, "graph", f"graphcast-dataset-source-era5_date-{dates[idx]}_res-1.0_levels-13_steps-4.nc")
        era5_ds = load_era5_data(era5_file)
        #era5_ds = extend_time_dim(era5_ds)
        #print(era5_ds)
        #exit()
        
        mcd_ds = mcd_ds.assign_coords(time=era5_ds.time.values[:6])
        mcd_ds_preprocessed = preprocess_mcd_data(mcd_ds)

        # Scale MCD variables to match ERA5 ranges
        #
        #
        #keep_vars = ['land_sea_mask', 'geopotential_at_surface', 'geopotential', 'total_precipitation_6hr', 'specific_humidity']

        swap_vars = ['2m_temperature', 'temperature']
        mean_vars = ['geopotential', 'mean_sea_level_pressure' , 'vertical_velocity', '10m_u_component_of_wind', '10m_v_component_of_wind', 'u_component_of_wind', 'v_component_of_wind']
        const_vars = ['land_sea_mask', 'geopotential_at_surface', 'total_precipitation_6hr', 'specific_humidity']
        # note: total 13 variables to be modified; leave out 'toa_incident_solar_radiation'
        assert len(swap_vars) + len(mean_vars) + len(const_vars) == 13
        const_vs = [1, 25.0, 0.0, 0.002]
        for var in mcd_ds_preprocessed.data_vars:
            if var in era5_ds.data_vars and var in swap_vars:
                mcd_ds_preprocessed[var] = scale_mcd_data(mcd_ds_preprocessed, era5_ds, var)
                # Replace ERA5 variable with MCD variable
                era5_ds[var][0,0:6,:,:] = mcd_ds_preprocessed[var][0:6,:,:]

        # Set variables to constants or means
        era5_ds = era5_to_mean(era5_ds, var_names=mean_vars)
        era5_ds = constants_to_era5(era5_ds, var_names=const_vars, var_values=const_vs)

        # Add MCD topo
        era5_ds['geopotential_at_surface'].values[:] = mcd_ds_preprocessed['geopotential_at_surface'].values[0,:,:]
        # Save to NetCDF
        for i in range(4):
            output_file = os.path.join(output_root, f"graphcast_dataset_source-era5-mcd_date-{dates[idx]}-T{hrs[i]}_res-1.0_levels-13_steps-3.nc")
            era5_ds.isel(time=slice(i, i+3)).to_netcdf(output_file, mode='w')
#        era5_ds['time'][i] = pd.to_datetime(era5_ds['time'][i].values).strftime('%Y-%m-%dT%H:%M:%SZ')
#        era5_ds.to_netcdf(output_file)
            print(f"Saved merged data to {output_file}")

if __name__ == "__main__":
    main()

    
