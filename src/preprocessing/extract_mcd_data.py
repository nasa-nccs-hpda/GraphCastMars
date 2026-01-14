#from fmcd import mcd
import os
import numpy as np
import mcd
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def ls_to_earth_date(Ls, mars_year=36):
    """
    Convert Mars solar longitude (Ls) to Earth calendar date (approximate).
    
    Parameters
    ----------
    Ls : float or array-like
        Solar longitude(s) in degrees [0–360).
    mars_year : int
        Mars Year (MY). Default = 36 (around 2021-2023).
    
    Returns
    -------
    list of str
        Earth dates as 'YYYY-MM-DD'.
    """
    # Mars year length in Earth days
    mars_year_days = 686.97  
    
    # Reference: Mars Year 1 began on 1955-04-11 (Earth date, Ls=0)
    ref_epoch = datetime(1955, 4, 11)  
    
    # offset years
    delta_years = mars_year - 1
    year_offset = delta_years * mars_year_days
    
    # ensure iterable
    Ls = np.atleast_1d(Ls)
    results = []
    
    for L in Ls:
        delta_days = (L / 360.0) * mars_year_days
        earth_date = ref_epoch + timedelta(days=year_offset + delta_days)
        results.append(earth_date.strftime("%Y-%m-%d"))
    
    return results if len(results) > 1 else results[0]

def mars_Z_xr(p, T, dim="level", g0=3.71, Rm=188.9, Z0=0.0):
    T_bar = 0.5 * (T.isel(**{dim: slice(None, -1)}) + T.isel(**{dim: slice(1, None)}))
    dZ = (Rm / g0) * T_bar * np.log(p.isel(**{dim: slice(None, -1)}) / p.isel(**{dim: slice(1, None)})) * 100.0  # Convert to cm
    Z = xr.concat(
        [xr.full_like(p.isel(**{dim: 0}), Z0), Z0 + dZ.cumsum(dim)],
        dim=dim
    )#.assign_coords({dim: p[dim]})
    Phi = g0 * Z * 1e5  # Convert to m²/s²
    return Z, Phi

# def compute_geopotential_mars(T, P, g0=3.71, Rm=188.9, Z0=0.0):
#     """
#     Compute geopotential height on Mars from 3D pressure and temperature arrays.

#     Parameters
#     ----------
#     p : ndarray
#         Pressure [Pa], shape (lat, lon, level). Must be ordered bottom→top.
#     T : ndarray
#         Temperature [K], shape (lat, lon, level), same shape as p.
#     g0 : float
#         Reference gravity [m/s^2].
#     Rm : float
#         Gas constant for Mars air [J/kg/K].
#     Z0 : float
#         Surface geopotential height at bottom level [m].

#     Returns
#     -------
#     Z : ndarray
#         Geopotential height [m], same shape as p.
#     Phi : ndarray
#         Geopotential [m^2/s^2], same shape as p.
#     """

#     # Layer-mean temperature: average along level axis
#     T_bar = 0.5 * (T[:, :, :-1] + T[:, :, 1:])

#     # Hypsometric thickness ΔZ for each layer
#     dZ = (Rm / g0) * T_bar * np.log(P[:, :, :-1] / P[:, :, 1:]) * 100.0

#     # Allocate Z array
#     Z = np.zeros_like(P)
#     Z[:, :, 0] = Z0

#     # Cumulative sum along vertical axis
#     Z[:, :, 1:] = Z0 + np.cumsum(dZ, axis=2)

#     # Geopotential
#     Phi = g0 * Z * 1e5
#     return Z, Phi

def mars_surface_geopotential(z: np.ndarray) -> np.ndarray:
    """
    Compute surface geopotential on Mars from orographic height.

    Parameters
    ----------
    z : float or ndarray
        Orographic height above mean radius [m].

    Returns
    -------
    phi : float or ndarray
        Geopotential at surface [m^2/s^2].
    """
    g0 = 3.72076         # m/s^2, mean surface gravity
    R = 3389.5e3         # m, mean Mars radius
    g = g0 * (R / (R + z))**2
    phi = g * z
    return phi

def get_single_var(var_id, level, query):
    """
    Fetch a single variable at a specific vertical level.
    """
    query.xz = level
    query.latlon()
    # Fetch the variable data
    field = query.getextvar(var_id)
    return field

def get_slice(query, var_id, xz):
    """
    Fetch a 2D slice of a variable at a specific xz coordinate.
    """
    query.xz = xz
    query.latlon()
    # Fetch the variable data
    field = query.getextvar(var_id)
    return field

def extract_2d_vars(query, datetime=None):
    """
    Get 2D variable.
    """
    t2m = get_slice(query, 93, 2)  # 2m temperature
    mslp = get_slice(query, 15, 1)  # mean sea level pressure
    u10 = get_slice(query, 94, 10)  # 10m u-component of wind
    v10 = get_slice(query, 95, 10)  # 10m v-component of wind
    sst = get_slice(query, 14, 1)  # using surface temperature to represent SST
    rain = np.zeros_like(t2m)  # Placeholder for rain, as MCD does not provide this
    lsm = np.ones_like(t2m)  # Placeholder for land-sea mask, 1=land
    toasw = get_slice(query, 29, 1)  # Top of atmosphere shortwave radiation
    orog = get_slice(query, 4, 1)  # Surface orographic height
    # Compute surface geopotential and height
    phi = mars_surface_geopotential(orog)

    coords = {
        'lat': query.ycoord,  # assumes query.lat is 1D array
        'lon': query.xcoord,  # assumes query.lon is 1D array
        'time': [datetime]
    }

    def to_da(var, name):
        return xr.DataArray(
            var[np.newaxis, ...],  # Add a new axis for time
            dims=('time', 'lat', 'lon'),
            coords=coords,
            name=name
        )
    ds_2d = xr.Dataset({
        '2m_temperature': to_da(t2m.T, '2m_temperature'),
        'mean_sea_level_pressure': to_da(mslp.T * 300.0, 'mean_sea_level_pressure'),
        '10m_u_component_of_wind': to_da(u10.T, '10m_u_component_of_wind'),
        '10m_v_component_of_wind': to_da(v10.T, '10m_v_component_of_wind'),
        'sea_surface_temperature': to_da(sst.T, 'sea_surface_temperature'),
        'geopotential_at_surface': to_da(phi.T, 'geopotential_at_surface'),
        'total_precipitation_12hr': to_da(rain.T, 'total_precipitation_12hr'),
        'land_sea_mask': to_da(lsm.T, 'land_sea_mask'),
        'toa_incident_solar_radiation': to_da(toasw.T, 'toa_incident_solar_radiation'),
    })
    return ds_2d

def extract_3d_vars(query, datetime=None):
    """
    Get 3D variables.
    """
    var_ids = [93, 17, 94, 95, 18, 41]  
    # Temperature, Pressure, U-component, V-component, Vertical velocity, Dust mass mixing ratio
    var_names = ['temperature', 'geopotential', 'u_component_of_wind','v_component_of_wind', 'vertical_velocity', 'specific_humidity']
    # vertical altitudes
    height = [50, 830, 1740, 3810, 5460, 7410, 9800, 12870, 14820, 17210, 20280, 24620, 32030]
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    # Initialize a dictionary to hold DataArrays
    data_vars = {}
    for id, vname in zip(var_ids, var_names):
        field_list = []
        for h in height[::-1]:

            query.xz = h
            query.latlon()
            field = query.getextvar(id)
            field_list.append(field[np.newaxis, :, :, np.newaxis]) # Add new axis for time and vertical level
        
        # Stack fields along a new axis (e.g., vertical level)
        field_3d = np.concatenate(field_list, axis=-1)

        # Convert to xarray DataArray
        da = xr.DataArray(
            np.transpose(field_3d, (0, 2, 1, 3)),  # Transpose to match (time, lat, lon, level)
            dims=('time', 'lat', 'lon', 'level'),
            coords={
                'time': [datetime],  # assumes datetime is provided
                'lat': query.ycoord,  # assumes query.lat is 1D array
                'lon': query.xcoord,  # assumes query.lon is 1D array
                'level': levels  # vertical levels
            },
            name=vname
        )
        # Store in dictionary
        data_vars[vname] = da

    return xr.Dataset(data_vars)

def get_temperature(query):
    # Function to get temperature data both 2D surface and 3D profile
    t2m = get_slice(query, 93, 2)  # 2m temperature
    temp_2d = t2m[np.newaxis, :, :]  # Add new axis for time
    ds_2d = xr.DataArray(
        np.tanspose(temp_2d, (0,2,1)),  
        dims=('time', 'lat', 'lon'),
        coords={
            'time': [datetime], # assumes datetime is provided
            'lat': query.ycoord,
            'lon': query.xcoord
        },
        name='2m_temperature'
    )

    # vertical altitudes and coordinates marks
    height = [50, 830, 1740, 3810, 5460, 7410, 9800, 12870, 14820, 17210, 20280, 24620, 32030]
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    temp_profile = []
    for h in height[::-1]:
        query.xz = h
        query.latlon()
        field = query.getextvar(93)
        temp_profile.append(field[np.newaxis, :, :, np.newaxis]) # Add new axis for time and vertical level
    # Stack fields along a new axis (e.g., vertical level)
    temp_3d = np.concatenate(temp_profile, axis=-1)
    ds_3d = xr.DataArray(
        np.transpose(temp_3d, (0, 2, 1, 3)),  # Transpose to match (time, lat, lon, level)
        dims=('time', 'lat', 'lon', 'level'),
        coords={
            'time': [datetime],  # assumes datetime is provided
            'lat': query.ycoord,  # assumes query.lat is 1D array
            'lon': query.xcoord,  # assumes query.lon is 1D array
            'level': levels  # vertical levels
        },
        name='temperature'
    )
    return ds_2d, ds_3d

dset = '/discover/nobackup/jli30/systest/MCD_6.1/data/'
query = mcd.mcd_class(dataloc=dset, dataver='6.1')
 # self.loct      = 0. 0  # Local time in hours
 # self.xdate  = 0.0      # "Mars date": xdate is the value of Ls

if __name__ == "__main__":
    # set up query parameters
    query.zkey = 3  # vertical coordinate key (e.g., height above surface)
    query.hrkey = 0  # no hi-res topo
    output_path = "/discover/nobackup/projects/nccs_interns/mvu2/jli/data/hrkey0"

    ##query.zkey = 4  # vertical coordinate key (e.g., pressure)
    ##levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    #query.latlon()
    #vars = query.getextvar(93)



    # variables needed for GenCast
    ### 2-D variables
    # '2m_temperature',
    # 'mean_sea_level_pressure',
    # '10m_v_component_of_wind',
    # '10m_u_component_of_wind',
    # 'sea_surface_temperature',
    ###
    ### 3-D variables
    # "temperature",
    # "geopotential",
    # "u_component_of_wind",
    # "v_component_of_wind",
    # "vertical_velocity",
    # "specific_humidity",
    ###

    #levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    # variables available in MCD 6.1
    #    14: "Surface temperature (K)", \
	#    15: "Surface pressure (Pa)", \
    #    18: "Downward Vertical wind component (m/s)" \
    #    41: "Dust mass mixing ratio (kg/kg)", \
    #    91: "Pressure (Pa)", \
    #    92: "Density (kg/m3)", \
    #    93: "Temperature (K)", \
    #    94: "W-E wind component (m/s)", \
    #    95: "S-N wind component (m/s)", \
    for Ls in range(0, 361, 5):
        # solar longitude (Ls) for Mars, 0-360 degrees
        # Ls = (month-1) * 30.0
        query.xdate = Ls
        ymd = ls_to_earth_date(Ls, mars_year=37)
        for lct in range(0, 24, 6):
        # Local time in hours with 12 hour intervals
            query.loct = lct

            dt_str = f"{ymd}T{lct:02d}:00:00"  # Example datetime string
            dt_stamp = pd.to_datetime(dt_str)
            ofname = f"mcd_output_Ls{Ls:02d}_hr{lct:02d}-rev-z.nc"
            output_file = os.path.join(output_path, ofname)
            ds_2d = extract_2d_vars(query, datetime=dt_stamp)
            ds_3d = extract_3d_vars(query, datetime=dt_stamp)
            # Combine 2D and 3D datasets
            ds = xr.merge([ds_2d, ds_3d])

            # Compute geopotential height and add to dataset
            #p = ds['pressure'] # Assuming pressure at the lowest level
            #T = ds['temperature']
            #Z, Phi = mars_Z_xr(p, T)
            #print(Phi)
            #exit()
            #ds['geopotential_height'] = Z
            ds['geopotential'] = ds['geopotential'] * 1e5 * 0.75
            
            ds.to_netcdf(output_file, mode='w', format='NETCDF4')
            print(f"Saved dataset for Ls {Ls}, hour {lct} to {output_file}")

