# src/preprocessing/graphcast_formatter.py

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Literal
from datetime import datetime, timedelta
import logging

import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe

logger = logging.getLogger(__name__)


@dataclass
class VariableStrategy:
    """Configuration for how to handle each variable"""
    name: str
    strategy: Literal['mcd', 'era5_mean', 'constant', 'keep_era5']
    constant_value: Optional[float] = None
    scale_to_era5: bool = False  # Whether to scale MCD data to ERA5 range
    
    def __post_init__(self):
        if self.strategy == 'constant' and self.constant_value is None:
            raise ValueError(f"Variable {self.name} with 'constant' strategy must have constant_value")


@dataclass
class GraphCastFormatterConfig:
    """Configuration for GraphCast data formatting"""
    # Input paths
    mcd_data_path: Union[str, Path]
    era5_sample_path: Union[str, Path]  # Template ERA5 files
    era5_stats_path: Union[str, Path]   # ERA5 statistics (mean, std)
    output_path: Union[str, Path]
    
    # Date range
    start_date: str = "2022-03-20"  # YYYY-MM-DD
    num_days: int = 361
    
    # Time settings
    time_step_hours: int = 6
    num_input_steps: int = 2  # Number of input timesteps
    num_output_steps: int = 1  # Number of output/forecast timesteps
    
    # Spatial settings
    target_resolution: float = 1.0  # degrees
    
    # Variable strategies
    variable_strategies: List[VariableStrategy] = field(default_factory=list)
    
    # MCD file naming
    mcd_filename_pattern: str = "mcd_temperature_{date}-hr{hour:02d}.nc"
    
    # ERA5 file naming
    era5_filename_pattern: str = "graphcast-dataset-source-era5_date-{date}_res-{res}_levels-13_steps-4.nc"
    
    # Output file naming
    output_filename_pattern: str = "graphcast_dataset_source-era5-mcd_date-{date}-T{hour:02d}_res-{res}_levels-13_steps-{steps}.nc"
    
    # Experiment name
    experiment_name: str = "mcd_experiment"
    
    def __post_init__(self):
        """Convert paths and set defaults"""
        self.mcd_data_path = Path(self.mcd_data_path)
        self.era5_sample_path = Path(self.era5_sample_path)
        self.era5_stats_path = Path(self.era5_stats_path)
        self.output_path = Path(self.output_path)
        
        # Set default variable strategies if none provided
        if not self.variable_strategies:
            self.variable_strategies = self._default_variable_strategies()
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def _default_variable_strategies(self) -> List[VariableStrategy]:
        """Default variable handling strategies"""
        return [
            # MCD variables (scaled to ERA5 range)
            VariableStrategy('2m_temperature', 'mcd', scale_to_era5=True),
            VariableStrategy('temperature', 'mcd', scale_to_era5=True),
            
            # ERA5 mean variables
            VariableStrategy('geopotential', 'era5_mean'),
            VariableStrategy('mean_sea_level_pressure', 'era5_mean'),
            VariableStrategy('vertical_velocity', 'era5_mean'),
            VariableStrategy('10m_u_component_of_wind', 'era5_mean'),
            VariableStrategy('10m_v_component_of_wind', 'era5_mean'),
            VariableStrategy('u_component_of_wind', 'era5_mean'),
            VariableStrategy('v_component_of_wind', 'era5_mean'),
            
            # Constant variables
            VariableStrategy('land_sea_mask', 'constant', constant_value=1.0),
            VariableStrategy('geopotential_at_surface', 'mcd'),  # Use MCD topography
            VariableStrategy('total_precipitation_6hr', 'constant', constant_value=0.0),
            VariableStrategy('specific_humidity', 'constant', constant_value=0.002),
            
            # Keep original ERA5
            VariableStrategy('toa_incident_solar_radiation', 'keep_era5'),
        ]
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'GraphCastFormatterConfig':
        """Load configuration from YAML"""
        import yaml
        
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse variable strategies
        if 'variable_strategies' in config_dict:
            config_dict['variable_strategies'] = [
                VariableStrategy(**vs) for vs in config_dict['variable_strategies']
            ]
        
        return cls(**config_dict)
    
    def to_yaml(self, output_path: Union[str, Path]):
        """Save configuration to YAML"""
        import yaml
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        config_dict = {
            'mcd_data_path': str(self.mcd_data_path),
            'era5_sample_path': str(self.era5_sample_path),
            'era5_stats_path': str(self.era5_stats_path),
            'output_path': str(self.output_path),
            'start_date': self.start_date,
            'num_days': self.num_days,
            'time_step_hours': self.time_step_hours,
            'num_input_steps': self.num_input_steps,
            'num_output_steps': self.num_output_steps,
            'target_resolution': self.target_resolution,
            'mcd_filename_pattern': self.mcd_filename_pattern,
            'era5_filename_pattern': self.era5_filename_pattern,
            'output_filename_pattern': self.output_filename_pattern,
            'experiment_name': self.experiment_name,
            'variable_strategies': [
                {
                    'name': vs.name,
                    'strategy': vs.strategy,
                    'constant_value': vs.constant_value,
                    'scale_to_era5': vs.scale_to_era5
                }
                for vs in self.variable_strategies
            ]
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


class DataRegridder:
    """Handle spatial regridding operations"""
    
    def __init__(self, target_resolution: float = 1.0):
        self.target_resolution = target_resolution
        self.target_grid = self._create_target_grid()
        self._regridder_cache = {}
    
    def _create_target_grid(self) -> xr.Dataset:
        """Create target grid for regridding"""
        res = self.target_resolution
        target_lat = np.arange(-90, 90 + res, res)
        target_lon = np.arange(0, 360, res)
        
        return xr.Dataset({
            'lat': (['lat'], target_lat),
            'lon': (['lon'], target_lon)
        })
    
    def regrid_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Regrid entire dataset to target resolution"""
        # Adjust longitude to 0-360 if needed
        if ds.lon.min() < 0:
            ds = ds.assign_coords(lon=(((ds.lon + 360) % 360)))
            ds = ds.sortby(ds.lon)
        
        # Create regridder
        regridder = xe.Regridder(
            ds, self.target_grid, 'bilinear',
            periodic=True, ignore_degenerate=True
        )
        
        regridded_vars = {}
        
        for var in ds.data_vars:
            dims = ds[var].dims
            out_list = []
            
            for t in ds.time:
                sub = ds[var].sel(time=t)
                
                if dims == ("time", "lat", "lon"):
                    regridded = regridder(sub)
                else:
                    # Handle multi-dimensional variables (e.g., with levels)
                    other_dims = [d for d in sub.dims if d not in ["lat", "lon"]]
                    stacked = sub.stack(z=other_dims)
                    regridded = regridder(stacked)
                    regridded = regridded.unstack("z")
                    regridded = regridded.transpose(*other_dims, "lat", "lon")
                
                out_list.append(regridded)
            
            regridded_vars[var] = xr.concat(out_list, dim="time")
        
        target_lat = self.target_grid.lat.values
        target_lon = self.target_grid.lon.values
        
        ds_out = xr.Dataset(regridded_vars, coords={"lat": target_lat, "lon": target_lon})
        return ds_out


class VariableProcessor:
    """Process variables according to strategies"""
    
    def __init__(self, config: GraphCastFormatterConfig):
        self.config = config
        self.era5_stats = self._load_era5_stats()
        self.strategy_map = {vs.name: vs for vs in config.variable_strategies}
    
    def _load_era5_stats(self) -> xr.Dataset:
        """Load ERA5 statistics for mean/std"""
        if self.config.era5_stats_path.exists():
            return xr.open_dataset(self.config.era5_stats_path)
        else:
            logger.warning(f"ERA5 stats file not found: {self.config.era5_stats_path}")
            return None
    
    def scale_to_era5_range(self, mcd_var: xr.DataArray, era5_var: xr.DataArray, 
                           var_name: str) -> xr.DataArray:
        """Scale MCD variable to match ERA5 range"""
        # Use first 6 timesteps of ERA5 as reference
        era5_sub = era5_var.isel(time=slice(0, 6))
        
        # Compute min/max per timestep
        orig_min = mcd_var.min(dim=("lat", "lon"))
        orig_max = mcd_var.max(dim=("lat", "lon"))
        target_min = era5_sub.min(dim=("lat", "lon"))
        target_max = era5_sub.max(dim=("lat", "lon"))
        
        # Scale each timestep
        scaled_list = []
        for t in range(mcd_var.sizes["time"]):
            scaled = (
                (mcd_var.isel(time=t) - orig_min[t]) /
                (orig_max[t] - orig_min[t]) *
                (target_max[t] - target_min[t]) +
                target_min[t]
            )
            scaled_list.append(scaled)
        
        scaled_data = xr.concat(scaled_list, dim="time")
        return scaled_data
    
    def apply_strategy(self, var_name: str, era5_ds: xr.Dataset, 
                      mcd_ds: Optional[xr.Dataset] = None) -> xr.DataArray:
        """Apply processing strategy to a variable"""
        strategy = self.strategy_map.get(var_name)
        
        if strategy is None:
            logger.warning(f"No strategy defined for {var_name}, keeping ERA5 data")
            return era5_ds[var_name]
        
        if strategy.strategy == 'keep_era5':
            return era5_ds[var_name]
        
        elif strategy.strategy == 'constant':
            # Set all values to constant
            result = era5_ds[var_name].copy()
            result.values[:] = strategy.constant_value
            return result
        
        elif strategy.strategy == 'era5_mean':
            # Set to ERA5 climatological mean
            if self.era5_stats is None or var_name not in self.era5_stats:
                logger.warning(f"No ERA5 mean for {var_name}, keeping original")
                return era5_ds[var_name]
            
            result = era5_ds[var_name].copy()
            mean_da = self.era5_stats[var_name]
            
            if "level" in result.dims:
                # Match levels
                common_levels = set(result.level.values).intersection(mean_da.level.values)
                for lev in common_levels:
                    result.loc[dict(level=lev)] = mean_da.sel(level=lev).values
            else:
                result.values[:] = mean_da.values
            
            return result
        
        elif strategy.strategy == 'mcd':
            # Use MCD data
            if mcd_ds is None or var_name not in mcd_ds:
                logger.warning(f"MCD data not available for {var_name}, keeping ERA5")
                return era5_ds[var_name]
            
            mcd_var = mcd_ds[var_name]
            
            # Scale to ERA5 range if requested
            if strategy.scale_to_era5 and var_name in era5_ds:
                mcd_var = self.scale_to_era5_range(mcd_var, era5_ds[var_name], var_name)
            
            return mcd_var
        
        else:
            raise ValueError(f"Unknown strategy: {strategy.strategy}")
    
    def process_all_variables(self, era5_ds: xr.Dataset, 
                             mcd_ds: Optional[xr.Dataset] = None,
                             num_timesteps: int = 6) -> xr.Dataset:
        """Process all variables according to strategies"""
        result_ds = era5_ds.copy()
        
        for var_name in era5_ds.data_vars:
            processed_var = self.apply_strategy(var_name, era5_ds, mcd_ds)
            
            # Replace first num_timesteps with processed data
            if mcd_ds is not None and var_name in mcd_ds:
                result_ds[var_name][0, 0:num_timesteps, ...] = processed_var[0:num_timesteps, ...]
            else:
                result_ds[var_name] = processed_var
        
        return result_ds


class GraphCastFormatter:
    """Main class for formatting GraphCast input data"""
    
    def __init__(self, config: GraphCastFormatterConfig):
        self.config = config
        self.regridder = DataRegridder(config.target_resolution)
        self.processor = VariableProcessor(config)
    
    def _generate_datetime_sequence(self) -> List[datetime]:
        """Generate sequence of start times at configured intervals"""
        start = datetime.strptime(self.config.start_date, '%Y-%m-%d')
        end = start + timedelta(days=self.config.num_days)
        datetimes = list(pd.date_range(
            start, end, 
            freq=f"{self.config.time_step_hours}h").to_pydatetime())
        return datetimes
    
    def _load_mcd_files(self, date: datetime) -> xr.Dataset:
        """Load MCD files for a specific date"""
        tstamps = pd.date_range(
            start = date,
            periods = self.config.num_input_steps + self.config.num_output_steps,
            freq = f"{self.config.time_step_hours}h"
        )
        
        mcd_files = []
        for ts in tstamps:
            h = ts.hour
            date = ts.strftime('%Y-%m-%d')
            mcd_files.append(
                self.config.mcd_data_path / self.config.mcd_filename_pattern.format(
                    date=date, hour=int(h))
            )  
        # Check if files exist
        missing = [f for f in mcd_files if not f.exists()]
        if missing:
            logger.error(f"Missing MCD files: {missing}")
        
        ds = xr.open_mfdataset([str(f) for f in mcd_files if f.exists()], engine='netcdf4')
        return ds
    
    def _extend_time_dim(self, ds: xr.Dataset, n_steps: int = 1) -> xr.Dataset:
        """Extend dataset by duplicating last timesteps"""
        time = ds.time
        dt = (time[1] - time[0])
        
        new_steps = []
        for i in range(n_steps):
            template = ds.isel(time=i)
            new_time = time[-1] + (i + 1) * dt
            new_step = template.assign_coords(time=new_time)
            new_steps.append(new_step)
        
        ds_extended = xr.concat([ds] + new_steps, dim='time')
        return ds_extended
    
    def _load_era5_template(self, date: str) -> xr.Dataset:
        """Load ERA5 template file"""
        ymd = date.strftime('%Y-%m-%d')
        era5_file = self.config.era5_sample_path / self.config.era5_filename_pattern.format(
            date=ymd, res=self.config.target_resolution
        )
        
        if not era5_file.exists():
            raise FileNotFoundError(f"ERA5 template not found: {era5_file}")
        
        return xr.open_dataset(era5_file)
    
    def process_single_date(self, date: datetime) -> List[Path]:
        """Process data for a single date"""
        logger.info(f"Processing date: {date}")
        
        # Load MCD data
        mcd_ds = self._load_mcd_files(date)
        
        # Load ERA5 template
        era5_ds = self._load_era5_template(date)
        
        # Extend MCD time dimension (if needed)
        # mcd_ds = self._extend_time_dim(mcd_ds, n_steps=self.config.num_output_steps)
        
        # Align MCD time coordinates with ERA5
        num_total_steps = self.config.num_input_steps + self.config.num_output_steps
        # mcd_ds = mcd_ds.assign_coords(time=era5_ds.time.values[:num_total_steps])
        
        # Regrid MCD data
        mcd_ds_regridded = self.regridder.regrid_dataset(mcd_ds)
        
        # Process variables
        processed_ds = self.processor.process_all_variables(
            era5_ds, 
            mcd_ds_regridded,
            num_timesteps=num_total_steps
        )
        
        # Save output files (one per starting hour)
        output_files = []
        hours = [h for h in range(0, 24, self.config.time_step_hours)]
        
        for i, hour in enumerate(hours):
            output_file = self.config.output_path / self.config.output_filename_pattern.format(
                date=date,
                hour=hour,
                res=self.config.target_resolution,
                steps=self.config.num_input_steps + self.config.num_output_steps
            )
            
            # Extract relevant time slice
            time_slice = slice(i, i + self.config.num_input_steps + self.config.num_output_steps)
            processed_ds.isel(time=time_slice).to_netcdf(output_file, mode='w')
            
            logger.info(f"Saved: {output_file}")
            output_files.append(output_file)
        
        return output_files
    
    def process_all_dates(self) -> Dict[str, List[Path]]:
        """Process all dates in the configuration"""
        dates = self._generate_datetime_sequence()
        results = {}
        
        for date in dates:
            try:
                output_files = self.process_single_date(date)
                results[date] = output_files
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                results[date] = []
        
        return results


def main():
    """Example usage"""
    config = GraphCastFormatterConfig(
        mcd_data_path="/discover/nobackup/projects/nccs_interns/mvu2/jli/data/hrkey0/temperature/climatology",
        era5_sample_path="/discover/nobackup/projects/QEFM/data/FMGenCast/6hr/samples/graph",
        era5_stats_path="/discover/nobackup/jli30/QEFM/qefm-core/qefm/models/checkpoints/graphcast/stats_mean_by_level.nc",
        output_path="/explore/nobackup/projects/ilab/data/qefm/graphcast/mcd_Temp_wohr_test",
        start_date="2022-03-20",
        num_days=361,
        experiment_name="mcd_Temp_wohr"
    )
    
    formatter = GraphCastFormatter(config)
    results = formatter.process_all_dates()
    
    logger.info(f"Processing complete. Generated {sum(len(v) for v in results.values())} files.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()