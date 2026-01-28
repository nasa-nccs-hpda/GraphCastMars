
import numpy as np
import xarray as xr
import mcd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from ruamel.yaml import YAML
import logging

logger = logging.getLogger(__name__)


@dataclass
class VariableConfig:
    """Configuration for MCD variable extraction"""
    var_id: int
    name: str
    xz_level: Optional[float] = None  # For 2D variables
    description: str = ""


@dataclass
class MCDConfig:
    """Configuration for MCD data extraction"""
    # Data source
    data_location: str
    data_version: str = '6.1'
    output_path: Union[str, Path] = './output'
    
    # Query parameters
    zkey: int = 3
    hrkey: int = 0
    
    # Time ranges
    ls_range: Tuple[int, int, int] = (0, 361, 5)
    lct_range: Tuple[int, int, int] = (0, 24, 6)
    mars_year: int = 37
    
    # Mars physical constants
    mars_gravity: float = 3.71  # m/s^2
    mars_gas_constant: float = 188.9  # J/kg/K
    mars_radius: float = 3389.5e3  # m
    
    # Vertical levels (height in meters)
    vertical_heights: List[float] = field(default_factory=lambda: [
        50, 830, 1740, 3810, 5460, 7410, 9800, 12870, 14820, 17210, 20280, 24620, 32030
    ])
    
    # Corresponding pressure levels (hPa)
    pressure_levels: List[int] = field(default_factory=lambda: [
        1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
    ])
    
    # 2D variables to extract
    variables_2d: List[VariableConfig] = field(default_factory=lambda: [
        VariableConfig(93, '2m_temperature', 2, '2m temperature'),
        VariableConfig(15, 'mean_sea_level_pressure', 1, 'mean sea level pressure'),
        VariableConfig(94, '10m_u_component_of_wind', 10, '10m u-wind'),
        VariableConfig(95, '10m_v_component_of_wind', 10, '10m v-wind'),
        VariableConfig(14, 'sea_surface_temperature', 1, 'surface temperature'),
        VariableConfig(29, 'toa_incident_solar_radiation', 1, 'TOA solar radiation'),
        VariableConfig(4, 'orography', 1, 'surface orographic height'),
    ])
    
    # 3D variables to extract
    variables_3d: List[VariableConfig] = field(default_factory=lambda: [
        VariableConfig(93, 'temperature', description='Temperature'),
        VariableConfig(17, 'geopotential', description='Geopotential'),
        VariableConfig(94, 'u_component_of_wind', description='U-wind'),
        VariableConfig(95, 'v_component_of_wind', description='V-component'),
        VariableConfig(18, 'vertical_velocity', description='Vertical velocity'),
        VariableConfig(41, 'specific_humidity', description='Dust mass mixing ratio'),
    ])
    
    # Processing options
    pressure_scaling: float = 300.0  # Scale factor for pressure
    geopotential_scaling: float = 1e5 * 0.75
    
    # Placeholder options
    include_precipitation: bool = True  # Add zero precipitation field
    include_land_sea_mask: bool = True  # Add land-sea mask (all land for Mars)
    
    # Reference date for Mars Year calculation
    mars_year_reference_date: str = "1955-04-11"  # Mars Year 1, Ls=0
    mars_year_days: float = 686.97
    
    netcdf_format: str = 'NETCDF4'
    
    def __post_init__(self):
        """Convert string paths to Path objects"""
        #self.data_location = Path(self.data_location)
        self.output_path = Path(self.output_path)
    
        """Validate configuration"""
        if len(self.vertical_heights) != len(self.pressure_levels):
            raise ValueError("vertical_heights and pressure_levels must have same length")
        
        """Ensure heights are in descending order for proper calculation"""
        self.vertical_heights = sorted(self.vertical_heights, reverse=True)
        self.pressure_levels = sorted(self.pressure_levels, reverse=True)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'MCDConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        yaml = YAML(typ="safe")

        with open(config_path, "r") as f:
            config_dict = yaml.load(f)

        return cls(**config_dict)
    
    def to_yaml(self, output_path: Union[str, Path]):
        """Update or write configuration to YAML while preserving format"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        yaml = YAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)

        # Load existing YAML if it exists
        if output_path.exists():
            with open(output_path, "r") as f:
                config = yaml.load(f)
        else:
            config = {}

        # Update / insert keys from self
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                v = str(v)
            config[k] = v   # updates if exists, inserts if not

        # Write back preserving format
        with open(output_path, "w") as f:
            yaml.dump(config, f)


class MarsPhysics:
    """Mars-specific physical calculations - can be static/utility class"""
    
    @staticmethod
    def surface_geopotential(z: np.ndarray, g0: float = 3.71, R: float = 3389.5e3) -> np.ndarray:
        """
        Compute surface geopotential from orographic height.
        
        Args:
            z: Orographic height above mean radius [m]
            g0: Mean surface gravity [m/s^2]
            R: Mean Mars radius [m]
            
        Returns:
            Geopotential at surface [m^2/s^2]
        """
        g = g0 * (R / (R + z))**2
        phi = g * z
        return phi
    
    @staticmethod
    def compute_geopotential_height(p: xr.DataArray, T: xr.DataArray, 
                                    g0: float = 3.71, Rm: float = 188.9, 
                                    Z0: float = 0.0, dim: str = "level") -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute geopotential height from pressure and temperature.
        
        SIMPLIFIED: Combined your mars_Z_xr function
        
        Args:
            p: Pressure [Pa]
            T: Temperature [K]
            g0: Reference gravity [m/s^2]
            Rm: Gas constant [J/kg/K]
            Z0: Surface height [m]
            dim: Dimension name for vertical coordinate
            
        Returns:
            (Z, Phi): Geopotential height [m] and geopotential [m^2/s^2]
        """
        # Layer-mean temperature
        T_bar = 0.5 * (T.isel(**{dim: slice(None, -1)}) + T.isel(**{dim: slice(1, None)}))
        
        # Hypsometric thickness
        dZ = (Rm / g0) * T_bar * np.log(
            p.isel(**{dim: slice(None, -1)}) / p.isel(**{dim: slice(1, None)})
        ) * 100.0  # Convert to cm
        
        # Cumulative height
        Z = xr.concat(
            [xr.full_like(p.isel(**{dim: 0}), Z0), Z0 + dZ.cumsum(dim)],
            dim=dim
        )
        
        Phi = g0 * Z * 1e5  # Convert to m²/s²
        return Z, Phi
    
    @staticmethod
    def ls_to_earth_date(Ls: float, mars_year: int = 36, 
                        reference_date: str = "1955-04-11",
                        mars_year_days: float = 686.97) -> str:
        """
        Convert Mars solar longitude to Earth calendar date.
        
        SIMPLIFIED: Removed array handling (not needed in current workflow)
        
        Args:
            Ls: Solar longitude [0-360 degrees]
            mars_year: Mars Year
            reference_date: Reference epoch (MY 1, Ls=0)
            mars_year_days: Length of Mars year in Earth days
            
        Returns:
            Earth date as 'YYYY-MM-DD'
        """
        ref_epoch = datetime.fromisoformat(reference_date)
        
        # Calculate offset
        delta_years = mars_year - 1
        year_offset = delta_years * mars_year_days
        delta_days = (Ls / 360.0) * mars_year_days
        
        earth_date = ref_epoch + timedelta(days=year_offset + delta_days)
        return earth_date.strftime("%Y-%m-%d")
    
    @staticmethod
    def ls_to_date(Ls: int) -> str:
        ref_date = datetime(2022, 3, 20)  # Approximate Earth date for Mars Ls=0 in MY36
        delta_day = Ls
        return (ref_date + timedelta(days=delta_day)).strftime("%Y-%m-%d")


class MCDQueryHelper:
    """Helper class to simplify querying MCD"""
    
    def __init__(self, query, config: MCDConfig):
        self.query = query
        self.config = config
        self.query.latlon()
    
    def get_variable_slice(self, var_id: int, xz: float) -> np.ndarray:
        """
        Fetch a 2D slice of a variable at specific vertical coordinate.
        """
        self.query.xz = xz
        #self.query.latlon()
        return self.query.getextvar(var_id)
    
    def extract_2d_variables(self, datetime: datetime) -> xr.Dataset:
        """
        Extract all configured 2D variables.
        """
        coords = {
            'lat': self.query.ycoord,
            'lon': self.query.xcoord,
            'time': [datetime]
        }
        data_vars = {}
        orography = None  # Store for geopotential calculation
        
        for var_config in self.config.variables_2d:
            field = self.get_variable_slice(var_config.var_id, var_config.xz_level)
            # coords = {
            #     'lat': self.query.ycoord,
            #     'lon': self.query.xcoord,
            #     'time': [datetime]
            # }
            
            # Apply scaling for specific variables
            if var_config.name == 'mean_sea_level_pressure':
                field = field * self.config.pressure_scaling
            
            # Store orography for later use
            if var_config.name == 'orography':
                orography = field
            
            # Create DataArray
            da = xr.DataArray(
                field.T[np.newaxis, ...],
                dims=('time', 'lat', 'lon'),
                coords=coords,
                name=var_config.name,
                attrs={'description': var_config.description}
            )
            data_vars[var_config.name] = da
        
        # Compute derived variables
        if orography is not None:
            phi = MarsPhysics.surface_geopotential(
                orography, 
                self.config.mars_gravity, 
                self.config.mars_radius
            )
            data_vars['geopotential_at_surface'] = xr.DataArray(
                phi.T,
                dims=('lat', 'lon'),
                coords={'lat': coords['lat'], 'lon': coords['lon']},
                name='geopotential_at_surface'
            )
        # Add placeholder variables if configured
        if self.config.include_precipitation:
            shape = (1, len(coords['lat']), len(coords['lon']))
            data_vars['total_precipitation_6hr'] = xr.DataArray(
                np.zeros(shape),
                dims=('time', 'lat', 'lon'),
                coords=coords,
                name='total_precipitation_6hr',
                attrs={'description': 'Placeholder - Mars has no precipitation'}
            )
        # Add land-sea mask (all land for Mars)
        if self.config.include_land_sea_mask:
            shape = (len(coords['lat']), len(coords['lon']))
            data_vars['land_sea_mask'] = xr.DataArray(
                np.ones(shape),
                dims=('lat', 'lon'),
                coords={'lat': coords['lat'], 'lon': coords['lon']},
                name='land_sea_mask',
                attrs={'description': 'All land for Mars'}
            )
        return xr.Dataset(data_vars)
    
    def extract_3d_variables(self, datetime: datetime) -> xr.Dataset:
        """
        Extract all configured 3D variables.
        
        SIMPLIFIED: Uses config instead of hardcoded variables and levels
        """
        coords = {
            'time': [datetime],
            'lat': self.query.ycoord,
            'lon': self.query.xcoord,
            'level': self.config.pressure_levels
        }        
        data_vars = {}
        
        for var_config in self.config.variables_3d:
            field_list = []
            
            # Extract at each vertical level
            for height in self.config.vertical_heights:
                field = self.get_variable_slice(var_config.var_id, height)
                field_list.append(field[np.newaxis, :, :, np.newaxis])
            
            # coords = {
            #     'time': [datetime],
            #     'lat': self.query.ycoord,
            #     'lon': self.query.xcoord,
            #     'level': self.config.pressure_levels
            # }
            
            # Stack along vertical dimension
            field_3d = np.concatenate(field_list, axis=-1)
            
            # Create DataArray
            da = xr.DataArray(
                np.transpose(field_3d, (0, 2, 1, 3)),
                dims=('time', 'lat', 'lon', 'level'),
                coords=coords,
                name=var_config.name,
                attrs={'description': var_config.description}
            )
            data_vars[var_config.name] = da
        
        return xr.Dataset(data_vars)


class MCDExtractor:
    """Main extractor class"""
    
    def __init__(self, config: MCDConfig):
        self.config = config
        
        self.query = mcd.mcd_class(
            dataloc=config.data_location,
            dataver=config.data_version
        )
        self.query.zkey = config.zkey
        self.query.hrkey = config.hrkey
        
        self.query_helper = MCDQueryHelper(self.query, config)

        self.output_path = Path(config.output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def extract_for_time(self, ls: float, lct: float) -> xr.Dataset:
        """Extract data for specific solar longitude and local time"""
        # Set query parameters
        self.query.xdate = ls
        self.query.loct = lct
        
        # Convert to Earth datetime
        ymd = MarsPhysics.ls_to_date(ls)
        dt_str = f"{ymd}T{int(lct):02d}:00:00"
        dt_stamp = pd.to_datetime(dt_str)
        
        logger.info(f"Extracting Ls={ls}, lct={lct} ({dt_stamp})")
        
        # Extract 2D and 3D variables
        ds_2d = self.query_helper.extract_2d_variables(dt_stamp)
        ds_3d = self.query_helper.extract_3d_variables(dt_stamp)
        
        # Merge datasets
        ds = xr.merge([ds_2d, ds_3d])
        
        # Apply post-processing
        ds = self._postprocess(ds)
        
        return ds
    
    def _postprocess(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply post-processing transformations"""
        # Scale geopotential if present
        if 'geopotential' in ds:
            ds['geopotential'] = ds['geopotential'] * self.config.geopotential_scaling
        
        # Add any other transformations here
        return ds
    
    def _generate_filename(self, ls: float, lct: float) -> str:
        """Generate output filename"""
        ymd = MarsPhysics.ls_to_date(ls)
        return f"mcd_output_{ymd}_hr{int(lct):02d}.nc"
    
    def extract_range(self, 
                     ls_range: Optional[Tuple[int, int, int]] = None,
                     lct_range: Optional[Tuple[int, int, int]] = None,
                     skip_existing: bool = True) -> List[str]:
        """
        Extract data for a range of times
        
        Args:
            ls_range: (start, stop, step) for solar longitude
            lct_range: (start, stop, step) for local time
            skip_existing: Skip files that already exist
            
        Returns:
            List of output file paths
        """
        ls_range = ls_range or self.config.ls_range
        lct_range = lct_range or self.config.lct_range
        
        output_files = []
        
        for ls in range(*ls_range):
            for lct in range(*lct_range):
                output_file = self.output_path / self._generate_filename(ls, lct)
                
                # Skip if file exists
                if skip_existing and output_file.exists():
                    logger.info(f"Skipping existing file: {output_file}")
                    output_files.append(output_file)
                    continue
                
                try:
                    # Extract data
                    ds = self.extract_for_time(ls, lct)
                    
                    # Save to NetCDF
                    ds.to_netcdf(
                        output_file,
                        mode='w',
                        format=self.config.netcdf_format
                    )
                    
                    logger.info(f"Saved: {output_file}")
                    output_files.append(output_file)
                    
                except Exception as e:
                    logger.error(f"Error processing Ls={ls}, lct={lct}: {e}")
                    continue
        
        return output_files
    
    def extract_single(self, ls: float, lct: float, 
                       output_file: Optional[Union[str, Path]] = None) -> str:
        """
        Extract data for a single time point
        
        Args:
            ls: Solar longitude
            lct: Local time
            output_file: Optional custom output filename
            
        Returns:
            Path to output file
        """
        if output_file is None:
            output_file = self.config.output_path / self._generate_filename(ls, lct)

        ds = self.extract_for_time(ls, lct)
        ds.to_netcdf(output_file, mode='w', format=self.config.netcdf_format)
        
        logger.info(f"Saved: {output_file}")
        return output_file


def main():
    config = MCDConfig(
        data_location='/discover/nobackup/jli30/systest/MCD_6.1/data/',
        data_version='6.1',
        output_path='/discover/nobackup/projects/nccs_interns/mvu2/jli/data/hrkey0_test',
        zkey=3,
        hrkey=0,
        ls_range=(0, 5, 1),
        lct_range=(0, 24, 6),
        mars_year=37
    )
    
    extractor = MCDExtractor(config)
    output_files = extractor.extract_range()
    
    logger.info(f"Extraction complete. Generated {len(output_files)} files.")


if __name__ == "__main__":
    main()