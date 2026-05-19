import cdsapi
import xarray as xr
import numpy as np

def get_era5_wind(lat, lon, date):
    """
    Retrieve ERA5 10m wind speed for a given location and date.
    Returns wind speed in m/s. Falls back to 3.5 m/s on any failure.
    Workstream 2: Translate Pixels to Physical Flow Rates
    """
    try:
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind"
                ],
                "year":  date[:4],
                "month": date[5:7],
                "day":   date[8:10],
                "time":  "12:00",
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],
            },
            "era5.nc"
        )
        ds = xr.open_dataset("era5.nc")
        u  = float(ds["u10"].mean().values)
        v  = float(ds["v10"].mean().values)
        ds.close()
        speed = float((u**2 + v**2)**0.5)
        print(f"ERA5 retrieved: u={u:.4f} m/s, v={v:.4f} m/s, speed={speed:.4f} m/s")
        return speed
    except Exception as e:
        print(f"ERA5 query failed: {e}")
        print("Falling back to climatological default: 3.5 m/s")
        return 3.5
