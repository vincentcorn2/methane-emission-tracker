"""
ERA5 wind retrieval client.
Workstream 2: Translate Pixels to Physical Flow Rates
Queries Copernicus CDS API for hourly ERA5 10m wind components.
Falls back to 3.5 m/s climatological default on any failure.
"""
import os
import numpy as np

try:
    import cdsapi
    import xarray as xr
    CDS_AVAILABLE = True
except ImportError:
    CDS_AVAILABLE = False

FALLBACK_WIND_SPEED  = 3.5
FALLBACK_WIND_SOURCE = "climatological_fallback_3.5ms"


class ERA5Client:

    def __init__(self, cds_api_key: str = None):
        self.cds_api_key = cds_api_key or os.environ.get("CDS_API_KEY")

    def get_wind(self, lat: float, lon: float,
                 date_str: str, hour: str = "12:00") -> dict:
        if not CDS_AVAILABLE:
            return self._fallback("cdsapi not installed")
        try:
            year  = date_str[:4]
            month = date_str[5:7]
            day   = date_str[8:10]
            # ERA5 is hourly — round HH:MM to nearest whole hour
            h, m = int(hour[:2]), int(hour[3:5]) if len(hour) > 2 else 0
            if m >= 30:
                h = (h + 1) % 24
            era5_hour = "{:02d}:00".format(h)
            c = cdsapi.Client(quiet=True)
            out_file = "/tmp/era5_wind_temp.nc"
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                    ],
                    "year": year, "month": month, "day": day,
                    "time": era5_hour,
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                    "area": [lat+0.25, lon-0.25, lat-0.25, lon+0.25],
                },
                out_file,
            )
            ds = xr.open_dataset(out_file)
            u  = float(ds["u10"].mean().values)
            v  = float(ds["v10"].mean().values)
            ds.close()
            speed     = float(np.sqrt(u**2 + v**2))
            direction = float((270 - np.degrees(np.arctan2(v, u))) % 360)
            return {
                "wind_speed_ms": round(speed, 4),
                "wind_dir_deg":  round(direction, 1),
                "wind_source":   "ERA5_reanalysis",
                "era5_u_ms":     round(u, 4),
                "era5_v_ms":     round(v, 4),
            }
        except Exception as exc:
            return self._fallback(str(exc))

    @staticmethod
    def _fallback(reason: str) -> dict:
        return {
            "wind_speed_ms": FALLBACK_WIND_SPEED,
            "wind_dir_deg":  None,
            "wind_source":   FALLBACK_WIND_SOURCE,
            "era5_u_ms":     None,
            "era5_v_ms":     None,
            "_fallback_reason": reason,
        }
