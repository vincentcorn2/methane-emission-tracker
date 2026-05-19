"""
scripts/fetch_era5_pending.py
==============================
Batch-fetch ERA5 reanalysis wind for all sites/dates that currently use the
climatological fallback (3.5 m/s) or haven't been fetched yet.

After running this script, re-run run_cemf_neurath_belchatow.py to get ±30%
uncertainty bounds instead of ±50%.

Sites / dates that need ERA5:
  neurath          2024-06-25T10:36 UTC  (51.038°N, 6.616°E)
  neurath_20240829 2024-08-29T10:36 UTC  (51.038°N, 6.616°E)
  belchatow_20240710 2024-07-10T09:50 UTC (51.264°N, 19.331°E)
  weisweiler       2024-09-18T10:36 UTC  (50.837°N, 6.322°E)

Prerequisites:
  conda activate methane
  ~/.cdsapirc must contain valid CDS credentials

Usage:
  python scripts/fetch_era5_pending.py
  python scripts/fetch_era5_pending.py --site neurath      # single site
  python scripts/fetch_era5_pending.py --dry-run           # print only
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.era5_client import ERA5Client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PENDING = [
    {
        "site": "neurath",
        "timestamp": "2024-06-25T10:36:31Z",
        "lat": 51.038,
        "lon": 6.616,
    },
    {
        "site": "neurath_20240829",
        "timestamp": "2024-08-29T10:36:29Z",
        "lat": 51.038,
        "lon": 6.616,
    },
    {
        "site": "belchatow_20240710",
        "timestamp": "2024-07-10T09:50:31Z",
        "lat": 51.264,
        "lon": 19.331,
    },
    {
        "site": "weisweiler",
        "timestamp": "2024-09-18T10:36:19Z",
        "lat": 50.837,
        "lon": 6.322,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Batch ERA5 wind fetch for pending sites")
    parser.add_argument("--site", help="Fetch only this site (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print requests without fetching")
    args = parser.parse_args()

    targets = [p for p in PENDING if args.site is None or p["site"] == args.site]
    if not targets:
        logger.error("No matching sites found for --site=%s", args.site)
        sys.exit(1)

    client = ERA5Client()
    results = {}

    for p in targets:
        site = p["site"]
        ts = p["timestamp"]
        lat, lon = p["lat"], p["lon"]

        if args.dry_run:
            print(f"[dry-run] {site}: would fetch ERA5 for {ts} at ({lat}, {lon})")
            continue

        # get_wind signature: (lat, lon, date_str="YYYY-MM-DD", hour="HH:MM")
        date_str = ts[:10]          # "2024-06-25"
        hour_str = ts[11:16]        # "10:36"

        logger.info("Fetching ERA5 for %s at %s %s ...", site, date_str, hour_str)
        try:
            wind = client.get_wind(lat=lat, lon=lon, date_str=date_str, hour=hour_str)
            speed = wind.get("wind_speed_ms")
            direction = wind.get("wind_dir_deg")
            u = wind.get("era5_u_ms")
            v = wind.get("era5_v_ms")
            source = wind.get("wind_source", "ERA5_reanalysis")
            results[site] = wind
            logger.info(
                "  %-22s  %.2f m/s  dir=%.1f°  u=%.3f  v=%.3f  source=%s",
                site, speed or 0, direction or 0, u or 0, v or 0, source,
            )
        except Exception as exc:
            logger.error("  %s FAILED: %s", site, exc)
            results[site] = {"error": str(exc)}

    if not args.dry_run:
        print("\n=== ERA5 Results (copy into SITES dict) ===")
        for site, wind in results.items():
            print(f"\n# {site}")
            print(json.dumps({"wind": wind}, indent=4))

        print("\nNext steps:")
        print("  1. Copy the wind dicts above into SITES in run_cemf_neurath_belchatow.py")
        print("     (replace 'wind': None  with the fetched values)")
        print("  2. Run: python scripts/run_cemf_neurath_belchatow.py")
        print("  3. Confirm ±30% uncertainty in the printed records")


if __name__ == "__main__":
    main()
