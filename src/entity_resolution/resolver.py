"""
Entity Resolution: Geospatial Pixel → Tradable Financial Instrument

This is THE differentiating module. Without it, you have a science project.
With it, you have an alternative data product.

The chain of custody:
  Satellite pixel → Geographic coordinate → Physical asset (WRI/JRC DB)
  → Legal subsidiary → Parent company (OpenCorporates)
  → LEI → ISIN → Ticker symbol

Data sources:
  1. WRI Global Power Plant Database (35,000+ plants worldwide)
     https://www.wri.org/research/global-database-power-plants
  2. JRC Open Power Plants Database (European energy infrastructure)
     https://data.jrc.ec.europa.eu/dataset/9810feeb-f062-49cd-8e76-8d8cfd488a05
  3. OpenCorporates (corporate hierarchy, LEI mapping)
  4. GLEIF (LEI → ISIN relationship files)

TODO for Phase 2:
  - Extend beyond power plants to oil/gas infrastructure
  - Add pipeline databases (e.g., PHMSA for US)
  - Add LNG terminal and refinery databases
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# --- For Phase 2: these will use GeoPandas with actual databases ---
# For now, define the data structures and the interface so the API
# contract is stable even before the databases are fully integrated.


@dataclass
class PhysicalAsset:
    """A geolocated industrial facility from WRI or JRC databases."""
    asset_id: str
    name: str
    latitude: float
    longitude: float
    country: str
    fuel_type: Optional[str] = None        # gas, oil, coal, etc.
    capacity_mw: Optional[float] = None    # electrical capacity
    source_database: str = "unknown"       # "wri" or "jrc"
    owner_name: Optional[str] = None       # immediate owner (often a subsidiary)
    distance_km: Optional[float] = None    # distance from plume centroid


@dataclass
class CorporateEntity:
    """Resolved corporate parent with financial identifiers."""
    subsidiary_name: str
    parent_name: Optional[str] = None
    lei: Optional[str] = None                # 20-char Legal Entity Identifier
    isin: Optional[str] = None               # International Securities ID
    ticker: Optional[str] = None             # e.g., "XOM"
    exchange: Optional[str] = None           # e.g., "NYSE"
    sector: Optional[str] = None             # GICS sector
    country_of_domicile: Optional[str] = None


@dataclass
class ResolvedEmission:
    """
    The final product: a detected emission event fully linked to a
    tradable financial instrument.

    This is what gets serialized into the API JSON response.
    A quant at a hedge fund receives this object and can immediately:
      1. Look up the ticker to check current positioning
      2. Calculate IRA Waste Emissions Charge liability
      3. Compare satellite-observed vs. self-reported emissions
      4. Adjust CVaR models for the issuer's bonds
    """
    event_uuid: str
    timestamp_utc: str
    latitude: float
    longitude: float
    flow_rate_kgh: Optional[float]          # from quantification module
    model_confidence: float
    asset: Optional[PhysicalAsset]
    corporate: Optional[CorporateEntity]

    @property
    def is_fully_resolved(self) -> bool:
        """True if we have a complete chain from pixel to ticker."""
        return (
            self.asset is not None
            and self.corporate is not None
            and self.corporate.ticker is not None
        )


class AssetDatabase:
    """
    In-memory geospatial index of industrial assets.

    Phase 1: Load CSV, do brute-force distance calculation.
    Phase 2: Build R-tree spatial index with GeoPandas for O(log n) lookups.

    The WRI database CSV columns we need:
      - gppd_idnr (unique ID)
      - name
      - latitude, longitude
      - country, country_long
      - primary_fuel
      - capacity_mw
      - owner

    The JRC database provides similar fields for European assets.
    """

    def __init__(self):
        self._assets: list[PhysicalAsset] = []
        self._loaded = False

    def load_wri_database(self, csv_path: str):
        """
        Load the WRI Global Power Plant Database.

        Download from: https://datasets.wri.org/dataset/globalpowerplantdatabase
        File: global_power_plant_database.csv
        """
        import csv

        count = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    asset = PhysicalAsset(
                        asset_id=row.get("gppd_idnr", f"wri_{count}"),
                        name=row.get("name", "Unknown"),
                        latitude=float(row["latitude"]),
                        longitude=float(row["longitude"]),
                        country=row.get("country_long", row.get("country", "")),
                        fuel_type=row.get("primary_fuel", None),
                        capacity_mw=float(row["capacity_mw"]) if row.get("capacity_mw") else None,
                        source_database="wri",
                        owner_name=row.get("owner", None),
                    )
                    self._assets.append(asset)
                    count += 1
                except (ValueError, KeyError) as e:
                    logger.debug("Skipping WRI row: %s", e)

        logger.info("Loaded %d assets from WRI database", count)
        self._loaded = True

    def load_jrc_database(self, csv_path: str):
        """
        Load the JRC Open Power Plants Database (Europe).

        Download from: https://data.jrc.ec.europa.eu/dataset/9810feeb-f062-49cd-8e76-8d8cfd488a05
        """
        import csv

        count = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    asset = PhysicalAsset(
                        asset_id=row.get("id", f"jrc_{count}"),
                        name=row.get("name", "Unknown"),
                        latitude=float(row["lat"]),
                        longitude=float(row["lon"]),
                        country=row.get("country", ""),
                        fuel_type=row.get("fuel", None),
                        capacity_mw=float(row["capacity_mw"]) if row.get("capacity_mw") else None,
                        source_database="jrc",
                        owner_name=row.get("company", None),
                    )
                    self._assets.append(asset)
                    count += 1
                except (ValueError, KeyError) as e:
                    logger.debug("Skipping JRC row: %s", e)

        logger.info("Loaded %d assets from JRC database", count)
        self._loaded = True

    def find_nearest_asset(
        self,
        lat: float,
        lon: float,
        max_distance_km: float = 5.0,
    ) -> Optional[PhysicalAsset]:
        """
        Find the nearest industrial asset to a detected plume centroid.

        Uses Haversine distance. The tolerance radius (max_distance_km)
        must account for:
          - Wind-driven plume drift (plume centroid != source location)
          - Sentinel-2 spatial resolution limits (~20m per pixel)
          - GPS uncertainty in asset databases

        5km default is conservative; can be tightened for known facilities
        with precise coordinates.

        Phase 2 TODO: Replace with GeoPandas R-tree spatial index
        for O(log n) instead of current O(n) brute force.
        """
        if not self._assets:
            logger.warning("Asset database is empty. Load WRI/JRC data first.")
            return None

        best_asset = None
        best_dist = float("inf")

        for asset in self._assets:
            dist = self._haversine(lat, lon, asset.latitude, asset.longitude)
            if dist < best_dist:
                best_dist = dist
                best_asset = asset

        if best_asset is not None and best_dist <= max_distance_km:
            best_asset.distance_km = round(best_dist, 3)
            return best_asset

        logger.info(
            "No asset within %.1f km of (%.4f, %.4f). Nearest was %.1f km.",
            max_distance_km, lat, lon, best_dist,
        )
        return None

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Haversine distance in kilometers."""
        R = 6371.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


class CorporateResolver:
    """
    Resolve asset owner → parent company → LEI → ISIN → ticker.

    This is Phase 2 work. The interface is defined now so the API
    schema is stable. Implementation will integrate:

    1. OpenCorporates API: subsidiary name → parent company
       https://api.opencorporates.com/
       ~50% of global LEIs are mapped natively

    2. GLEIF API: company name → LEI
       https://api.gleif.org/api/v1/lei-records

    3. GLEIF-ANNA relationship files: LEI → ISIN
       Published daily, maps LEI to all associated ISINs

    4. Financial mapping: ISIN → ticker
       Via SEC EDGAR, Bloomberg, or Refinitiv APIs

    For now, returns a stub that passes through the owner name.
    """

    def resolve(self, owner_name: Optional[str]) -> CorporateEntity:
        """
        Resolve an asset owner to its full corporate identity.

        Phase 1: Returns stub with just the owner name.
        Phase 2: Full API integration chain.
        """
        if not owner_name:
            return CorporateEntity(subsidiary_name="Unknown")

        # TODO Phase 2: Implement the full resolution chain
        # 1. Query OpenCorporates for parent company
        # 2. Query GLEIF for LEI
        # 3. Map LEI → ISIN via GLEIF-ANNA files
        # 4. Map ISIN → ticker via financial data provider

        return CorporateEntity(
            subsidiary_name=owner_name,
            parent_name=None,  # Phase 2
            lei=None,          # Phase 2
            isin=None,         # Phase 2
            ticker=None,       # Phase 2
        )
