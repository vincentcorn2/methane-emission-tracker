"""
scripts/timeseries/timeseries_builder.py
=========================================
OOP refactor of the per-site annual CH4Net + CEMF+IME time-series pipelines.

Consolidates:
    belchatow_annual_timeseries.py        (Bełchatów KWB mine, T34UCB)
    rybnik_chwalowice_annual_timeseries.py (KWK ROW Ruch Chwałowice, T34UCA)

Class hierarchy::

    BaseTimeseriesBuilder(ABC)
    ├── BelchatowTimeseriesBuilder         – Climate TRACE CH4 comparison
    └── RybnikChwalowiceTimeseriesBuilder  – Carbon Mapper + TROPOMI comparison

Pipeline per (year, month):
  1. Search CDSE for cloud-free L1C acquisitions in the target tile.
  2. Download + convert to .npy if not already cached.
  3. Run CH4Net v8 inference; compute S/C ratio against the site crop.
  4. If S/C > DetectionThreshold: run CEMF+IME with ERA5 wind using the
     mine polygon as the quantification boundary.
  5. Append record to results_analysis/<site>_annual_timeseries.json.

Outputs require CDSE credentials (env: CDSE_USERNAME / CDSE_PASSWORD) and
CH4Net v8 model weights.  Use ``--dry-run`` to search the catalogue only.

Usage::

    from scripts.timeseries.timeseries_builder import BelchatowTimeseriesBuilder
    builder = BelchatowTimeseriesBuilder()
    store = builder.Run(years=[2024], months=list(range(1, 13)))
    builder.PrintSummary(store)
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import zipfile
from abc import ABC, abstractmethod
from calendar import monthrange
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Path setup (must precede local imports) ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # methane-api/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "detection"))

# ── Global constants ──────────────────────────────────────────────────────────
ACQ_DATE_RE             = re.compile(r"_(\d{8})T")
DEFAULT_YEARS           = list(range(2019, 2026))   # 2019–2025
DEFAULT_MONTHS          = list(range(1, 13))
DEFAULT_MAX_CLOUD       = 20.0
DEFAULT_MAX_CLOUD_FALLBACK = 40.0
DEFAULT_MAX_CANDIDATES  = 0                         # 0 = all candidates
DETECTION_FLOOR_KGH     = 300.0                     # floor for annualisation imputation


# ── Abstract base class ───────────────────────────────────────────────────────

class BaseTimeseriesBuilder(ABC):
    """Abstract base for annual per-site CH4Net + CEMF+IME time-series builders.

    Subclasses must implement all abstract properties to declare site-specific
    configuration and all abstract methods for external-reference loading,
    per-record annotation, summary building, and console output.

    Parameters
    ----------
    log_to_file : bool
        If True (default) write a log file alongside the JSON output.
    """

    def __init__(self, log_to_file: bool = True) -> None:
        log_path = (
            ROOT / "results_analysis" / f"{self.SiteName}_annual_timeseries.log"
        )
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_to_file:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(str(log_path)))
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
        )
        self._log = logging.getLogger(f"{self.SiteName}_timeseries")

    # ── Abstract properties (site configuration) ──────────────────────────────

    @property
    @abstractmethod
    def SiteName(self) -> str:
        """Short snake_case identifier, e.g. ``'belchatow'``."""

    @property
    @abstractmethod
    def TileId(self) -> str:
        """MGRS tile identifier, e.g. ``'T34UCB'``."""

    @property
    @abstractmethod
    def SiteLat(self) -> float:
        """Decimal-degree latitude of the emission source pin."""

    @property
    @abstractmethod
    def SiteLon(self) -> float:
        """Decimal-degree longitude of the emission source pin."""

    @property
    @abstractmethod
    def MinePolygon(self) -> list[tuple[float, float]]:
        """Ordered (lat, lon) vertices of the mine/concession polygon.

        Used to build the quantification mask: pixels outside this polygon
        are excluded from the CEMF+IME flow-rate computation.
        """

    @property
    @abstractmethod
    def DetectionThreshold(self) -> float:
        """Classic S/C ratio threshold for declaring a CH4 detection."""

    @property
    @abstractmethod
    def ConformalTau(self) -> float:
        """Conformal τ at α=0.10 (n_cal=35, global guarantee)."""

    @property
    @abstractmethod
    def ScOffsets(self) -> tuple[float, float, float, float]:
        """(offset_N, offset_S, offset_E, offset_W) in decimal degrees.

        Control-crop offsets passed to ``compute_sc_ratio``.  Use asymmetric
        values when the mine spans more than ~20 km in one direction.
        """

    @property
    @abstractmethod
    def DownloadDir(self) -> Path:
        """Local directory for raw S2 ZIP downloads (deleted after .npy conversion)."""

    @property
    @abstractmethod
    def OutJson(self) -> Path:
        """Path to the output JSON time-series file."""

    # ── Abstract methods (site-specific behaviour) ────────────────────────────

    @abstractmethod
    def _LoadExternalReferences(self) -> dict:
        """Load site-specific reference data (CT CSV, CM CSV, TROPOMI JSON, …).

        Returns a dict whose keys are understood by :meth:`_AnnotateRecord`
        and :meth:`_BuildSummary`.
        """

    @abstractmethod
    def _AnnotateRecord(
        self, rec: dict, year: int, month: int, refs: dict
    ) -> None:
        """In-place: attach site-specific reference fields to a record dict.

        Parameters
        ----------
        rec:   the record dict being built for this (year, month) acquisition
        year, month: calendar coordinates
        refs:  the dict returned by :meth:`_LoadExternalReferences`
        """

    @abstractmethod
    def _BuildSummary(
        self,
        store:        dict,
        records:      list[dict],
        years:        list[int],
        obs_months:   list[str],
        det_months:   list[str],
        detected:     list[dict],
        above_tau:    list[dict],
        cfar_pass:    list[dict],
        refs:         dict,
    ) -> dict:
        """Build and return the site-specific summary dict.

        Called once after the main loop completes.  Responsible for
        annualisation, external-reference comparison, and all site-specific
        summary fields.
        """

    @abstractmethod
    def PrintSummary(self, store: dict) -> None:
        """Print a human-readable summary to stdout from a completed store dict."""

    # ── Concrete utility helpers ───────────────────────────────────────────────

    @staticmethod
    def _GetCredentials() -> tuple[str, str]:
        """Return (username, password) from env vars or interactive prompt."""
        user = os.environ.get("CDSE_USERNAME")
        pw   = os.environ.get("CDSE_PASSWORD")
        if user and pw:
            return user, pw
        user = input("CDSE username: ")
        pw   = getpass.getpass("CDSE password: ")
        return user, pw

    @staticmethod
    def _BboxWkt(lat: float, lon: float, margin: float = 0.25) -> str:
        """Return a WKT POLYGON bounding box centred on (lat, lon)."""
        return (
            f"POLYGON(({lon - margin} {lat - margin},{lon + margin} {lat - margin},"
            f"{lon + margin} {lat + margin},{lon - margin} {lat + margin},"
            f"{lon - margin} {lat - margin}))"
        )

    @staticmethod
    def _MonthWindow(year: int, month: int) -> tuple[str, str]:
        """Return (start_iso, end_iso) for the full calendar month."""
        last_day = monthrange(year, month)[1]
        start = f"{year}-{month:02d}-01T00:00:00.000Z"
        end   = f"{year}-{month:02d}-{last_day:02d}T23:59:59.999Z"
        return start, end

    def _CachedForDate(self, date_str: str) -> list[Path]:
        """Return .npy paths in NPY_CACHE matching tile and acquisition date."""
        from apply_bitemporal_diff import NPY_CACHE  # type: ignore[import]
        out = []
        for p in NPY_CACHE.glob(f"*{self.TileId}*.npy"):
            if "_ref_" in p.name or "MSIL1C" not in p.name:
                continue
            m = ACQ_DATE_RE.search(p.name)
            if m and m.group(1) == date_str.replace("-", ""):
                out.append(p)
        return sorted(out)

    @staticmethod
    def _ZipIsValid(zip_path: str) -> bool:
        """Return True if the file exists and is a readable ZIP archive."""
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.namelist()
            return True
        except Exception:
            return False

    def _DownloadOne(self, client, product) -> Path:
        """Download a single S2 product, convert to .npy, delete the ZIP.

        Parameters
        ----------
        client:  a ``CopernicusClient`` instance
        product: a product object returned by ``client.search_products``

        Returns
        -------
        Path
            Absolute path to the newly-created .npy file.

        Raises
        ------
        RuntimeError
            If the download returns None or a corrupt ZIP after two attempts.
        """
        from src.ingestion.preprocessing import safe_to_npy  # type: ignore[import]
        from apply_bitemporal_diff import NPY_CACHE           # type: ignore[import]

        self.DownloadDir.mkdir(parents=True, exist_ok=True)
        NPY_CACHE.mkdir(parents=True, exist_ok=True)
        self._log.info("  Downloading %s ...", product.name[:70])

        zip_path = client.download_product(product, str(self.DownloadDir))
        if zip_path is None:
            raise RuntimeError(
                f"download_product returned None for {product.name}"
            )
        if not self._ZipIsValid(zip_path):
            self._log.warning(
                "  ZIP appears corrupt — re-downloading: %s", Path(zip_path).name
            )
            Path(zip_path).unlink(missing_ok=True)
            zip_path = client.download_product(product, str(self.DownloadDir))
            if zip_path is None or not self._ZipIsValid(zip_path):
                raise RuntimeError(
                    f"Re-download also produced corrupt ZIP for {product.name}"
                )
            self._log.info("  Re-download OK: %s", Path(zip_path).name)

        self._log.info("  Converting to .npy ...")
        extract_dir = tempfile.mkdtemp(prefix=f"s2_{self.SiteName}_annual_")
        try:
            npy_path, _ = safe_to_npy(
                zip_path=zip_path,
                output_dir=str(NPY_CACHE),
                tile_id=self.TileId,
                acquisition_date=product.acquisition_date,
                satellite=product.satellite,
                extract_dir=extract_dir,
            )
            self._log.info("  Saved: %s", Path(npy_path).name)
            try:
                Path(zip_path).unlink()
                self._log.info("  Deleted ZIP: %s", Path(zip_path).name)
            except Exception as exc:
                self._log.warning("  Could not delete ZIP %s: %s", zip_path, exc)
            return Path(npy_path)
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def _AcquireAllMonth(
        self,
        year:           int,
        month:          int,
        client,
        max_cloud:      float,
        dry_run:        bool,
        max_candidates: int = 0,
    ) -> list[dict]:
        """Search CDSE and return per-acquisition dicts for one calendar month.

        Downloads (or pulls from cache) all cloud-free L1C acquisitions for
        the site tile in the given month.  Providing more observations per month
        avoids missing emission events that occurred on a non-optimal acquisition
        day.

        Parameters
        ----------
        max_candidates: stop after this many successful downloads (0 = all).

        Returns
        -------
        list of ``{"npy_path": Path|None, "search": dict}`` dicts.
        A single-element list with ``status="no_products"`` or
        ``"search_failed"`` when no usable acquisitions are found.
        """
        start, end = self._MonthWindow(year, month)
        self._log.info(
            "── %d-%02d : searching CDSE (cloud <= %.0f%%) ──",
            year, month, max_cloud,
        )

        try:
            products = client.search_products(
                wkt_polygon=self._BboxWkt(self.SiteLat, self.SiteLon),
                start_date=start,
                end_date=end,
                collection="SENTINEL-2",
                max_cloud_cover=max_cloud,
            )
        except Exception as exc:
            self._log.error("  CDSE search failed: %s", exc)
            return [{"npy_path": None,
                     "search": {"status": "search_failed", "error": str(exc)}}]

        l1c = [
            p for p in products
            if self.TileId in getattr(p, "tile_id", "")
            and "MSIL1C" in p.name
        ]
        self._log.info("  %d L1C candidates", len(l1c))

        if not l1c:
            return [{"npy_path": None,
                     "search": {"status": "no_products",
                                "month": f"{year}-{month:02d}"}}]

        l1c.sort(key=lambda p: (getattr(p, "cloud_cover", None) or 100.0))

        results: list[dict] = []
        for product in l1c:
            meta = {
                "product_name":     product.name,
                "cloud_cover":      getattr(product, "cloud_cover", None),
                "acquisition_date": product.acquisition_date,
            }
            cached = self._CachedForDate(product.acquisition_date[:10])
            if cached:
                self._log.info("  Cached: %s", cached[0].name)
                meta["status"] = "cached"
                results.append({"npy_path": cached[0], "search": meta})
                continue

            if dry_run:
                meta["status"] = "would_download"
                results.append({"npy_path": None, "search": meta})
                continue

            try:
                npy_path = self._DownloadOne(client, product)
                meta["status"] = "downloaded"
                results.append({"npy_path": npy_path, "search": meta})
            except Exception as exc:
                self._log.error(
                    "  Download failed for %s: %s", product.name[:60], exc
                )
                meta["status"] = "download_failed"
                meta["error"]  = str(exc)
                results.append({"npy_path": None, "search": meta})

            if max_candidates > 0 and len(results) >= max_candidates:
                self._log.info(
                    "  max_candidates=%d reached, stopping early.", max_candidates
                )
                break

        return results

    def _InferenceAndSc(self, npy_path: Path, detector) -> dict:
        """Run CH4Net inference (or load from cache) and compute S/C ratio.

        Parameters
        ----------
        npy_path: path to the input .npy tile
        detector: a ``CH4NetDetector`` instance (or None to skip inference)

        Returns
        -------
        dict with keys: tif, sc_ratio, sc_cfar, site_mean, ctrl_mean,
        ctrl_mu, ctrl_sigma, cv_ctrl, cfar_thresh_ratio, cfar_detect,
        cfar_margin, uniform_field.  Returns ``{"status": "no_geo_meta"}``
        if the geo-metadata companion file is absent.
        """
        from apply_bitemporal_diff import (  # type: ignore[import]
            OUT_DIR, compute_sc_ratio, find_geo_meta, run_inference,
        )

        geo_meta = find_geo_meta(npy_path)
        if geo_meta is None:
            return {"status": "no_geo_meta"}

        site_dir = OUT_DIR / self.SiteName
        site_dir.mkdir(parents=True, exist_ok=True)
        out_tif = site_dir / f"original_{npy_path.stem}.tif"

        if not out_tif.exists():
            self._log.info("  Running CH4Net inference ...")
            target = np.load(npy_path)
            run_inference(target, detector, geo_meta, out_tif)
            del target
        else:
            self._log.info("  Inference cached: %s", out_tif.name)

        off_n, off_s, off_e, off_w = self.ScOffsets
        sc = compute_sc_ratio(
            out_tif, self.SiteLat, self.SiteLon,
            offset_n=off_n, offset_s=off_s,
            offset_e=off_e, offset_w=off_w,
        )

        # Uniform-field guard: site_mean ≈ ctrl_mean → CH4Net produced a
        # spatially degenerate probability map (snow, uniform cloud-free scene).
        sm = sc.get("site_mean") or 0.0
        cm_val = sc.get("ctrl_mean") or 0.0
        uniform_field = bool(sm > 0 and abs(sm - cm_val) < 1e-4)

        return {
            "tif":               str(out_tif),
            "sc_ratio":          sc.get("sc_ratio"),
            "sc_cfar":           sc.get("sc_cfar"),
            "site_mean":         sc.get("site_mean"),
            "ctrl_mean":         sc.get("ctrl_mean"),
            "ctrl_mu":           sc.get("ctrl_mu"),
            "ctrl_sigma":        sc.get("ctrl_sigma"),
            "cv_ctrl":           sc.get("cv_ctrl"),
            "cfar_thresh_ratio": sc.get("cfar_thresh_ratio"),
            "cfar_detect":       sc.get("cfar_detect"),
            "cfar_margin":       sc.get("cfar_margin"),
            "uniform_field":     uniform_field,
        }

    def _Quantify(self, npy_path: Path, sc_record: dict, era5_client) -> dict:
        """If S/C exceeds DetectionThreshold, run CEMF+IME with ERA5 wind.

        Uses the mine polygon (``self.MinePolygon``) to bound the quantification
        crop: pixels outside the polygon are masked before computing the flow rate.

        Parameters
        ----------
        npy_path:   path to the input .npy tile
        sc_record:  the dict returned by :meth:`_InferenceAndSc`
        era5_client: an ``ERA5Client`` instance

        Returns
        -------
        dict with ``status`` key.  Status values:
        ``"below_threshold"``, ``"no_acquisition_date"``,
        ``"quant_failed"``, ``"quantified"``.
        """
        sc = sc_record.get("sc_ratio")
        if sc is None or sc <= self.DetectionThreshold:
            return {"status": "below_threshold", "sc_ratio": sc}

        self._log.info(
            "  S/C = %.2f >= %.2f -> running quantification",
            sc, self.DetectionThreshold,
        )

        import rasterio
        from rasterio.features import geometry_mask
        from shapely.geometry import Polygon as ShapelyPolygon

        from apply_bitemporal_diff import B11_IDX, B12_IDX, lonlat_to_pixel  # type: ignore[import]
        from src.quantification.runner import SiteCfg, run_quantification      # type: ignore[import]

        tif_path = Path(sc_record["tif"])
        with rasterio.open(tif_path) as src:
            prob_full = src.read(1).astype(np.float32)
            H, W = prob_full.shape
            transform = src.transform

            poly_rows, poly_cols = [], []
            for plat, plon in self.MinePolygon:
                pr, pc = lonlat_to_pixel(tif_path, plon, plat)
                poly_rows.append(pr)
                poly_cols.append(pc)

            r0 = max(0, min(poly_rows))
            r1 = min(H, max(poly_rows))
            c0 = max(0, min(poly_cols))
            c1 = min(W, max(poly_cols))
            self._log.info(
                "  Mine polygon bounding box: rows %d–%d, cols %d–%d "
                "(%d×%d px @ 10m = %.1f×%.1f km)",
                r0, r1, c0, c1, r1 - r0, c1 - c0,
                (r1 - r0) * 10 / 1000, (c1 - c0) * 10 / 1000,
            )

            poly_xy = [
                (transform.c + c * transform.a, transform.f + r * transform.e)
                for r, c in zip(poly_rows, poly_cols)
            ]
            poly_crs = ShapelyPolygon(poly_xy)
            window_transform = rasterio.transform.from_bounds(
                transform.c + c0 * transform.a,
                transform.f + r1 * transform.e,
                transform.c + c1 * transform.a,
                transform.f + r0 * transform.e,
                c1 - c0, r1 - r0,
            )
            mine_mask = ~geometry_mask(
                [poly_crs.__geo_interface__],
                out_shape=(r1 - r0, c1 - c0),
                transform=window_transform,
                invert=False,
            )

        prob = prob_full[r0:r1, c0:c1]
        del prob_full

        arr  = np.load(npy_path)
        b11  = arr[r0:r1, c0:c1, B11_IDX].astype(np.float32) / 255.0
        b12  = arr[r0:r1, c0:c1, B12_IDX].astype(np.float32) / 255.0
        del arr

        mask_original = ((prob >= 0.18) & mine_mask).astype(np.float32)
        self._log.info(
            "  Polygon mask: %d / %d px active (%.1f%% of bounding box)",
            int(mine_mask.sum()), mine_mask.size,
            100 * mine_mask.sum() / mine_mask.size,
        )

        m = ACQ_DATE_RE.search(npy_path.name)
        acq = m.group(1) if m else None
        if acq is None:
            return {"status": "no_acquisition_date"}
        acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T10:00:00Z"

        cfg = SiteCfg(
            site=self.SiteName,
            scene_id=npy_path.stem,
            acquisition_timestamp=acq_iso,
            lat=self.SiteLat,
            lon=self.SiteLon,
            b11=b11,
            b12=b12,
            mask_original=mask_original,
            mask_bitemporal=None,
            era5_hour="10:00",
            ch4net_peak_probability=float(prob.max()),
        )

        try:
            record = run_quantification(cfg, dry_run=True, era5_client=era5_client)
        except Exception as exc:
            self._log.error("  Quantification failed: %s", exc)
            return {"status": "quant_failed", "error": str(exc)}

        flags = []
        if record.wind_source != "ERA5_reanalysis":
            flags.append("WIND_FALLBACK")

        return {
            "status":                     "quantified",
            "sc_ratio":                   sc,
            "flow_rate_kgh":              record.flow_rate_kgh,
            "flow_rate_lower_kgh":        record.flow_rate_lower_kgh,
            "flow_rate_upper_kgh":        record.flow_rate_upper_kgh,
            "wind_speed_ms":              record.wind_speed_ms,
            "wind_dir_deg":               record.wind_dir_deg,
            "wind_source":                record.wind_source,
            "uncertainty_pct":            record.uncertainty_pct,
            "annual_tonnes_if_continuous": record.annual_tonnes_if_continuous,
            "n_plume_pixels":             record.n_plume_pixels,
            "governance_flags":           flags,
        }

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def Run(
        self,
        years:              list[int] | None = None,
        months:             list[int] | None = None,
        max_cloud:          float = DEFAULT_MAX_CLOUD,
        max_cloud_fallback: float = DEFAULT_MAX_CLOUD_FALLBACK,
        dry_run:            bool  = False,
        max_candidates:     int   = DEFAULT_MAX_CANDIDATES,
    ) -> dict:
        """Run the full annual time-series pipeline and return the store dict.

        Parameters
        ----------
        years:              calendar years to process (default: 2019–2025)
        months:             months per year (default: 1–12)
        max_cloud:          primary CDSE cloud-cover ceiling (%)
        max_cloud_fallback: retry ceiling when primary search finds nothing
        dry_run:            search catalogue only; no downloads or inference
        max_candidates:     max acquisitions to evaluate per month (0 = all)

        Returns
        -------
        dict
            ``{"site": …, "years": …, "records": […], "summary": {…}}``
        """
        from src.ingestion.copernicus_client import CopernicusClient  # type: ignore[import]
        from src.ingestion.era5_client import ERA5Client               # type: ignore[import]
        from apply_bitemporal_diff import CH4NetDetector, WEIGHTS      # type: ignore[import]

        years  = years  or DEFAULT_YEARS
        months = months or DEFAULT_MONTHS
        refs   = self._LoadExternalReferences()

        self.OutJson.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialise persisted store
        if self.OutJson.exists():
            with open(self.OutJson) as f:
                store = json.load(f)
            bak = self.OutJson.with_suffix(".json.bak")
            with open(bak, "w") as f:
                json.dump(store, f, indent=2)
            self._log.info(
                "Loaded existing JSON (%d records); backup -> %s",
                len(store.get("records", [])), bak.name,
            )
        else:
            store = {"site": self.SiteName, "years": years, "records": []}

        records: list[dict] = store.setdefault("records", [])
        store["years"] = years
        store["site"]  = self.SiteName

        # Already-processed lookup (skip by product_name; retry download_failed)
        done_products = {
            r.get("search", {}).get("product_name")
            for r in records
            if "detection" in r and r.get("search", {}).get("product_name")
        }
        done_no_products = {
            r.get("month")
            for r in records
            if r.get("search", {}).get("status") == "no_products"
        }

        def _save() -> None:
            with open(self.OutJson, "w") as fp:
                json.dump(store, fp, indent=2)

        # Initialise clients
        user, pw = self._GetCredentials()
        cdse = CopernicusClient(username=user, password=pw)
        era5 = ERA5Client()

        detector = None
        if not dry_run:
            self._log.info("Loading CH4Net v8 weights ...")
            detector = CH4NetDetector(WEIGHTS)

        # Main loop
        for year in years:
            for month in months:
                month_key = f"{year}-{month:02d}"
                if month_key in done_no_products:
                    self._log.info("=" * 65)
                    self._log.info(
                        "── %s : no products (confirmed), skipping", month_key
                    )
                    continue

                self._log.info("=" * 65)
                acquisitions = self._AcquireAllMonth(
                    year, month, cdse, max_cloud, dry_run, max_candidates
                )

                # Retry with looser cloud ceiling if nothing found
                if (
                    len(acquisitions) == 1
                    and acquisitions[0]["search"].get("status") == "no_products"
                    and max_cloud_fallback > max_cloud
                ):
                    self._log.info(
                        "  Retrying with cloud <= %.0f%% ...", max_cloud_fallback
                    )
                    acquisitions = self._AcquireAllMonth(
                        year, month, cdse, max_cloud_fallback,
                        dry_run, max_candidates,
                    )

                first_status = acquisitions[0]["search"].get("status")
                if first_status in ("no_products", "search_failed"):
                    if not any(
                        r.get("month") == month_key
                        and r.get("search", {}).get("status") == first_status
                        for r in records
                    ):
                        rec: dict = {
                            "month":  month_key,
                            "search": acquisitions[0]["search"],
                        }
                        self._AnnotateRecord(rec, year, month, refs)
                        records.append(rec)
                        _save()
                    continue

                for acq in acquisitions:
                    search_meta  = acq["search"]
                    npy_path     = acq["npy_path"]
                    product_name = search_meta.get("product_name", "")
                    acq_date     = search_meta.get("acquisition_date", "")[:10]

                    if product_name in done_products:
                        self._log.info(
                            "  Already processed: %s", product_name[:60]
                        )
                        continue

                    rec = {
                        "month":            month_key,
                        "acquisition_date": acq_date,
                        "search":           search_meta,
                    }
                    self._AnnotateRecord(rec, year, month, refs)

                    if npy_path is None or dry_run:
                        records.append(rec)
                        _save()
                        continue

                    rec["npy"] = npy_path.name
                    sc_rec     = self._InferenceAndSc(npy_path, detector)
                    rec["detection"] = sc_rec

                    uf = sc_rec.get("uniform_field", False)
                    if uf:
                        self._log.info(
                            "  %s [%s]  ← UNIFORM FIELD "
                            "(site_mean≈ctrl_mean=%.5f) — excluded",
                            month_key, acq_date, sc_rec.get("site_mean", 0),
                        )

                    if sc_rec.get("sc_ratio") is not None and not uf:
                        quant_rec = self._Quantify(npy_path, sc_rec, era5)
                        rec["quantification"] = quant_rec
                        self._log.info(
                            "  %s [%s]  S/C=%.2f  CFAR=%s  Q=%s kg/h",
                            month_key, acq_date,
                            sc_rec["sc_ratio"],
                            "DETECT" if sc_rec.get("cfar_detect") else "no",
                            f"{quant_rec.get('flow_rate_kgh'):.0f}"
                            if quant_rec.get("flow_rate_kgh") else "—",
                        )
                    elif uf:
                        rec["quantification"] = {"status": "uniform_field_excluded"}

                    records.append(rec)
                    done_products.add(product_name)
                    _save()

        # Post-run statistics
        self._log.info("=" * 65)
        self._log.info("Annual time series complete. Records: %d", len(records))

        detected = [
            r for r in records
            if r.get("quantification", {}).get("status") == "quantified"
        ]
        above_tau = [
            r for r in records
            if (r.get("detection") or {}).get("sc_ratio") is not None
            and not r.get("detection", {}).get("uniform_field")
            and r["detection"]["sc_ratio"] > self.ConformalTau
        ]
        cfar_pass = [
            r for r in records
            if (r.get("detection") or {}).get("cfar_detect")
            and not r.get("detection", {}).get("uniform_field")
        ]
        valid_acq = [
            r for r in records
            if "detection" in r
            and not r.get("detection", {}).get("uniform_field")
        ]
        obs_months = sorted({r["month"] for r in valid_acq})
        det_months = sorted({r["month"] for r in detected})

        summary = self._BuildSummary(
            store, records, years, obs_months, det_months,
            detected, above_tau, cfar_pass, refs,
        )
        store["summary"] = summary
        _save()
        return store

    def SaveResults(self, store: dict, path: Path | None = None) -> None:
        """Write ``store`` to JSON.  Defaults to ``self.OutJson``."""
        out = path or self.OutJson
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(store, f, indent=2)

    # ── Shared annualisation helper ────────────────────────────────────────────

    @staticmethod
    def _AnnualisationFields(
        detected:    list[dict],
        det_months:  list[str],
        obs_months:  list[str],
    ) -> dict:
        """Compute the three annualisation framings (month-level).

        Non-detection months at a persistent emitter are missing observations,
        not zero-emission months.  Three framings bound the true annual total:
        (1) upper: every month emits at the mean detected rate.
        (2) floor-imputed: non-detection months imputed at the detection floor.
        (3) lower bound: non-detection months treated as Q = 0.

        Returns a dict ready to be merged into the summary.
        """
        flows = [
            r["quantification"]["flow_rate_kgh"]
            for r in detected
            if r["quantification"].get("flow_rate_kgh") is not None
        ]
        if not flows:
            return {}

        n_det_mo  = len(det_months)
        n_obs_mo  = len(obs_months)
        n_ndet_mo = n_obs_mo - n_det_mo
        mean_q    = float(np.mean(flows))

        return {
            "mean_flow_kg_h":                     mean_q,
            "median_flow_kg_h":                   float(np.median(flows)),
            "min_flow_kg_h":                      float(np.min(flows)),
            "max_flow_kg_h":                      float(np.max(flows)),
            "annualised_t_per_yr_upper":           float(mean_q * 8760 / 1000),
            "annualised_t_per_yr_floor_imputed":   float(
                (mean_q * (n_det_mo / n_obs_mo)
                 + DETECTION_FLOOR_KGH * (n_ndet_mo / n_obs_mo)) * 8760 / 1000
            ),
            "annualised_t_per_yr_lower_bound":     float(
                mean_q * 8760 / 1000 * (n_det_mo / n_obs_mo)
            ),
            "detection_floor_kgh_assumed":         DETECTION_FLOOR_KGH,
            "n_detection_months":                  n_det_mo,
            "n_non_detection_months":              n_ndet_mo,
            "n_observed_months":                   n_obs_mo,
        }


# ── Bełchatów KWB coal mine (T34UCB) ─────────────────────────────────────────

class BelchatowTimeseriesBuilder(BaseTimeseriesBuilder):
    """Annual CH4Net + CEMF+IME time series for KWB Bełchatów coal mine.

    Site: Climate TRACE asset 16168, KWB Bełchatów, 51.242°N 19.275°E.
    External reference: Climate TRACE monthly CH4 emissions CSV.
    Output: ``results_analysis/belchatow_annual_timeseries.json``
    """

    @property
    def SiteName(self) -> str:
        return "belchatow"

    @property
    def TileId(self) -> str:
        return "T34UCB"

    @property
    def SiteLat(self) -> float:
        return 51.242

    @property
    def SiteLon(self) -> float:
        return 19.275

    @property
    def MinePolygon(self) -> list[tuple[float, float]]:
        """Approximate OSM boundary corners of KWB Bełchatów (~21 km × 4.2 km)."""
        return [
            (51.257,  19.097),   # NW
            (51.2566, 19.390),   # NE
            (51.219,  19.3996),  # SE
            (51.2185, 19.099),   # SW
        ]

    @property
    def DetectionThreshold(self) -> float:
        return 1.15

    @property
    def ConformalTau(self) -> float:
        return 3.5796

    @property
    def ScOffsets(self) -> tuple[float, float, float, float]:
        """Asymmetric offsets — mine spans ~21 km E-W."""
        return (0.20, 0.20, 0.30, 0.39)

    @property
    def DownloadDir(self) -> Path:
        return ROOT / "data" / "downloads" / "annual"

    @property
    def OutJson(self) -> Path:
        return ROOT / "results_analysis" / "belchatow_annual_timeseries.json"

    _CT_CSV = ROOT / "data" / "16168_climate_trace_ch4.csv"

    def _LoadExternalReferences(self) -> dict:
        """Load Climate TRACE monthly CH4 CSV → ``{(year, month): t_ch4}``."""
        ct: dict[tuple[int, int], float] = {}
        if not self._CT_CSV.exists():
            self._log.warning(
                "Climate TRACE CSV not found at %s — CT comparison disabled",
                self._CT_CSV,
            )
            return {"ct_ch4": ct}
        with open(self._CT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("gas", "").strip().lower() != "ch4":
                    continue
                parts = row["start_time"].split("-")
                key = (int(parts[0]), int(parts[1]))
                ct[key] = float(row["emissions_quantity"])
        self._log.info(
            "Loaded %d monthly CT CH4 records (2021–2025)", len(ct)
        )
        return {"ct_ch4": ct}

    def _AnnotateRecord(
        self, rec: dict, year: int, month: int, refs: dict
    ) -> None:
        rec["ct_ch4_t_month"] = refs["ct_ch4"].get((year, month))

    def _BuildSummary(
        self,
        store:        dict,
        records:      list[dict],
        years:        list[int],
        obs_months:   list[str],
        det_months:   list[str],
        detected:     list[dict],
        above_tau:    list[dict],
        cfar_pass:    list[dict],
        refs:         dict,
    ) -> dict:
        ct_ch4 = refs.get("ct_ch4", {})
        summary: dict = {
            "years":                  years,
            "site":                   self.SiteName,
            "acquisitions_processed": len([r for r in records if "detection" in r]),
            "uniform_field_excluded": len(
                [r for r in records
                 if r.get("detection", {}).get("uniform_field")]
            ),
            "valid_acquisitions":     len(
                [r for r in records
                 if "detection" in r
                 and not r.get("detection", {}).get("uniform_field")]
            ),
            "n_observed_months":      len(obs_months),
            "detection_acquisitions": len(detected),
            "detection_months":       len(det_months),
            "detections_above_tau":   len(above_tau),
            "detections_cfar_pass":   len(cfar_pass),
            "detection_rate_by_month": (
                len(det_months) / len(obs_months) if obs_months else 0
            ),
            "quantification_boundary": "KWB mine polygon (~21km × 4.2km)",
        }
        summary.update(
            self._AnnualisationFields(detected, det_months, obs_months)
        )

        # Climate TRACE annual totals
        ct_annual = {
            yr: sum(v for (y, m), v in ct_ch4.items() if y == yr)
            for yr in years
        }
        ct_annual = {yr: v for yr, v in ct_annual.items() if v > 0}
        summary["ct_ch4_annual_t_by_year"] = ct_annual
        summary["ct_ch4_monthly_reference"] = {
            f"{y}-{m:02d}": round(v, 2)
            for (y, m), v in sorted(ct_ch4.items())
            if y in years
        }
        return summary

    def PrintSummary(self, store: dict) -> None:
        s = store.get("summary", {})
        years     = s.get("years", [])
        years_str = (
            "–".join(str(y) for y in (min(years), max(years)))
            if len(years) > 1 else str(years[0]) if years else "?"
        )
        print("\n" + "=" * 72)
        print(
            f"Bełchatów {years_str} time series  "
            f"(quant crop: KWB mine polygon ~21km × 4.2km)"
        )
        print("=" * 72)
        print(f"Acquisitions processed:      {s.get('acquisitions_processed', '?')}")
        print(f"Uniform-field excluded:      {s.get('uniform_field_excluded', '?')}")
        print(f"Valid acquisitions:          {s.get('valid_acquisitions', '?')}")
        print(f"Observed months:             {s.get('n_observed_months', '?')}")
        print(f"Detection acquisitions:      {s.get('detection_acquisitions', '?')}  (S/C > 1.15)")
        print(f"Detection months:            {s.get('detection_months', '?')}")
        print(f"Detections above conformal:  {s.get('detections_above_tau', '?')}  "
              f"(τ={self.ConformalTau})")
        print(f"Detections CFAR-confirmed:   {s.get('detections_cfar_pass', '?')}")
        if s.get("mean_flow_kg_h") is not None:
            n_det  = s.get('n_detection_months', '?')
            n_ndet = s.get('n_non_detection_months', '?')
            n_obs  = s.get('n_observed_months', '?')
            det_rt = s.get('detection_rate_by_month')
            print(
                f"Det months / non-det / obs:  "
                f"{n_det} / {n_ndet} / {n_obs}"
            )
            if det_rt is not None:
                print(f"Detection rate (by month):   {det_rt*100:.1f}%")
            else:
                print(f"Detection rate (by month):   ?")
            print(f"Mean flow rate (detections): {s['mean_flow_kg_h']:.0f} kg/h")
            print(f"Range:                       {s['min_flow_kg_h']:.0f}–{s['max_flow_kg_h']:.0f} kg/h")
            print(f"Annualisation — three framings:")
            print(f"  Upper:                       {s['annualised_t_per_yr_upper']:.0f} t/yr")
            print(
                f"  Floor-imputed (Q={s['detection_floor_kgh_assumed']:.0f} kg/h):"
                f"    {s['annualised_t_per_yr_floor_imputed']:.0f} t/yr"
            )
            print(f"  Lower bound:                 {s['annualised_t_per_yr_lower_bound']:.0f} t/yr")
        for yr, total in sorted(s.get("ct_ch4_annual_t_by_year", {}).items()):
            print(
                f"Climate TRACE {yr} annual CH4:      {total:,.1f} t/yr  "
                f"(asset 16168, sum of monthly)"
            )
        print(f"Output: {self.OutJson}")
        print("=" * 72)


# ── KWK ROW Ruch Chwałowice (T34UCA) ─────────────────────────────────────────

class RybnikChwalowiceTimeseriesBuilder(BaseTimeseriesBuilder):
    """Annual CH4Net + CEMF+IME time series for KWK ROW Ruch Chwałowice.

    Site: Carbon Mapper confirmed source pin, 50.0781°N 18.5451°E.
    External references: Carbon Mapper detections CSV + TROPOMI positives JSON.
    Output: ``results_analysis/rybnik_chwalowice_annual_timeseries.json``
    """

    @property
    def SiteName(self) -> str:
        return "rybnik_chwalowice"

    @property
    def TileId(self) -> str:
        return "T34UCA"

    @property
    def SiteLat(self) -> float:
        return 50.0781

    @property
    def SiteLon(self) -> float:
        return 18.5451

    @property
    def MinePolygon(self) -> list[tuple[float, float]]:
        """Approximate concession boundary of KWK ROW Ruch Chwałowice (~7.5 km × 3.6 km)."""
        return [
            (50.092, 18.508),   # NW
            (50.092, 18.580),   # NE
            (50.060, 18.580),   # SE
            (50.060, 18.508),   # SW
        ]

    @property
    def DetectionThreshold(self) -> float:
        return 1.15

    @property
    def ConformalTau(self) -> float:
        return 3.5796

    @property
    def ScOffsets(self) -> tuple[float, float, float, float]:
        """Symmetric 0.20° offsets — mine is ~7.5 km × 3.6 km."""
        return (0.20, 0.20, 0.20, 0.20)

    @property
    def DownloadDir(self) -> Path:
        return ROOT / "data" / "downloads" / "rybnik_chwalowice_annual"

    @property
    def OutJson(self) -> Path:
        return ROOT / "results_analysis" / "rybnik_chwalowice_annual_timeseries.json"

    _CM_CSV      = ROOT / "data" / "rybnik_chwalowice_carbon_mapper.csv"
    _TROPOMI_JSON = ROOT / "results_analysis" / "tropomi_positives.json"
    _TROPOMI_SITE = "silesia_rybnik"

    def _LoadExternalReferences(self) -> dict:
        """Load Carbon Mapper CSV and TROPOMI JSON.

        Returns
        -------
        dict with keys ``"cm_detections"`` and ``"tropomi_events"``, each a
        ``{(year, month): [event_dict, ...]}`` mapping.
        """
        cm_by_month:      dict = {}
        tropomi_by_month: dict = {}

        # Carbon Mapper
        if not self._CM_CSV.exists():
            self._log.warning(
                "Carbon Mapper CSV not found at %s — CM comparison disabled",
                self._CM_CSV,
            )
        else:
            with open(self._CM_CSV, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    dt_str = row.get("datetime", "").strip()
                    if not dt_str:
                        continue
                    try:
                        dt = datetime.fromisoformat(
                            dt_str.replace("+00", "+00:00")
                        )
                    except ValueError:
                        self._log.warning("Could not parse CM datetime: %s", dt_str)
                        continue
                    key = (dt.year, dt.month)
                    em_str = row.get("emission_auto", "").strip()
                    ws_str = row.get("wind_speed_avg_auto", "").strip()
                    wd_str = row.get("wind_direction_avg_auto", "").strip()
                    cm_by_month.setdefault(key, []).append({
                        "plume_id":      row.get("plume_id", "").strip(),
                        "date":          dt.strftime("%Y-%m-%d"),
                        "time_utc":      dt.strftime("%H:%M"),
                        "instrument":    row.get("instrument", "").strip(),
                        "platform":      row.get("platform", "").strip(),
                        "mission_phase": row.get("mission_phase", "").strip(),
                        "emission_kgh":  float(em_str) if em_str else None,
                        "wind_speed_ms": float(ws_str) if ws_str else None,
                        "wind_dir_deg":  float(wd_str) if wd_str else None,
                    })
            total_cm = sum(len(v) for v in cm_by_month.values())
            self._log.info(
                "Loaded %d Carbon Mapper detections across %d months",
                total_cm, len(cm_by_month),
            )

        # TROPOMI
        if not self._TROPOMI_JSON.exists():
            self._log.warning(
                "TROPOMI JSON not found at %s — TROPOMI tagging disabled",
                self._TROPOMI_JSON,
            )
        else:
            with open(self._TROPOMI_JSON) as f:
                events = json.load(f)
            for e in events:
                if e.get("site") != self._TROPOMI_SITE:
                    continue
                date_str = e.get("date", "")
                if not date_str:
                    continue
                try:
                    year_s, month_s, _ = date_str.split("-")
                    key = (int(year_s), int(month_s))
                except ValueError:
                    continue
                tropomi_by_month.setdefault(key, []).append({
                    "date":            date_str,
                    "enhancement_ppb": e.get("enhancement_ppb"),
                    "n_near_pixels":   e.get("n_near_pixels"),
                    "validated":       e.get("validated", False),
                    "product_name":    e.get("product_name", ""),
                })
            total_trop = sum(len(v) for v in tropomi_by_month.values())
            self._log.info(
                "Loaded %d TROPOMI events across %d months (site: %s)",
                total_trop, len(tropomi_by_month), self._TROPOMI_SITE,
            )

        return {"cm_detections": cm_by_month, "tropomi_events": tropomi_by_month}

    def _AnnotateRecord(
        self, rec: dict, year: int, month: int, refs: dict
    ) -> None:
        rec["cm_detections"]  = refs["cm_detections"].get((year, month), [])
        rec["tropomi_events"] = refs["tropomi_events"].get((year, month), [])

    def _BuildSummary(
        self,
        store:        dict,
        records:      list[dict],
        years:        list[int],
        obs_months:   list[str],
        det_months:   list[str],
        detected:     list[dict],
        above_tau:    list[dict],
        cfar_pass:    list[dict],
        refs:         dict,
    ) -> dict:
        cm_dets     = refs.get("cm_detections", {})
        trop_events = refs.get("tropomi_events", {})

        cm_confirmed_months = {
            f"{y}-{m:02d}" for (y, m) in cm_dets if y in years
        }
        co_detected_months = sorted(set(det_months) & cm_confirmed_months)

        valid_acq = [
            r for r in records
            if "detection" in r
            and not r.get("detection", {}).get("uniform_field")
        ]

        summary: dict = {
            "years":                   years,
            "site":                    self.SiteName,
            "lat":                     self.SiteLat,
            "lon":                     self.SiteLon,
            "tile":                    self.TileId,
            "acquisitions_processed":  len([r for r in records if "detection" in r]),
            "uniform_field_excluded":  len(
                [r for r in records
                 if r.get("detection", {}).get("uniform_field")]
            ),
            "valid_acquisitions":      len(valid_acq),
            "n_observed_months":       len(obs_months),
            "detection_acquisitions":  len(detected),
            "detection_months":        len(det_months),
            "detections_above_tau":    len(above_tau),
            "detections_cfar_pass":    len(cfar_pass),
            "detection_rate_by_month": (
                len(det_months) / len(obs_months) if obs_months else 0
            ),
            "quantification_boundary": "KWK Chwałowice polygon (~7.5km × 3.6km)",
            "cm_co_detected_months":   co_detected_months,
            "n_cm_co_detected_months": len(co_detected_months),
        }
        summary.update(
            self._AnnualisationFields(detected, det_months, obs_months)
        )

        # Carbon Mapper reference
        summary["cm_detections_by_month"] = {
            f"{y}-{m:02d}": [{k: v for k, v in d.items()} for d in dets]
            for (y, m), dets in sorted(cm_dets.items())
            if y in years
        }

        # TROPOMI reference
        tropomi_ref = {
            f"{y}-{m:02d}": [{k: v for k, v in e.items()} for e in evts]
            for (y, m), evts in sorted(trop_events.items())
            if y in years
        }
        tropomi_validated = {
            k: [e for e in v if e.get("validated")]
            for k, v in tropomi_ref.items()
        }
        tropomi_validated = {k: v for k, v in tropomi_validated.items() if v}
        all_ppb = [
            e["enhancement_ppb"]
            for evts in tropomi_validated.values()
            for e in evts
            if e.get("enhancement_ppb") is not None
        ]
        summary["tropomi_events_by_month"]       = tropomi_ref
        summary["tropomi_validated_months"]       = sorted(tropomi_validated.keys())
        summary["n_tropomi_validated_months"]     = len(tropomi_validated)
        summary["tropomi_mean_enhancement_ppb"]   = (
            float(np.mean(all_ppb)) if all_ppb else None
        )
        return summary

    def PrintSummary(self, store: dict) -> None:
        s = store.get("summary", {})
        years     = s.get("years", [])
        years_str = (
            "–".join(str(y) for y in (min(years), max(years)))
            if len(years) > 1 else str(years[0]) if years else "?"
        )
        print("\n" + "=" * 72)
        print(
            f"KWK Chwałowice ({self.SiteName}) {years_str}  "
            f"(quant crop: {s.get('quantification_boundary', '?')})"
        )
        print("=" * 72)
        print(f"Acquisitions processed:      {s.get('acquisitions_processed', '?')}")
        print(f"Uniform-field excluded:      {s.get('uniform_field_excluded', '?')}")
        print(f"Valid acquisitions:          {s.get('valid_acquisitions', '?')}")
        print(f"Observed months:             {s.get('n_observed_months', '?')}")
        print(f"Detection acquisitions:      {s.get('detection_acquisitions', '?')}  (S/C > 1.15)")
        print(f"Detection months:            {s.get('detection_months', '?')}")
        print(f"Detections above conformal:  {s.get('detections_above_tau', '?')}  "
              f"(τ={self.ConformalTau})")
        print(f"CFAR-confirmed detections:   {s.get('detections_cfar_pass', '?')}")
        print(
            f"CM co-detected months:       {s.get('n_cm_co_detected_months', '?')}  "
            f"{s.get('cm_co_detected_months', [])}"
        )
        if s.get("mean_flow_kg_h") is not None:
            n_det  = s.get('n_detection_months', '?')
            n_ndet = s.get('n_non_detection_months', '?')
            n_obs  = s.get('n_observed_months', '?')
            det_rt = s.get('detection_rate_by_month')
            print(
                f"Det months / non-det / obs:  "
                f"{n_det} / {n_ndet} / {n_obs}"
            )
            if det_rt is not None:
                print(f"Detection rate (by month):   {det_rt*100:.1f}%")
            else:
                print(f"Detection rate (by month):   ?")
            print(f"Mean flow rate (detections): {s['mean_flow_kg_h']:.0f} kg/h")
            print(f"CM range (Tanager+EMIT):     1,150–2,019 kg/h  (6 detections)")
            print(f"Annualisation — three framings:")
            print(f"  Upper:                       {s['annualised_t_per_yr_upper']:.0f} t/yr")
            print(
                f"  Floor-imputed (Q={s['detection_floor_kgh_assumed']:.0f} kg/h):"
                f"    {s['annualised_t_per_yr_floor_imputed']:.0f} t/yr"
            )
            print(f"  Lower bound:                 {s['annualised_t_per_yr_lower_bound']:.0f} t/yr")
        cm_ref = s.get("cm_detections_by_month", {})
        if cm_ref:
            print("\nCarbon Mapper detections in run years:")
            for mkey, dets in sorted(cm_ref.items()):
                for d in dets:
                    em = (
                        f"{d['emission_kgh']:.0f} kg/h"
                        if d.get("emission_kgh") else "(no auto-emission)"
                    )
                    print(
                        f"  {d['date']}  {d.get('instrument', '').upper():5s}  {em}"
                    )
        trop_val = {
            k: v for k, v in s.get("tropomi_events_by_month", {}).items()
            if any(e.get("validated") for e in v)
        }
        if trop_val:
            print(
                f"\nTROPOMI validated events ({s.get('n_tropomi_validated_months', 0)} months, "
                f"mean enhancement {s.get('tropomi_mean_enhancement_ppb') or 0:.1f} ppb):"
            )
            for mkey, evts in sorted(trop_val.items()):
                for e in [x for x in evts if x.get("validated")]:
                    print(
                        f"  {e['date']}  +{e.get('enhancement_ppb', 0):.1f} ppb  "
                        f"({e.get('n_near_pixels', '?')} px)"
                    )
        print(f"\nOutput: {self.OutJson}")
        print("=" * 72)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annual CH4Net + CEMF+IME time series builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--site",
        choices=["belchatow", "rybnik_chwalowice"],
        required=True,
        help="Which site to run",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_YEARS,
        help="Years to process",
    )
    parser.add_argument(
        "--months", nargs="+", type=int, default=DEFAULT_MONTHS,
        help="Months to process per year",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=DEFAULT_MAX_CLOUD,
        help="CDSE cloud-cover ceiling (%%)",
    )
    parser.add_argument(
        "--max-cloud-fallback", type=float, default=DEFAULT_MAX_CLOUD_FALLBACK,
        help="Retry ceiling if first search returns no products",
    )
    parser.add_argument(
        "--max-acq", type=int, default=DEFAULT_MAX_CANDIDATES,
        help="Max acquisitions per month (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Search catalogue only; no downloads or inference",
    )
    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    builder: BaseTimeseriesBuilder
    if args.site == "belchatow":
        builder = BelchatowTimeseriesBuilder()
    else:
        builder = RybnikChwalowiceTimeseriesBuilder()

    store = builder.Run(
        years=args.years,
        months=args.months,
        max_cloud=args.max_cloud,
        max_cloud_fallback=args.max_cloud_fallback,
        dry_run=args.dry_run,
        max_candidates=args.max_acq,
    )
    builder.PrintSummary(store)
