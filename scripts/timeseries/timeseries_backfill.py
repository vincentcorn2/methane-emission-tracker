"""
scripts/timeseries/timeseries_backfill.py
==========================================
OOP refactor of the WS4 historical backfill pipeline.

Consolidates:
    historical_backfill_download.py    (WS4 step 1: download 2020–2023 tiles)
    historical_backfill_timeseries.py  (WS4 step 2: inference + S/C time series)
    repair_backfill_coverage.py        (WS4 step 3: flag partial-swath no-coverage tiles)

Class hierarchy::

    BackfillDownloader         – Search CDSE and download 2020-2023 summer tiles
    BackfillTimeseriesBuilder  – Run CH4Net inference and compile S/C time series
    BackfillCoverageRepairer   – Detect and repair degenerate partial-swath records

Typical pipeline::

    from scripts.timeseries.timeseries_backfill import (
        BackfillDownloader, BackfillTimeseriesBuilder, BackfillCoverageRepairer
    )
    manifest = BackfillDownloader().Run()
    timeseries = BackfillTimeseriesBuilder().Run()
    BackfillCoverageRepairer().Run()

All three classes write JSON to ``results_analysis/`` and support incremental
resume: re-running skips already-completed (site, year) pairs or tiles.

**Requires:** CDSE credentials (env: COPERNICUS_USER / COPERNICUS_PASS),
CH4Net v8 weights, and ``data/npy_cache/`` populated by step 1.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── Path setup (must precede local imports) ───────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # methane-api/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "detection"))

# ── Shared constants ──────────────────────────────────────────────────────────
NPY_CACHE    = ROOT / "data" / "npy_cache"
DOWNLOAD_DIR = ROOT / "data" / "downloads" / "historical"
MANIFEST_OUT = ROOT / "results_analysis" / "historical_backfill_manifest.json"
TIMESERIES_OUT = ROOT / "results_analysis" / "historical_backfill_timeseries.json"

DEFAULT_MAX_CLOUD  = 20.0
FALLBACK_MAX_CLOUD = 35.0
MIN_YEAR           = 2020
MAX_YEAR           = 2023    # 2024+ already in npy_cache

# Detection criteria
CLASSIC_THRESH    = 1.15
MIN_SITE_VALID_FRAC = 0.50   # fraction of non-zero pixels required in site crop

# SC fields nulled when a tile is flagged as no-coverage
SC_FIELDS = [
    "sc_ratio", "sc_cfar", "site_mean", "ctrl_mean", "ctrl_mu",
    "cv_ctrl", "cfar_thresh_ratio", "cfar_detect", "cfar_margin", "ctrl_n",
]

# Acquisition-date regex (matches S2A/B_MSIL1C_YYYYMMDD T...)
_ACQ_DATE_RE = re.compile(r"S2[AB]_MSIL1C_(\d{8})T\d{6}_")
_ACQ_YEAR_RE = re.compile(r"S2[AB]_MSIL1C_(\d{4})\d{4}T")

# ── Site catalogue (8 priority emitters + reference sites) ────────────────────
# Used by both BackfillDownloader and BackfillTimeseriesBuilder.
# skip_bitemporal=True  → classic S/C > 1.15 detection criterion
# skip_bitemporal=False → CFAR adaptive threshold
BACKFILL_SITES: dict[str, dict] = {
    "weisweiler":  dict(lat=50.837, lon=6.322,  tile_id="T31UGS", skip_bitemporal=True),
    "rybnik":      dict(lat=50.135, lon=18.522, tile_id="T34UCA", skip_bitemporal=True),
    "belchatow":   dict(lat=51.242, lon=19.275, tile_id="T34UCB", skip_bitemporal=True),
    "lippendorf":  dict(lat=51.178, lon=12.378, tile_id="T33UUS", skip_bitemporal=True),
    "neurath":     dict(lat=51.038, lon=6.616,  tile_id="T32ULB", skip_bitemporal=True),
    "boxberg":     dict(lat=51.416, lon=14.565, tile_id="T33UVT", skip_bitemporal=True),
    "groningen":   dict(lat=53.252, lon=6.682,  tile_id="T31UGV", skip_bitemporal=False),
    "maasvlakte":  dict(lat=51.944, lon=4.067,  tile_id="T31UET", skip_bitemporal=True),
}


# ── BackfillDownloader ────────────────────────────────────────────────────────

class BackfillDownloader:
    """Search CDSE for summer 2020–2023 S2 L1C tiles and download to npy_cache.

    Selection criterion: lowest-cloud-cover scene in months June–August per
    (site, year).  After downloading, run
    :class:`BackfillTimeseriesBuilder` to compile the S/C time series.

    Parameters
    ----------
    log_to_file : bool
        If True, append a log file to ``results_analysis/``.
    """

    def __init__(self, log_to_file: bool = True) -> None:
        Path("results_analysis").mkdir(exist_ok=True)
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_to_file:
            handlers.append(
                logging.FileHandler(
                    str(ROOT / "results_analysis" / "historical_backfill_download.log")
                )
            )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
        )
        self._log = logging.getLogger("backfill_download")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _SummerWindow(year: int) -> tuple[str, str]:
        """Return (start_iso, end_iso) for the Jun–Aug window of ``year``."""
        return (
            f"{year}-06-01T00:00:00.000Z",
            f"{year}-08-31T23:59:59.999Z",
        )

    @staticmethod
    def _BboxWkt(lat: float, lon: float, margin: float = 0.25) -> str:
        return (
            f"POLYGON(({lon - margin} {lat - margin},{lon + margin} {lat - margin},"
            f"{lon + margin} {lat + margin},{lon - margin} {lat + margin},"
            f"{lon - margin} {lat - margin}))"
        )

    @staticmethod
    def _TileAlreadyCached(tile_id: str, year: int) -> list[Path]:
        """Return .npy files already in npy_cache for this tile and acquisition year.

        Matches on the acquisition year in the S2 filename, not the processing year,
        to avoid false positives from the processing-date field.
        """
        year_str = str(year)
        result = []
        for p in NPY_CACHE.glob(f"*{tile_id}*.npy"):
            if "MSIL1C" not in p.name or "_ref_" in p.name:
                continue
            m = _ACQ_YEAR_RE.search(p.name)
            if m and m.group(1) == year_str:
                result.append(p)
        return sorted(result)

    def _SearchSiteYear(
        self,
        client,
        site:      str,
        cfg:       dict,
        year:      int,
        max_cloud: float,
    ) -> list:
        """Return all L1C products found by CDSE for (site, year) summer."""
        start, end = self._SummerWindow(year)
        tile_id    = cfg["tile_id"]
        try:
            products = client.search_products(
                wkt_polygon=self._BboxWkt(cfg["lat"], cfg["lon"]),
                start_date=start,
                end_date=end,
                collection="SENTINEL-2",
                max_cloud_cover=max_cloud,
            )
        except Exception as exc:
            self._log.error("  Search failed for %s/%d: %s", site, year, exc)
            return []

        l1c = [
            p for p in products
            if tile_id in p.tile_id and "MSIL1C" in p.name
        ]
        self._log.info(
            "  %s %d: %d L1C scenes (of %d total) at cloud ≤ %.0f%%",
            site, year, len(l1c), len(products), max_cloud,
        )
        return l1c

    def _DownloadAndConvert(
        self,
        product,
        client,
        site:    str,
        tile_id: str,
    ) -> dict:
        """Download a product and convert to .npy in npy_cache.

        Returns
        -------
        dict with keys: status, npy, meta, cloud_cover, acquisition_date, product_name.

        Raises
        ------
        RuntimeError on download failure.
        """
        from src.ingestion.preprocessing import safe_to_npy  # type: ignore[import]

        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        NPY_CACHE.mkdir(parents=True, exist_ok=True)

        self._log.info("  Downloading %s ...", product.name[:70])
        zip_path = client.download_product(product, str(DOWNLOAD_DIR))
        if zip_path is None:
            raise RuntimeError(f"Download returned None for {product.name}")

        self._log.info("  Converting to .npy ...")
        extract_dir = tempfile.mkdtemp(prefix=f"s2_{site}_hist_")
        try:
            npy_path, meta_path = safe_to_npy(
                zip_path=zip_path,
                output_dir=str(NPY_CACHE),
                tile_id=tile_id,
                acquisition_date=product.acquisition_date,
                satellite=product.satellite,
                extract_dir=extract_dir,
            )
            self._log.info("  Saved: %s", Path(npy_path).name)
            return {
                "status":           "ok",
                "npy":              npy_path,
                "meta":             meta_path,
                "cloud_cover":      product.cloud_cover,
                "acquisition_date": product.acquisition_date,
                "product_name":     product.name,
            }
        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)

    def _ProcessSiteYear(
        self,
        site:      str,
        cfg:       dict,
        year:      int,
        client,
        max_cloud: float,
        dry_run:   bool,
    ) -> dict:
        """Handle one (site, year) pair.  Returns a result dict."""
        tile_id = cfg["tile_id"]
        self._log.info("── %s  %d  (%s) ──", site, year, tile_id)

        cached = self._TileAlreadyCached(tile_id, year)
        if cached:
            self._log.info("  Already cached: %s", cached[0].name)
            return {
                "site": site, "year": year, "tile_id": tile_id,
                "status": "cached", "npy": str(cached[0]),
            }

        products = self._SearchSiteYear(client, site, cfg, year, max_cloud)
        if not products and max_cloud < FALLBACK_MAX_CLOUD:
            self._log.warning(
                "  No L1C at %.0f%% — retrying at %.0f%%",
                max_cloud, FALLBACK_MAX_CLOUD,
            )
            products = self._SearchSiteYear(
                client, site, cfg, year, FALLBACK_MAX_CLOUD
            )

        if not products:
            self._log.warning("  No L1C found for %s in summer %d", site, year)
            return {
                "site": site, "year": year, "tile_id": tile_id,
                "status": "not_found",
            }

        products.sort(
            key=lambda p: (p.cloud_cover or 99, p.acquisition_date or "")
        )
        best = products[0]
        self._log.info(
            "  Best: %s  cloud=%.1f%%  date=%s",
            best.name[:60], best.cloud_cover or 0, best.acquisition_date or "?",
        )

        if dry_run:
            return {
                "site": site, "year": year, "tile_id": tile_id,
                "status": "dry_run",
                "would_download":    best.name,
                "cloud_cover":       best.cloud_cover,
                "acquisition_date":  best.acquisition_date,
            }

        existing = list(NPY_CACHE.glob(f"*{best.name[:44]}*.npy"))
        if existing:
            self._log.info("  Already in npy_cache: %s", existing[0].name)
            return {
                "site": site, "year": year, "tile_id": tile_id,
                "status": "cached", "npy": str(existing[0]),
                "cloud_cover":      best.cloud_cover,
                "acquisition_date": best.acquisition_date,
            }

        try:
            result = self._DownloadAndConvert(best, client, site, tile_id)
            result.update({"site": site, "year": year, "tile_id": tile_id})
            return result
        except Exception as exc:
            self._log.error("  Download failed: %s", exc)
            return {
                "site": site, "year": year, "tile_id": tile_id,
                "status": "error", "error": str(exc),
            }

    # ── Public API ────────────────────────────────────────────────────────────

    def Run(
        self,
        sites:     list[str] | None = None,
        years:     list[int] | None = None,
        max_cloud: float = DEFAULT_MAX_CLOUD,
        dry_run:   bool  = False,
    ) -> dict:
        """Download summer tiles for all (site, year) pairs and return manifest.

        Parameters
        ----------
        sites:     subset of BACKFILL_SITES keys (default: all)
        years:     calendar years to download (default: 2020–2023)
        max_cloud: primary cloud ceiling %; automatically falls back to 35%
        dry_run:   search catalogue only; no downloads

        Returns
        -------
        dict
            Manifest keyed by ``"{site}_{year}"``, each value a result dict.
        """
        from src.ingestion.copernicus_client import CopernicusClient  # type: ignore[import]

        sites_to_run = {
            k: v for k, v in BACKFILL_SITES.items()
            if sites is None or k in sites
        }
        years_to_run = years or list(range(MIN_YEAR, MAX_YEAR + 1))
        total_pairs  = len(sites_to_run) * len(years_to_run)

        print("=" * 72)
        print("  WS4 Historical Backfill — S2 L1C Download")
        print(f"  Sites: {list(sites_to_run.keys())}")
        print(f"  Years: {years_to_run}  (2024+ already cached)")
        print(f"  Cloud limit: {max_cloud:.0f}%   Fallback: {FALLBACK_MAX_CLOUD:.0f}%")
        print(f"  Total (site, year) pairs: {total_pairs}")
        if dry_run:
            print("  MODE: DRY RUN")
        print("=" * 72)

        # Authenticate
        username = os.environ.get("COPERNICUS_USER", "").strip()
        password = os.environ.get("COPERNICUS_PASS", "").strip()
        if not username:
            username = input("\nCopernicus username (email): ").strip()
        if not password:
            password = getpass.getpass("Copernicus password: ")

        client = CopernicusClient(username, password)
        _ = client.token

        # Load existing manifest (incremental resume)
        manifest: dict = {}
        if MANIFEST_OUT.exists():
            try:
                manifest = json.loads(MANIFEST_OUT.read_text())
                already = sum(
                    1 for r in manifest.values()
                    if r.get("status") in ("ok", "cached")
                )
                self._log.info(
                    "Resuming: %d (site, year) pairs already done", already
                )
            except Exception:
                pass

        for site, cfg in sites_to_run.items():
            for year in years_to_run:
                key = f"{site}_{year}"
                if key in manifest and manifest[key].get("status") in ("ok", "cached"):
                    self._log.info("Skip %s — already done", key)
                    continue
                result = self._ProcessSiteYear(
                    site, cfg, year, client, max_cloud, dry_run
                )
                manifest[key] = result
                if not dry_run:
                    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
                    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))

        self.PrintSummary(manifest, total_pairs)
        return manifest

    def PrintSummary(self, manifest: dict, total_pairs: int | None = None) -> None:
        """Print a download summary table."""
        print("\n" + "=" * 72)
        print("  Download Summary:")
        counts: dict[str, int] = {}
        for key, r in sorted(manifest.items()):
            status = r.get("status", "?")
            counts[status] = counts.get(status, 0) + 1
            icon = {
                "ok": "✓", "cached": "✓", "dry_run": "○",
                "not_found": "—", "error": "✗",
            }.get(status, "?")
            cloud = r.get("cloud_cover")
            npy_name = (
                Path(r["npy"]).name[:45]
                if r.get("npy") else r.get("would_download", "")[:45]
            )
            cloud_str = f"  cloud={cloud:.1f}%" if cloud is not None else ""
            print(
                f"  {icon} {r.get('site', key.split('_')[0]):<14} "
                f"{r.get('year', '')}  {status:<12}  {npy_name}{cloud_str}"
            )
        print(f"\n  Totals: {counts}")
        n_ready = sum(
            1 for r in manifest.values() if r.get("status") in ("ok", "cached")
        )
        total = total_pairs or len(manifest)
        print(f"\n  {n_ready}/{total} (site, year) pairs ready for inference.\n")

    def SaveManifest(self, manifest: dict) -> None:
        """Write manifest JSON to ``MANIFEST_OUT``."""
        MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
        self._log.info("Manifest saved → %s", MANIFEST_OUT)


# ── BackfillTimeseriesBuilder ─────────────────────────────────────────────────

class BackfillTimeseriesBuilder:
    """Compile multi-year S/C time series from npy_cache for 8 priority sites.

    For each .npy tile in npy_cache matching a site's MGRS tile_id:
      1. Run (or cache) CH4Net inference → ``original_<stem>.tif``
      2. Check site coverage (guard against partial-swath tiles)
      3. Call ``compute_sc_ratio`` to extract S/C, CFAR fields
      4. Append a structured record to the time series

    Incremental: re-running merges new tiles into the existing JSON.

    Parameters
    ----------
    log_to_file : bool
        If True, append a log file to ``results_analysis/``.
    """

    def __init__(self, log_to_file: bool = True) -> None:
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
        if log_to_file:
            Path("results_analysis").mkdir(exist_ok=True)
            handlers.append(
                logging.FileHandler(
                    str(ROOT / "results_analysis" / "historical_backfill_timeseries.log")
                )
            )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=handlers,
        )
        self._log = logging.getLogger("backfill_timeseries")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ParseAcquisitionDate(stem: str) -> str | None:
        """Extract ISO date (YYYY-MM-DD) from S2 product filename stem."""
        m = _ACQ_DATE_RE.search(stem)
        if not m:
            return None
        d = m.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    @staticmethod
    def _FindAllTargetNpys(tile_id: str) -> list[Path]:
        """Return all non-reference S2 .npy tiles for a given MGRS tile_id."""
        candidates = [
            p for p in NPY_CACHE.glob(f"*_{tile_id}_*.npy")
            if "_ref_" not in p.name and "_bitemporal" not in p.name
        ]
        return sorted(candidates, key=lambda p: p.name)

    @staticmethod
    def _CheckSiteCoverage(
        npy_path:  Path,
        tif_path:  Path,
        lat:       float,
        lon:       float,
        crop_px:   int = 100,
    ) -> float:
        """Return fraction of non-zero pixels in the site crop of the NPY.

        A value below ``MIN_SITE_VALID_FRAC`` indicates the plant falls
        outside the Sentinel-2 swath boundary for this pass (partial-swath
        tile).  Inference on such tiles produces a degenerate sc_ratio=1.0.
        """
        import rasterio
        from pyproj import Transformer  # type: ignore[import]

        arr = np.load(npy_path)   # (H, W, 12) uint8
        with rasterio.open(tif_path) as src:
            epsg = src.crs.to_epsg()
            transformer = Transformer.from_crs(
                "EPSG:4326", f"EPSG:{epsg}", always_xy=True
            )
            xs, ys = transformer.transform([lon], [lat])
            s_row, s_col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])

        half = crop_px // 2
        H, W = arr.shape[:2]
        r0 = max(0, int(s_row) - half)
        r1 = min(H, int(s_row) + half)
        c0 = max(0, int(s_col) - half)
        c1 = min(W, int(s_col) + half)
        patch = arr[r0:r1, c0:c1, :]
        if patch.size == 0:
            return 0.0
        return float(patch.any(axis=-1).mean())

    def _GetOrRunInference(
        self,
        npy_path:  Path,
        site_name: str,
        detector,
        dry_run:   bool,
    ) -> Path | None:
        """Return path to the inference .tif; create it if absent and possible."""
        from apply_bitemporal_diff import (  # type: ignore[import]
            OUT_DIR, find_geo_meta, run_inference,
        )

        site_dir = OUT_DIR / site_name
        out_tif  = site_dir / f"original_{npy_path.stem}.tif"

        if out_tif.exists():
            self._log.info("    [hit]  %s", out_tif.name)
            return out_tif

        if dry_run or detector is None:
            self._log.info(
                "    [miss] %s — skipping (no-inference / dry-run)", npy_path.name
            )
            return None

        geo_meta = find_geo_meta(npy_path)
        if geo_meta is None:
            self._log.warning(
                "    No geo metadata for %s — cannot run inference", npy_path.name
            )
            return None

        site_dir.mkdir(parents=True, exist_ok=True)
        self._log.info("    [run]  Running inference on %s ...", npy_path.name)
        try:
            arr = np.load(npy_path)
            self._log.info(
                "           shape=%s  dtype=%s", arr.shape, arr.dtype
            )
            run_inference(arr, detector, geo_meta, out_tif)
            del arr
            self._log.info("           Saved: %s", out_tif.name)
            return out_tif
        except Exception as exc:
            self._log.error(
                "    Inference failed for %s: %s", npy_path.name, exc
            )
            return None

    def _ProcessSite(
        self,
        site_name:    str,
        cfg:          dict,
        detector,
        years_filter: list[int] | None,
        dry_run:      bool,
    ) -> list[dict]:
        """Process all tiles for one site; return sorted per-tile records."""
        from apply_bitemporal_diff import (  # type: ignore[import]
            OUT_DIR, compute_sc_ratio,
        )

        tile_id = cfg["tile_id"]
        lat, lon = cfg["lat"], cfg["lon"]

        npys = self._FindAllTargetNpys(tile_id)
        if not npys:
            self._log.warning(
                "  No .npy tiles found for %s (%s)", site_name, tile_id
            )
            return []

        self._log.info(
            "  %s (%s): %d tiles in cache", site_name, tile_id, len(npys)
        )
        records: list[dict] = []

        for npy_path in npys:
            acq_date = self._ParseAcquisitionDate(npy_path.stem)
            year     = int(acq_date[:4]) if acq_date else None

            if years_filter and year not in years_filter:
                continue

            self._log.info(
                "  ── %s  date=%s", npy_path.name[:70], acq_date or "?"
            )

            # Coverage pre-check: skip partial-swath tiles
            site_dir     = OUT_DIR / site_name
            candidate_tif = site_dir / f"original_{npy_path.stem}.tif"
            if candidate_tif.exists():
                try:
                    vf = self._CheckSiteCoverage(npy_path, candidate_tif, lat, lon)
                    if vf < MIN_SITE_VALID_FRAC:
                        self._log.warning(
                            "    [skip]  %s  valid_frac=%.2f < %.2f "
                            "— plant outside swath",
                            npy_path.name, vf, MIN_SITE_VALID_FRAC,
                        )
                        records.append({
                            "site":             site_name,
                            "tile_id":          tile_id,
                            "acquisition_date": acq_date,
                            "year":             year,
                            "npy":              npy_path.name,
                            "tif":              str(candidate_tif),
                            "status":           "no_coverage",
                            "valid_fraction":   round(vf, 4),
                            "coverage_note":    (
                                f"site crop {vf*100:.0f}% valid — "
                                "partial-swath tile, plant outside S2 swath"
                            ),
                        })
                        continue
                except Exception as exc:
                    self._log.warning("    Coverage check failed: %s", exc)

            tif_path = self._GetOrRunInference(
                npy_path, site_name, detector, dry_run
            )
            if tif_path is None:
                records.append({
                    "site":             site_name,
                    "tile_id":          tile_id,
                    "acquisition_date": acq_date,
                    "year":             year,
                    "npy":              npy_path.name,
                    "tif":              None,
                    "status":           "no_tif",
                })
                continue

            sc = compute_sc_ratio(tif_path, lat, lon)
            if sc.get("error"):
                self._log.warning("    SC error: %s", sc["error"])
                records.append({
                    "site":             site_name,
                    "tile_id":          tile_id,
                    "acquisition_date": acq_date,
                    "year":             year,
                    "npy":              npy_path.name,
                    "tif":              str(tif_path),
                    "status":           "sc_error",
                    "error":            sc["error"],
                })
                continue

            records.append({
                "site":              site_name,
                "tile_id":           tile_id,
                "acquisition_date":  acq_date,
                "year":              year,
                "npy":               npy_path.name,
                "tif":               str(tif_path),
                "status":            "ok",
                "sc_ratio":          sc.get("sc_ratio"),
                "sc_cfar":           sc.get("sc_cfar"),
                "site_mean":         sc.get("site_mean"),
                "ctrl_mean":         sc.get("ctrl_mean"),
                "ctrl_mu":           sc.get("ctrl_mu"),
                "cv_ctrl":           sc.get("cv_ctrl"),
                "cfar_thresh_ratio": sc.get("cfar_thresh_ratio"),
                "cfar_detect":       sc.get("cfar_detect"),
                "cfar_margin":       sc.get("cfar_margin"),
                "ctrl_n":            sc.get("ctrl_n"),
            })

            r_val  = sc.get("sc_ratio")
            sc_c   = sc.get("sc_cfar")
            thr    = sc.get("cfar_thresh_ratio")
            det    = sc.get("cfar_detect")
            self._log.info(
                "    S/C=%-8s  sc_cfar=%-8s  thr_ratio=%-6s  CFAR=%s",
                f"{r_val:.4f}" if r_val is not None else "—",
                f"{sc_c:.4f}"  if sc_c  is not None else "—",
                f"{thr:.3f}"   if thr   is not None else "—",
                "DETECT" if det else "no",
            )

        return sorted(records, key=lambda r: r.get("acquisition_date") or "")

    @staticmethod
    def IsDetection(record: dict, skip_bitemporal: bool) -> bool:
        """True if ``record`` counts as a detection under the correct criterion.

        Parameters
        ----------
        skip_bitemporal:
            True → industrial emitter (classic S/C > 1.15)
            False → heterogeneous-background site (CFAR)
        """
        sc = record.get("sc_ratio")
        if sc is None:
            return False
        if skip_bitemporal:
            return sc > CLASSIC_THRESH
        return bool(record.get("cfar_detect"))

    # ── Public API ────────────────────────────────────────────────────────────

    def Run(
        self,
        sites:        list[str] | None = None,
        years:        list[int] | None = None,
        no_inference: bool = False,
        weights:      str  | None = None,
    ) -> dict:
        """Compile the multi-year S/C time series and return it.

        Parameters
        ----------
        sites:        subset of BACKFILL_SITES keys (default: all)
        years:        filter to specific acquisition years (default: all in cache)
        no_inference: skip CH4Net inference; only process tiles with existing .tifs
        weights:      path to CH4Net weights file (default: from apply_bitemporal_diff)

        Returns
        -------
        dict
            ``{site_name: [record, ...], ...}`` sorted by acquisition date.
        """
        from apply_bitemporal_diff import (  # type: ignore[import]
            CH4NetDetector, THRESHOLD, WEIGHTS as DEFAULT_WEIGHTS,
        )

        weights       = weights or DEFAULT_WEIGHTS
        sites_to_run  = {
            k: v for k, v in BACKFILL_SITES.items()
            if sites is None or k in sites
        }
        years_filter  = years

        print("=" * 72)
        print("  WS4 Historical Backfill — Time Series Compilation")
        print(f"  Sites: {list(sites_to_run.keys())}")
        if years_filter:
            print(f"  Years filter: {years_filter}")
        if no_inference:
            print("  MODE: NO-INFERENCE (only existing tifs)")
        print("=" * 72)

        # Load model
        detector = None
        if not no_inference:
            if not Path(weights).exists():
                raise FileNotFoundError(
                    f"CH4Net weights not found: {weights}\n"
                    "Run with no_inference=True to skip inference."
                )
            detector = CH4NetDetector(weights, threshold=THRESHOLD)
            self._log.info("CH4Net loaded (device: %s)", detector.device)
        else:
            self._log.info("Skipping model load (no_inference=True)")

        # Load existing timeseries (incremental resume)
        timeseries: dict = {}
        if TIMESERIES_OUT.exists():
            try:
                timeseries = json.loads(TIMESERIES_OUT.read_text())
                n_existing = sum(len(v) for v in timeseries.values())
                self._log.info(
                    "Loaded existing timeseries: %d records across %d sites",
                    n_existing, len(timeseries),
                )
            except Exception:
                timeseries = {}

        for site_name, cfg in sites_to_run.items():
            self._log.info(
                "\n── %s ──────────────────────────────────────────────",
                site_name.upper(),
            )
            new_records = self._ProcessSite(
                site_name, cfg, detector, years_filter, no_inference
            )

            if years_filter and site_name in timeseries:
                kept = [
                    r for r in timeseries[site_name]
                    if r.get("year") not in years_filter
                ]
                timeseries[site_name] = sorted(
                    kept + new_records,
                    key=lambda r: r.get("acquisition_date") or "",
                )
            else:
                timeseries[site_name] = new_records

            TIMESERIES_OUT.parent.mkdir(parents=True, exist_ok=True)
            TIMESERIES_OUT.write_text(json.dumps(timeseries, indent=2, default=str))

        self.PrintSummary(timeseries)
        return timeseries

    def PrintSummary(self, timeseries: dict) -> None:
        """Print a compact multi-year S/C ratio table to stdout."""
        years_all = sorted(
            {r["year"] for recs in timeseries.values()
             for r in recs if r.get("year")}
        )
        col_w = 9
        header_years = "  ".join(f"{y:>{col_w}}" for y in years_all)
        print("\n" + "=" * (20 + (col_w + 2) * len(years_all) + 24))
        print("  WS4 Historical Backfill — S/C Ratio by Site and Year")
        print(f"  {'Site':<18}  {header_years}  {'Detections'}")
        print("  " + "-" * (18 + (col_w + 2) * len(years_all) + 26))

        for site, records in sorted(timeseries.items()):
            cfg       = BACKFILL_SITES.get(site, {})
            skip_bt   = cfg.get("skip_bitemporal", True)
            by_year: dict[int, list] = {}
            for r in records:
                yr = r.get("year")
                if yr:
                    by_year.setdefault(yr, []).append(r)

            cells: list[str] = []
            detect_years: list[str] = []
            for yr in years_all:
                recs = by_year.get(yr, [])
                ok   = [r for r in recs
                        if r.get("status") == "ok"
                        and r.get("sc_ratio") is not None]
                if not ok:
                    cells.append(f"{'—':>{col_w}}")
                else:
                    best    = max(ok, key=lambda r: r.get("sc_ratio") or 0)
                    sc_val  = best["sc_ratio"]
                    any_det = any(self.IsDetection(r, skip_bt) for r in ok)
                    marker  = "✓" if any_det else " "
                    cells.append(f"{marker}{sc_val:>{col_w - 1}.3f}")
                    if any_det:
                        detect_years.append(str(yr))

            crit       = "S/C>1.15" if skip_bt else "CFAR"
            detect_str = ", ".join(detect_years) if detect_years else "none"
            print(
                f"  {site:<18}  {'  '.join(cells)}  {detect_str}  [{crit}]"
            )

        print("=" * (20 + (col_w + 2) * len(years_all) + 24))
        print("  ✓ = detection on any tile that year  |  value = max S/C ratio")
        print()

    def SaveTimeseries(self, timeseries: dict) -> None:
        """Write timeseries to ``TIMESERIES_OUT``."""
        TIMESERIES_OUT.parent.mkdir(parents=True, exist_ok=True)
        TIMESERIES_OUT.write_text(json.dumps(timeseries, indent=2, default=str))
        self._log.info("Time series saved → %s", TIMESERIES_OUT)


# ── BackfillCoverageRepairer ──────────────────────────────────────────────────

class BackfillCoverageRepairer:
    """Detect and repair partial-swath no-coverage records in the backfill JSON.

    Root cause: some tiles are partial-swath downloads where the plant falls
    entirely outside the actual S2 swath (0% valid pixels at the site crop).
    CH4Net inference on such tiles produces a degenerate probability map and
    sc_ratio=1.0 with site_mean == ctrl_mean — a false "no detection" rather
    than a properly flagged gap.

    This class:
      1. Loads ``historical_backfill_timeseries.json``.
      2. For every record whose site_mean == ctrl_mean (±1e-6) and sc_ratio == 1.0,
         loads the corresponding .npy and measures the valid-pixel fraction in the
         100×100 px site crop.
      3. If the fraction is below ``min_valid``, replaces the record's status with
         ``"no_coverage"`` and nulls all SC-derived fields.
      4. Writes the repaired JSON in-place (keeping a .bak copy).

    Parameters
    ----------
    min_valid : float
        Minimum valid-pixel fraction required to keep a record as ``"ok"``
        (default 0.50).
    """

    def __init__(self, min_valid: float = MIN_SITE_VALID_FRAC) -> None:
        self.min_valid = min_valid
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self._log = logging.getLogger("backfill_repair")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _LonlatToPixel(
        tif_path: Path, lon: float, lat: float
    ) -> tuple[int, int]:
        """Convert (lon, lat) → (row, col) in the raster coordinate system."""
        import rasterio
        from pyproj import Transformer  # type: ignore[import]

        with rasterio.open(tif_path) as src:
            epsg = src.crs.to_epsg()
            transformer = Transformer.from_crs(
                "EPSG:4326", f"EPSG:{epsg}", always_xy=True
            )
            xs, ys = transformer.transform([lon], [lat])
            row, col = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
        return int(row), int(col)

    @staticmethod
    def _SiteValidFraction(
        npy_path: Path,
        tif_path: Path,
        lon:      float,
        lat:      float,
        crop_px:  int = 100,
    ) -> float:
        """Return fraction of non-zero pixels in the 100×100 site crop."""
        arr = np.load(npy_path)   # (H, W, 12) uint8
        row, col = BackfillCoverageRepairer._LonlatToPixel(tif_path, lon, lat)
        half = crop_px // 2
        H, W = arr.shape[:2]
        r0   = max(0, row - half)
        r1   = min(H, row + half)
        c0   = max(0, col - half)
        c1   = min(W, col + half)
        patch = arr[r0:r1, c0:c1, :]
        if patch.size == 0:
            return 0.0
        return float(patch.any(axis=-1).mean())

    @staticmethod
    def _IsDegenerate(record: dict) -> bool:
        """True if the record shows the zero-coverage fingerprint.

        Two degenerate variants are detected:
        A) Classic: site_mean == ctrl_mean (6 dp), cv_ctrl == 0.0, sc_ratio == 1.0
        B) Mixed:   site_mean == ctrl_mean (6 dp), sc_ratio == 1.0,
                    site_mean ≈ 0.425702 (partial swath edge variant)
        """
        if record.get("status") != "ok":
            return False
        sm = record.get("site_mean")
        cm = record.get("ctrl_mean")
        sc = record.get("sc_ratio")
        if sm is None or cm is None or sc is None:
            return False
        return abs(sm - cm) < 1e-6 and sc == 1.0

    # ── Repair logic ──────────────────────────────────────────────────────────

    def Repair(
        self,
        timeseries: dict,
        dry_run:    bool = False,
    ) -> tuple[dict, int, int]:
        """Scan and repair degenerate records in-place.

        Parameters
        ----------
        timeseries: the loaded time-series dict
        dry_run:    report changes without writing

        Returns
        -------
        (repaired_timeseries, n_checked, n_fixed)
        """
        n_checked = 0
        n_fixed   = 0

        for site, records in timeseries.items():
            cfg = BACKFILL_SITES.get(site)
            if cfg is None:
                self._log.warning("Unknown site %s — skipping", site)
                continue
            lat, lon = cfg["lat"], cfg["lon"]

            for rec in records:
                if not self._IsDegenerate(rec):
                    continue

                n_checked += 1
                npy_name     = rec.get("npy")
                tif_path_str = rec.get("tif")
                acq_date     = rec.get("acquisition_date", "?")

                if not npy_name or not tif_path_str:
                    self._log.warning(
                        "  %s %s — no npy/tif path, marking no_coverage",
                        site, acq_date,
                    )
                    if not dry_run:
                        rec["status"]       = "no_coverage"
                        rec["coverage_note"] = "missing npy or tif path"
                        for fld in SC_FIELDS:
                            rec[fld] = None
                    n_fixed += 1
                    continue

                npy_path = NPY_CACHE / npy_name
                tif_path = ROOT / tif_path_str

                if not npy_path.exists():
                    self._log.warning(
                        "  %s %s — NPY not found (%s)", site, acq_date, npy_name
                    )
                    if not dry_run:
                        rec["status"]        = "no_coverage"
                        rec["coverage_note"] = "npy_not_found"
                        for fld in SC_FIELDS:
                            rec[fld] = None
                    n_fixed += 1
                    continue

                try:
                    vf = self._SiteValidFraction(npy_path, tif_path, lon, lat)
                except Exception as exc:
                    self._log.warning(
                        "  %s %s — coverage check error: %s", site, acq_date, exc
                    )
                    vf = 0.0

                self._log.info(
                    "  %s  %s  valid_frac=%.3f  %s",
                    site, acq_date, vf,
                    "FAIL" if vf < self.min_valid else "ok (not degenerate?)",
                )

                if vf < self.min_valid:
                    n_fixed += 1
                    if not dry_run:
                        rec["status"]         = "no_coverage"
                        rec["valid_fraction"]  = round(vf, 4)
                        rec["coverage_note"]   = (
                            f"site crop {vf*100:.0f}% valid pixels — "
                            "partial-swath tile, plant outside S2 swath"
                        )
                        for fld in SC_FIELDS:
                            rec[fld] = None

        return timeseries, n_checked, n_fixed

    # ── Public API ────────────────────────────────────────────────────────────

    def Run(self, dry_run: bool = False) -> dict:
        """Load, repair, and save the backfill time-series JSON.

        Parameters
        ----------
        dry_run: report what would change without writing

        Returns
        -------
        dict with keys: n_checked, n_fixed, timeseries
        """
        self._log.info("Loading %s", TIMESERIES_OUT)
        timeseries = json.loads(TIMESERIES_OUT.read_text())
        total = sum(len(v) for v in timeseries.values())
        self._log.info(
            "Loaded %d sites, %d total records", len(timeseries), total
        )

        timeseries, n_checked, n_fixed = self.Repair(timeseries, dry_run)

        self._log.info("")
        self._log.info("Checked %d degenerate records, fixed %d", n_checked, n_fixed)

        if dry_run:
            self._log.info("DRY RUN — no files written")
            return {"n_checked": n_checked, "n_fixed": n_fixed, "timeseries": timeseries}

        bak = TIMESERIES_OUT.with_suffix(".json.bak")
        shutil.copy2(TIMESERIES_OUT, bak)
        self._log.info("Backup written to %s", bak.name)
        TIMESERIES_OUT.write_text(json.dumps(timeseries, indent=2))
        self._log.info("Repaired JSON written to %s", TIMESERIES_OUT)

        self.PrintSummary(timeseries)
        return {"n_checked": n_checked, "n_fixed": n_fixed, "timeseries": timeseries}

    def PrintSummary(self, timeseries: dict) -> None:
        """Print per-site no-coverage counts."""
        print("\n=== Repair Summary ===")
        for site, records in timeseries.items():
            no_cov = [r for r in records if r.get("status") == "no_coverage"]
            ok     = [r for r in records if r.get("status") == "ok"]
            if no_cov:
                dates = [r.get("acquisition_date", "?") for r in no_cov]
                print(
                    f"  {site}: {len(ok)} ok, {len(no_cov)} no_coverage → {dates}"
                )


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WS4 historical backfill pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # download
    dl = sub.add_parser("download", help="Step 1: download 2020–2023 summer tiles")
    dl.add_argument("--sites", nargs="+", default=None,
                    choices=list(BACKFILL_SITES.keys()))
    dl.add_argument("--years", nargs="+", type=int, default=None)
    dl.add_argument("--max-cloud", type=float, default=DEFAULT_MAX_CLOUD)
    dl.add_argument("--dry-run", action="store_true")

    # timeseries
    ts = sub.add_parser("timeseries", help="Step 2: compile S/C time series")
    ts.add_argument("--sites", nargs="+", default=None,
                    choices=list(BACKFILL_SITES.keys()))
    ts.add_argument("--years", nargs="+", type=int, default=None)
    ts.add_argument("--no-inference", action="store_true")
    ts.add_argument("--weights", default=None)

    # repair
    rp = sub.add_parser("repair", help="Step 3: repair partial-swath records")
    rp.add_argument("--dry-run", action="store_true")
    rp.add_argument("--min-valid", type=float, default=MIN_SITE_VALID_FRAC)

    return parser


if __name__ == "__main__":
    args = _build_cli().parse_args()
    if args.cmd == "download":
        BackfillDownloader().Run(
            sites=args.sites, years=args.years,
            max_cloud=args.max_cloud, dry_run=args.dry_run,
        )
    elif args.cmd == "timeseries":
        BackfillTimeseriesBuilder().Run(
            sites=args.sites, years=args.years,
            no_inference=args.no_inference, weights=args.weights,
        )
    elif args.cmd == "repair":
        BackfillCoverageRepairer(min_valid=args.min_valid).Run(dry_run=args.dry_run)
