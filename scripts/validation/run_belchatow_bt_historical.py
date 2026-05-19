"""
scripts/run_belchatow_bt_historical.py
=======================================
Run bitemporal differencing on the two CFAR-confirmed historical Belchatow
acquisitions (2020-06-01, 2021-06-06) using the production v8 model and the
cached December 2023 reference tile.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apply_bitemporal_diff import (
    CH4NetDetector,
    NPY_CACHE,
    OUT_DIR,
    WEIGHTS,
    B12_IDX,
    apply_bitemporal_difference,
    compute_ring_profile,
    compute_sc_ratio,
    find_geo_meta,
    ring_gradient,
    run_inference,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("belchatow_bt_historical")

SITE_NAME = "belchatow"
LAT, LON  = 51.266, 19.315
TILE_ID   = "T34UCB"
REF_NPY   = NPY_CACHE / "T34UCB_ref_20231218.npy"

RUNS = [
    {
        "key":        "belchatow_2020-06-01",
        "date":       "2020-06-01",
        "target_npy": NPY_CACHE / "S2A_MSIL1C_20200601T095041_N0500_R079_T34UCB_20230402T070951.npy",
    },
    {
        "key":        "belchatow_2021-06-06",
        "date":       "2021-06-06",
        "target_npy": NPY_CACHE / "S2A_MSIL1C_20210606T095031_N0500_R079_T34UCB_20230313T135138.npy",
    },
]

OUT_JSON = Path("results_analysis/bitemporal_comparison.json")


def sc_record(sc_dict, ring_grad, tif_path):
    return {
        "tif":               str(tif_path),
        "sc_ratio":          sc_dict.get("sc_ratio"),
        "sc_cfar":           sc_dict.get("sc_cfar"),
        "site_mean":         sc_dict.get("site_mean"),
        "ctrl_mean":         sc_dict.get("ctrl_mean"),
        "ctrl_mu":           sc_dict.get("ctrl_mu"),
        "ctrl_sigma":        sc_dict.get("ctrl_sigma"),
        "cv_ctrl":           sc_dict.get("cv_ctrl"),
        "cfar_thresh_ratio": sc_dict.get("cfar_thresh_ratio"),
        "cfar_thresh":       sc_dict.get("cfar_thresh"),
        "cfar_detect":       sc_dict.get("cfar_detect"),
        "cfar_margin":       sc_dict.get("cfar_margin"),
        "ctrl_n":            sc_dict.get("ctrl_n"),
        "ring_gradient":     ring_grad,
        "sc_error":          sc_dict.get("error"),
    }


def process_one(run, detector, reference):
    target_npy = run["target_npy"]
    if not target_npy.exists():
        log.error("Target .npy not found: %s", target_npy)
        return {"status": "missing_target", "target_npy": target_npy.name}

    geo_meta = find_geo_meta(target_npy)
    if geo_meta is None:
        log.error("No geo metadata for %s", target_npy.name)
        return {"status": "no_geo_meta", "target_npy": target_npy.name}

    site_dir = OUT_DIR / SITE_NAME
    site_dir.mkdir(parents=True, exist_ok=True)

    out_orig = site_dir / f"original_{target_npy.stem}.tif"
    if not out_orig.exists():
        log.info("[%s] Running baseline inference (cache miss)...", run["date"])
        target = np.load(target_npy)
        run_inference(target, detector, geo_meta, out_orig)
        del target
    else:
        log.info("[%s] Baseline tif already cached: %s", run["date"], out_orig.name)

    sc_orig   = compute_sc_ratio(out_orig, LAT, LON)
    ring_orig = compute_ring_profile(out_orig, LAT, LON)
    grad_orig = ring_gradient(ring_orig)

    out_bt = site_dir / f"bitemporal_{target_npy.stem}.tif"
    if not out_bt.exists():
        log.info("[%s] Computing B12 delta against %s...", run["date"], REF_NPY.name)
        target = np.load(target_npy)
        if target.shape != reference.shape:
            log.error("[%s] Shape mismatch target=%s ref=%s",
                      run["date"], target.shape, reference.shape)
            return {"status": "shape_mismatch", "target_npy": target_npy.name}

        bt_array = apply_bitemporal_difference(
            target, reference, replace_channels=[B12_IDX]
        )
        log.info("[%s] Running CH4Net on BT array...", run["date"])
        run_inference(bt_array, detector, geo_meta, out_bt)
        del target, bt_array
    else:
        log.info("[%s] BT tif already cached: %s", run["date"], out_bt.name)

    sc_bt   = compute_sc_ratio(out_bt, LAT, LON)
    ring_bt = compute_ring_profile(out_bt, LAT, LON)
    grad_bt = ring_gradient(ring_bt)

    log.info("[%s] baseline  S/C=%.3f  CFAR=%s  cv_ctrl=%.3f",
             run["date"],
             sc_orig.get("sc_ratio") or float("nan"),
             "DETECT" if sc_orig.get("cfar_detect") else "no",
             sc_orig.get("cv_ctrl") or float("nan"))
    log.info("[%s] BT        S/C=%.3f  CFAR=%s  cv_ctrl=%.3f",
             run["date"],
             sc_bt.get("sc_ratio") or float("nan"),
             "DETECT" if sc_bt.get("cfar_detect") else "no",
             sc_bt.get("cv_ctrl") or float("nan"))

    return {
        "tile_id":    TILE_ID,
        "date":       run["date"],
        "target_npy": target_npy.name,
        "ref_npy":    REF_NPY.name,
        "status":     "ok",
        "original":   sc_record(sc_orig, grad_orig, out_orig),
        "bitemporal": sc_record(sc_bt,   grad_bt,   out_bt),
    }


def main():
    if not REF_NPY.exists():
        log.error("Reference tile missing: %s", REF_NPY)
        sys.exit(1)

    log.info("Loading CH4Net v8 weights from %s", WEIGHTS)
    detector = CH4NetDetector(WEIGHTS)

    log.info("Loading reference tile (one-time, ~1.4 GB)...")
    reference = np.load(REF_NPY)
    log.info("Reference shape: %s", reference.shape)

    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            store = json.load(f)
        backup = OUT_JSON.with_suffix(".json.bak")
        with open(backup, "w") as f:
            json.dump(store, f, indent=2)
        log.info("Loaded existing JSON (%d entries); backup -> %s",
                 len(store), backup.name)
    else:
        store = {}
        log.info("Creating new bitemporal_comparison.json")

    for run in RUNS:
        log.info("=" * 65)
        log.info("Site: %s   Date: %s", SITE_NAME.upper(), run["date"])
        result = process_one(run, detector, reference)
        store[run["key"]] = result

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(store, f, indent=2)
    log.info("Wrote %s (%d entries total)", OUT_JSON, len(store))

    print("\n" + "=" * 70)
    print(f"{'Date':<14}{'Baseline S/C':>14}{'BT S/C':>14}{'cv_ctrl (BT)':>16}{'BT CFAR':>10}")
    print("-" * 70)
    for run in RUNS:
        r = store[run["key"]]
        if r.get("status") != "ok":
            print(f"{run['date']:<14}  {r.get('status'):>50}")
            continue
        o, b = r["original"], r["bitemporal"]
        print(
            f"{run['date']:<14}"
            f"{o.get('sc_ratio') or 0:>14.3f}"
            f"{b.get('sc_ratio') or 0:>14.3f}"
            f"{b.get('cv_ctrl') or 0:>16.3f}"
            f"{'DETECT' if b.get('cfar_detect') else 'no':>10}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
