"""
validate_weisweiler_multidate.py
================================
Runs CH4Net inference and S/C + CFAR evaluation on all 3 cached Weisweiler
(T31UGS) acquisition dates to test temporal consistency.

Confirmed emitter date: R008 2024-09-18 (TROPOMI-validated)
Additional dates:       R108 2024-08-31, R108 2024-09-20

Usage:
    cd ~/Downloads/methane-api
    python validate_weisweiler_multidate.py --weights weights/european_model_v8.pth
    python validate_weisweiler_multidate.py --weights weights/european_model.pth   # v9 after training
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from apply_bitemporal_diff import (
    apply_bitemporal_difference,
    compute_sc_ratio,
    run_inference,
    B12_IDX,
    THRESHOLD,
    CFAR_K,
)
from src.detection.ch4net_model import CH4NetDetector

# ── Site config ───────────────────────────────────────────────────────────────
WEISWEILER_LAT = 50.866
WEISWEILER_LON = 6.316
REF_NPY        = Path("data/npy_cache/T31UGS_ref_20240127.npy")
NPY_CACHE      = Path("data/npy_cache")

TARGET_TILES = [
    {
        "stem": "S2B_MSIL1C_20240918T103619_N0511_R008_T31UGS_20240918T142046",
        "date": "2024-09-18",
        "orbit": "R008",
        "tropomi": "confirmed emitter",
    },
    {
        "stem": "S2A_MSIL1C_20240831T103021_N0511_R108_T31UGS_20240831T130645",
        "date": "2024-08-31",
        "orbit": "R108",
        "tropomi": "unknown",
    },
    {
        "stem": "S2A_MSIL1C_20240920T102721_N0511_R108_T31UGS_20240920T123207",
        "date": "2024-09-20",
        "orbit": "R108",
        "tropomi": "unknown",
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/european_model.pth")
    parser.add_argument("--output-dir", default="results_weisweiler_multidate")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-run inference even if TIF already exists")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.weights).exists():
        print(f"ERROR: weights not found: {args.weights}")
        sys.exit(1)

    if not REF_NPY.exists():
        print(f"ERROR: reference tile not found: {REF_NPY}")
        sys.exit(1)

    print("=" * 70)
    print(f"  Weisweiler multi-date validation — CH4Net")
    print(f"  Weights:   {args.weights}")
    print(f"  Reference: {REF_NPY.name}")
    print(f"  CFAR k:    {CFAR_K}σ")
    print("=" * 70)

    detector = CH4NetDetector(args.weights, threshold=THRESHOLD)
    ref_arr  = np.load(REF_NPY)

    rows = []

    for tile in TARGET_TILES:
        stem    = tile["stem"]
        date    = tile["date"]
        orbit   = tile["orbit"]
        tropomi = tile["tropomi"]

        npy_path = NPY_CACHE / f"{stem}.npy"
        geo_path = NPY_CACHE / f"{stem}_geo.json"

        if not npy_path.exists():
            print(f"\n  [{date}] MISSING: {npy_path.name}")
            continue

        geo_meta = json.loads(geo_path.read_text())

        print(f"\n  [{date}]  orbit={orbit}  TROPOMI={tropomi}")

        # ── Baseline (no BT diff) ─────────────────────────────────────────
        out_tif_orig = out_dir / f"baseline_{stem}.tif"
        if not out_tif_orig.exists() or args.no_cache:
            target = np.load(npy_path)
            print(f"    Running baseline inference...")
            run_inference(target, detector, geo_meta, out_tif_orig)
            del target
        else:
            print(f"    Baseline TIF cached: {out_tif_orig.name}")

        sc_orig = compute_sc_ratio(out_tif_orig, WEISWEILER_LAT, WEISWEILER_LON)

        # ── Bi-temporal (B12-only diff) ───────────────────────────────────
        out_tif_bt = out_dir / f"bitemporal_{stem}.tif"
        if not out_tif_bt.exists() or args.no_cache:
            target = np.load(npy_path)
            print(f"    Running bitemporal inference...")
            bt_arr = apply_bitemporal_difference(target, ref_arr, replace_channels=[B12_IDX])
            run_inference(bt_arr, detector, geo_meta, out_tif_bt)
            del target, bt_arr
        else:
            print(f"    Bitemporal TIF cached: {out_tif_bt.name}")

        sc_bt = compute_sc_ratio(out_tif_bt, WEISWEILER_LAT, WEISWEILER_LON)

        def fv(d, k): return d.get(k)
        sc_o  = fv(sc_orig, "sc_ratio")
        sc_b  = fv(sc_bt,   "sc_ratio")
        cf_o  = fv(sc_orig, "cfar_detect")
        cf_b  = fv(sc_bt,   "cfar_detect")
        sig_o = fv(sc_orig, "ctrl_sigma")
        sig_b = fv(sc_bt,   "ctrl_sigma")
        thr_o = fv(sc_orig, "cfar_thresh")
        thr_b = fv(sc_bt,   "cfar_thresh")

        print(f"    Baseline  S/C={sc_o:.3f}  CFAR={'DETECT' if cf_o else 'no':6}  "
              f"σ={sig_o:.5f}  thresh={thr_o:.5f}")
        print(f"    Biotemp   S/C={sc_b:.3f}  CFAR={'DETECT' if cf_b else 'no':6}  "
              f"σ={sig_b:.5f}  thresh={thr_b:.5f}")

        rows.append(dict(
            date=date, orbit=orbit, tropomi=tropomi,
            sc_baseline=sc_o, cfar_baseline=cf_o, sigma_baseline=sig_o,
            sc_bt=sc_b,       cfar_bt=cf_b,       sigma_bt=sig_b,
        ))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n")
    print("=" * 80)
    print("  WEISWEILER TEMPORAL CONSISTENCY SUMMARY")
    print("=" * 80)
    print(f"  {'Date':<13} {'Orbit':<7} {'TROPOMI':<20} "
          f"{'Base S/C':>9} {'CFAR':>6}  {'BT S/C':>8} {'CFAR BT':>8}")
    print("  " + "-" * 76)
    for r in rows:
        cf  = "✓" if r["cfar_baseline"] else "✗"
        cfb = "✓" if r["cfar_bt"] else "✗"
        sc_o = f"{r['sc_baseline']:.3f}" if r["sc_baseline"] else "—"
        sc_b = f"{r['sc_bt']:.3f}" if r["sc_bt"] else "—"
        print(f"  {r['date']:<13} {r['orbit']:<7} {r['tropomi']:<20} "
              f"{sc_o:>9} {cf:>6}  {sc_b:>8} {cfb:>8}")
    print("=" * 80)

    # Save JSON
    out_json = out_dir / "multidate_results.json"
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\n  Results saved → {out_json}")
    print(f"\n  Ideal outcome: Sep-18 CFAR=✓ on both modes; Aug-31 and Sep-20 ideally")
    print(f"  also ✓ if plant was operating, or ✗ if it was offline (not a failure).")


if __name__ == "__main__":
    main()
