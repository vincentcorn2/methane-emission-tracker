#!/usr/bin/env python3
"""Quick diagnostic: current state of conformal calibration set."""
import json
from pathlib import Path

SCORES = Path("results_analysis/nonemitter_sc_scores.json")
THRESH = Path("results_analysis/calibrated_threshold.json")

with open(SCORES) as f:
    sites = json.load(f)

with open(THRESH) as f:
    thresh = json.load(f)

print("=" * 60)
print("CONFORMAL CALIBRATION DIAGNOSTIC")
print("=" * 60)

all_sites     = [s for s in sites]
ok_sites      = [s for s in sites if s.get("status") == "ok"]
valid_sites   = [s for s in ok_sites if s.get("sc_cfar") is not None]
invalid_sites = [s for s in ok_sites if s.get("sc_cfar") is None]
failed_sites  = [s for s in sites if s.get("status") != "ok"]

print(f"\n  Total in scores file : {len(all_sites)}")
print(f"  status=ok            : {len(ok_sites)}")
print(f"  valid sc_cfar        : {len(valid_sites)}")
print(f"  sc_cfar=None (ok)    : {len(invalid_sites)}")
print(f"  failed/error         : {len(failed_sites)}")

print(f"\n--- Sites where status=ok but sc_cfar=None ---")
for s in invalid_sites:
    print(f"  {s['location_id']:20s}  ecoregion={s.get('ecoregion','?'):15s}  status={s['status']}")

print(f"\n--- Failed / non-ok sites ---")
for s in failed_sites:
    print(f"  {s['location_id']:20s}  status={s['status']}")

# Ecoregion breakdown of valid sites
from collections import Counter
eco_counts = Counter(s.get("ecoregion", "unknown") for s in valid_sites)
print(f"\n--- Ecoregion breakdown (valid sc_cfar only, n={len(valid_sites)}) ---")
for eco, n in sorted(eco_counts.items()):
    print(f"  {eco:25s}: {n}")

# New sites (notemit_020+)
def _id_num(s):
    lid = s["location_id"]
    for prefix in ("nonemit_", "notemit_"):
        lid = lid.replace(prefix, "")
    return int(lid)

new_valid   = [s for s in valid_sites   if _id_num(s) >= 20]
new_invalid = [s for s in invalid_sites if _id_num(s) >= 20]
print(f"\n--- New sites (notemit_020+) ---")
print(f"  Added with valid sc_cfar : {len(new_valid)}")
print(f"  Added with sc_cfar=None  : {len(new_invalid)}")
if new_valid:
    for s in new_valid:
        print(f"    {s['location_id']:20s}  sc_cfar={s['sc_cfar']:.4f}  ecoregion={s.get('ecoregion','?')}")

print(f"\n--- calibrated_threshold.json ---")
print(f"  n_calibration : {thresh.get('n_calibration')}")
print(f"  tau (α=0.10)  : {thresh.get('tau_alpha_0.10') or thresh.get('tau')}")
print(f"  computed_at   : {thresh.get('computed_at','?')}")

print()
if len(valid_sites) > thresh.get("n_calibration", 0):
    delta = len(valid_sites) - thresh["n_calibration"]
    print(f"  ⚠  {delta} valid site(s) not yet reflected in threshold — re-run conformal_threshold.py")
else:
    print(f"  ✓  threshold is up to date with valid scores (n={len(valid_sites)})")
print("=" * 60)
