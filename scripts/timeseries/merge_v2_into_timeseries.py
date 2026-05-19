"""
scripts/merge_v2_into_timeseries.py  (v2)
==========================================
Merge successful v2 quantifications back into the main time series JSON.

The v2 JSON structure: list of records with keys
  - matched_npy        scene filename ending in .npy
  - winner             the best successful quantification block (already chosen)
  - attempts           list of all crop sizes tried (for reference)
  - outcome            "quantified" if winner is non-null

Strategy: for every time-series record whose npy matches a v2 record with
outcome="quantified", replace its quantification block with the v2 winner.

Backup is written to belchatow_annual_timeseries.json.pre_v2_merge.bak.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
TIMESERIES = ROOT / "results_analysis" / "belchatow_annual_timeseries.json"
V2_RESULTS = ROOT / "results_analysis" / "requant_fallback_v2.json"
BACKUP = TIMESERIES.with_suffix(".json.pre_v2_merge.bak")


def main():
    store = json.loads(TIMESERIES.read_text())
    records = store.get("records", []) if isinstance(store, dict) else store
    v2 = json.loads(V2_RESULTS.read_text())
    v2_records = v2.get("records", []) if isinstance(v2, dict) else v2

    if not BACKUP.exists():
        shutil.copy(TIMESERIES, BACKUP)
        print(f"Backup written: {BACKUP}")
    else:
        print(f"Backup already exists, not overwriting: {BACKUP}")

    # Build v2 lookup by matched_npy
    v2_by_npy = {}
    for v in v2_records:
        npy = v.get("matched_npy")
        if npy:
            v2_by_npy[npy] = v

    print(f"v2 records loaded:   {len(v2_records)}")
    print(f"v2 with quantified outcome: {sum(1 for v in v2_records if v.get('outcome') == 'quantified')}")
    print(f"Time-series records: {len(records)}")
    print()

    updated = 0
    skipped_no_winner = 0
    no_match = 0

    for r in records:
        npy = r.get("npy")
        if not npy:
            no_match += 1
            continue
        v_entry = v2_by_npy.get(npy)
        if v_entry is None:
            no_match += 1
            continue

        winner = v_entry.get("winner")
        if not winner or winner.get("status") != "quantified":
            skipped_no_winner += 1
            continue

        # Build the merged quantification block
        new_quant = {
            "status":              "quantified",
            "sc_ratio":            v_entry.get("sc_ratio") or (r.get("detection") or {}).get("sc_ratio"),
            "flow_rate_kgh":       winner.get("flow_rate_kgh"),
            "flow_rate_lower_kgh": winner.get("flow_rate_lower_kgh"),
            "flow_rate_upper_kgh": winner.get("flow_rate_upper_kgh"),
            "wind_speed_ms":       winner.get("wind_speed_ms"),
            "wind_dir_deg":        winner.get("wind_dir_deg"),
            "wind_source":         winner.get("wind_source"),
            "uncertainty_pct":     winner.get("uncertainty_pct"),
            "annual_tonnes_if_continuous": winner.get("annual_tonnes_if_continuous"),
            "n_plume_pixels":      winner.get("n_plume_pixels"),
            "governance_flags":    winner.get("governance_flags", []),
            "crop_px_used":        winner.get("crop_px"),
            "source":              "requant_fallback_v2",
            "merged_at":           datetime.utcnow().isoformat() + "Z",
        }
        r["quantification"] = new_quant
        updated += 1

    # Write back
    if isinstance(store, dict):
        store["records"] = records
        TIMESERIES.write_text(json.dumps(store, indent=2))
    else:
        TIMESERIES.write_text(json.dumps(records, indent=2))

    print("=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Records updated with v2 quantification: {updated}")
    print(f"v2 matched but winner not quantified:   {skipped_no_winner}")
    print(f"Time-series records not in v2:          {no_match}")
    print()
    print(f"Updated: {TIMESERIES}")


if __name__ == "__main__":
    main()
