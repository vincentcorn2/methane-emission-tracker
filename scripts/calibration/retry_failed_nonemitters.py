"""
scripts/retry_failed_nonemitters.py
=====================================
Find non-emitter calibration sites that didn't reach status=ok and retry them.

The expansion run (expand_nonemitter_calibration.py) writes a record for every
attempted site, including failures (no_products, download_failed, no_geo_meta,
exception). The conformal threshold computation drops these via the
status=="ok" filter, but the site IDs remain in the JSON which means
expand_nonemitter_calibration.py won't retry them on subsequent runs.

This script:
  1. Reads results_analysis/nonemitter_sc_scores.json
  2. Identifies entries with status != "ok"
  3. Removes them from the list (so the next expand run won't skip them)
  4. Re-invokes the expansion script with the failed location_ids via --ids
  5. Lets expand_nonemitter_calibration.py do the actual work

Usage:
  python scripts/retry_failed_nonemitters.py --dry-run   # show what would retry
  caffeinate -i python scripts/retry_failed_nonemitters.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORES_JSON = ROOT / "results_analysis" / "nonemitter_sc_scores.json"
EXPAND_SCRIPT = ROOT / "scripts" / "expand_nonemitter_calibration.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List failed sites without retrying")
    args = parser.parse_args()

    if not SCORES_JSON.exists():
        print(f"ERROR: {SCORES_JSON} not found")
        sys.exit(1)

    data = json.loads(SCORES_JSON.read_text())
    if not isinstance(data, list):
        print(f"ERROR: expected list at top level of {SCORES_JSON}, got {type(data).__name__}")
        sys.exit(1)

    failed = [s for s in data if s.get("status") != "ok"]
    if not failed:
        print("No failed non-emitter entries to retry. Done.")
        return

    print(f"Found {len(failed)} failed entries:")
    for s in failed:
        print(f"  {s.get('location_id'):<14}  status={s.get('status'):<25}  "
              f"error={(s.get('error') or '')[:60]}")

    failed_ids = [s["location_id"] for s in failed if s.get("location_id")]
    if not failed_ids:
        print("\nNo location_ids on failed entries — cannot retry.")
        return

    if args.dry_run:
        print(f"\nDRY RUN — would remove these from JSON and retry: {failed_ids}")
        return

    # Remove failed entries from JSON so expand_nonemitter_calibration's
    # existing_ids filter doesn't skip them. Backup first.
    backup = SCORES_JSON.with_suffix(".json.bak_before_retry")
    backup.write_text(SCORES_JSON.read_text())
    print(f"\nBackup written: {backup}")

    kept = [s for s in data if s.get("status") == "ok"]
    SCORES_JSON.write_text(json.dumps(kept, indent=2))
    print(f"Removed {len(failed)} failed entries from {SCORES_JSON}")
    print(f"Remaining OK entries: {len(kept)}")

    # Invoke expand with --ids of the failed sites
    cmd = [
        sys.executable, str(EXPAND_SCRIPT),
        "--ids", *failed_ids,
    ]
    print(f"\n$ {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
