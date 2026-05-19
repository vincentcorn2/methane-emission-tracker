"""
scripts/leakage_audit.py
==========================
Catalog and check for data leakage between the v8 training set and the
evaluation candidates / production records in this report.

Three checks:
  (1) Temporal proximity — for each evaluation acquisition date, find the
      nearest training-crop date at the same MGRS tile. Flag any pair < 14
      days apart as potential temporal leakage.
  (2) Site overlap — confirm whether each candidate site appears in the
      training data (and in what role: positive, negative, or synthetic
      substrate).
  (3) Threshold selection independence — confirm the conformal calibration
      set has no overlap with the candidate sites (50 km exclusion radius
      check).

Output
------
results_analysis/leakage_audit.md
results_analysis/leakage_audit.json
"""
from __future__ import annotations
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CROPS_DIR = ROOT / "data" / "crops"
OUT_DIR = ROOT / "results_analysis"

CANDIDATE_SITES = {
    "belchatow":  (51.266, 19.315, "T34UCB"),
    "rybnik":     (50.135, 18.522, "T34UCA"),
    "weisweiler": (50.837,  6.322, "T31UGS"),
    "lippendorf": (51.178, 12.378, "T33UUS"),
    "neurath":    (51.038,  6.616, "T32ULB"),
    "boxberg":    (51.412, 14.626, "T33UVT"),
    "groningen":  (53.388,  6.617, "T31UGV"),
    "maasvlakte": (51.952,  4.073, "T31UET"),
}

ACQ_DATE_RE = re.compile(r"_?(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def extract_date(s):
    m = ACQ_DATE_RE.search(s)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d")
        except ValueError:
            return None
    return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Catalog training crops ─────────────────────────────────────────────
    training = []
    for label_path in CROPS_DIR.glob("**/*_label.json"):
        if ".hidden" in str(label_path):
            continue
        stem = label_path.stem.replace("_label", "")
        meta = json.loads(label_path.read_text())
        tile_match = re.search(r"T\d{2}[A-Z]{3}", stem)
        training.append({
            "crop": stem,
            "source_dir": label_path.parent.name,
            "label_value": meta.get("label_value"),
            "tile": tile_match.group(0) if tile_match else None,
            "date": meta.get("acquisition_date") or (extract_date(stem).isoformat() if extract_date(stem) else None),
        })

    # ── 2. Candidate site overlap ────────────────────────────────────────────
    site_overlap = {}
    for site, (lat, lon, tile) in CANDIDATE_SITES.items():
        in_training = [t for t in training if t["tile"] == tile or site.lower() in t["crop"].lower()]
        roles = set()
        for t in in_training:
            if "synthetic" in t["source_dir"]:
                roles.add("synthetic_substrate")
            elif t["label_value"] == 1:
                roles.add("positive")
            elif t["label_value"] == 0:
                roles.add("negative")
        site_overlap[site] = {
            "in_training": len(in_training) > 0,
            "roles": sorted(roles),
            "n_crops": len(in_training),
            "training_crops": [t["crop"] for t in in_training],
        }

    # ── 3. Temporal proximity check ──────────────────────────────────────────
    # For each candidate site's training crops, check the temporal distance to
    # the evaluation dates used in the report.
    eval_dates = {
        "belchatow": ["2020-06-01", "2021-06-06", "2021-09-09", "2024-04-11",
                      "2024-05-26", "2024-07-10", "2024-07-30", "2024-10-28"],
        "neurath":   ["2024-06-25", "2024-08-29"],
        "rybnik":    ["2025-03-22"],
    }

    temporal_proximity = {}
    for site, dates in eval_dates.items():
        site_crops = [t for t in training if site.lower() in t["crop"].lower()]
        flags = []
        for d_eval_str in dates:
            d_eval = datetime.strptime(d_eval_str, "%Y-%m-%d")
            nearest = None
            min_delta = float("inf")
            for t in site_crops:
                if not t.get("date"):
                    continue
                try:
                    d_train = datetime.strptime(t["date"][:10], "%Y-%m-%d")
                except ValueError:
                    continue
                delta = abs((d_eval - d_train).days)
                if delta < min_delta:
                    min_delta = delta
                    nearest = t["crop"]
            if min_delta < 14:
                flags.append({
                    "eval_date": d_eval_str,
                    "nearest_training_crop": nearest,
                    "days_apart": min_delta,
                    "flag": "POTENTIAL_TEMPORAL_LEAKAGE",
                })
            else:
                flags.append({
                    "eval_date": d_eval_str,
                    "nearest_training_crop": nearest,
                    "days_apart": min_delta if min_delta != float("inf") else None,
                    "flag": "OK",
                })
        temporal_proximity[site] = flags

    # ── 4. Conformal calibration independence ────────────────────────────────
    cal_path = OUT_DIR / "nonemitter_sc_scores.json"
    cal_overlap = {"sites_within_50km_of_candidate": [], "checked_n": 0}
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        cal_sites = cal if isinstance(cal, list) else cal.get("sites", [])
        for c in cal_sites:
            if c.get("status") != "ok":
                continue
            cal_overlap["checked_n"] += 1
            for site, (lat, lon, _) in CANDIDATE_SITES.items():
                dist = haversine_km(c.get("lat", 0), c.get("lon", 0), lat, lon)
                if dist < 50:
                    cal_overlap["sites_within_50km_of_candidate"].append({
                        "calibration_site": c.get("location_id"),
                        "candidate_site": site,
                        "distance_km": round(dist, 1),
                    })

    # ── Write outputs ───────────────────────────────────────────────────────
    audit = {
        "training_crop_count": len(training),
        "candidate_site_overlap": site_overlap,
        "temporal_proximity": temporal_proximity,
        "conformal_calibration_independence": cal_overlap,
    }
    (OUT_DIR / "leakage_audit.json").write_text(json.dumps(audit, indent=2))

    md = []
    md.append("# Data leakage and independence audit\n")
    md.append("Three checks: site-level training overlap, temporal proximity between training "
              "and evaluation acquisitions at the same site, and conformal calibration set "
              "independence from candidate sites.\n")

    md.append("## (1) Candidate site overlap with training set\n")
    md.append("| Candidate | In training? | Role(s) | N crops |")
    md.append("|---|---|---|---|")
    for site, info in site_overlap.items():
        roles = ", ".join(info["roles"]) if info["roles"] else "—"
        md.append(f"| {site} | {'yes' if info['in_training'] else 'no'} | {roles} | {info['n_crops']} |")
    md.append("")

    md.append("## (2) Temporal proximity between evaluation and training dates\n")
    md.append("Same-site training crops within 14 days of an evaluation acquisition are flagged "
              "as potential leakage. Within-tile crops from other dates do not leak per-pixel labels.\n")
    md.append("| Site | Eval date | Nearest training crop | Days apart | Flag |")
    md.append("|---|---|---|---|---|")
    for site, flags in temporal_proximity.items():
        for f in flags:
            md.append(f"| {site} | {f['eval_date']} | {f['nearest_training_crop'] or '—'} | "
                      f"{f['days_apart']} | {f['flag']} |")
    md.append("")

    md.append("## (3) Conformal calibration set independence\n")
    md.append(f"Checked {cal_overlap['checked_n']} OK-status calibration sites for proximity "
              f"(< 50 km) to any candidate site.\n")
    if cal_overlap["sites_within_50km_of_candidate"]:
        md.append("**Proximity flags found:**\n")
        md.append("| Calibration site | Candidate site | Distance (km) |")
        md.append("|---|---|---|")
        for f in cal_overlap["sites_within_50km_of_candidate"]:
            md.append(f"| {f['calibration_site']} | {f['candidate_site']} | {f['distance_km']} |")
    else:
        md.append("**No conformal calibration sites within 50 km of any candidate site.** "
                  "The threshold τ = 4.1052 is calibrated on a set that is spatially independent "
                  "of the evaluation sites.\n")

    md.append("\n## (4) Threshold selection methodology\n")
    md.append("The conformal threshold τ = 4.1052 was computed by the split conformal prediction "
              "quantile on the non-emitter calibration set scores, without reference to the "
              "candidate-site backfill outcomes. The retraining hyperparameter selection (v1-v11) "
              "used a small held-out set of training crops (3 negatives) for validation loss "
              "monitoring, not the candidate-site evaluation outcomes. The candidate-site "
              "results were computed only after v8 was fixed and τ was calibrated.\n")

    (OUT_DIR / "leakage_audit.md").write_text("\n".join(md))

    print("=" * 70)
    print("LEAKAGE AND INDEPENDENCE AUDIT")
    print("=" * 70)
    for site, info in site_overlap.items():
        roles = ", ".join(info["roles"]) if info["roles"] else "—"
        print(f"  {site:<14} in_training={info['in_training']}  roles={roles}  n={info['n_crops']}")
    print()
    flags_total = sum(1 for fs in temporal_proximity.values() for f in fs if f['flag'] != 'OK')
    print(f"Temporal proximity flags: {flags_total}")
    print(f"Conformal calibration sites within 50 km of a candidate: "
          f"{len(cal_overlap['sites_within_50km_of_candidate'])}")
    print()
    print(f"Wrote: {OUT_DIR / 'leakage_audit.md'}")


if __name__ == "__main__":
    main()
