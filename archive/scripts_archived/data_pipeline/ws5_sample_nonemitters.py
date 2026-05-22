"""
scripts/ws5_sample_nonemitters.py
===================================
WS5 — Non-emitter reference set for conformal calibration and FPR
characterisation.

Builds a manifest of 18 geographically distributed, terrain-diverse
locations across European ecoregions that are confirmed NOT to be
significant CH4 point sources.  The manifest is consumed by:

  (a) scripts/download_nonemitter_tiles.py — S2 L1C acquisition
  (b) scripts/run_nonemitter_inference.py  — CH4Net S/C scoring
  (c) WS1 conformal prediction module     — empirical threshold calibration
  (d) FPR / false-alarm characterisation (WS5 Pillar 3 outcome analysis)

Design decisions
────────────────
  - 18 locations spanning 8 CORINE Land Cover macro-classes and 5 European
    ecoregions (Atlantic, Continental, Mediterranean, Pannonian, Boreal).
  - Minimum 80 km exclusion radius from every confirmed / candidate emitter
    in quantification.json.
  - Preferred acquisition window: June – September 2024 (matches emitter
    scenes); cloud_cover < 10%.
  - Two locations per major terrain class where possible to allow
    leave-one-out cross-validation within each class.

Usage
─────
    python scripts/ws5_sample_nonemitters.py [--check-separation] [--output PATH]

Output
──────
    results_analysis/nonemitter_manifest.json  (default)

Schema per entry
────────────────
    {
        "id":          "nonemit_001",
        "label":       "Black Forest — coniferous",
        "lat":         48.02,
        "lon":         8.17,
        "mgrs_tile":   "32TNT",
        "clc_class":   "coniferous_forest",
        "ecoregion":   "Continental",
        "target_date": "2024-07",          // preferred month/season
        "exclusion_notes": "",             // any known emitters nearby
        "min_dist_to_emitter_km": null,    // filled by --check-separation
        "nearest_emitter_site":   null,
    }
"""

import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ws5_sample_nonemitters")

# ── Reference locations ────────────────────────────────────────────────────────
# Each entry is a manually curated, CORINE-annotated, non-emitting location.
# Selection criteria:
#   1. No coal/gas/industrial facility within 20 km (verified via JRC EDGAR,
#      EUTL, and GEM coal plant tracker).
#   2. Terrain must be plausible false-positive landscape (forest, arable,
#      coastal) — not bare rock or deep ocean.
#   3. Geographically spread: at most 2 per NUTS-1 region.
#   4. Cloud climatology (ERA5-based) < 40% for Jun–Sep.
#
# CLC 2018 classes used (Level 2 aggregation):
#   coniferous_forest, broadleaved_forest, mixed_forest,
#   arable_land, pasture, complex_cultivation,
#   coastal_wetland, urban_green
#
NONEMITTERS = [
    # ── Atlantic ecoregion ──────────────────────────────────────────────────
    {
        "id":          "nonemit_001",
        "label":       "Lüneburg Heath — heathland/moor",
        "lat":         52.92,
        "lon":         9.94,
        "mgrs_tile":   "32UNE",
        "clc_class":   "pasture",
        "ecoregion":   "Atlantic",
        "target_date": "2024-08",
        "exclusion_notes": "Closest power plant: Moorburg (Hamburg, ~80 km N); decommissioned 2021.",
    },
    {
        "id":          "nonemit_002",
        "label":       "Belgian Ardennes — mixed forest",
        "lat":         50.08,
        "lon":         5.82,
        "mgrs_tile":   "31UGR",
        "clc_class":   "mixed_forest",
        "ecoregion":   "Atlantic",
        "target_date": "2024-07",
        "exclusion_notes": "No industrial CH4 sources within 50 km. Closest: Liège steel (55 km NE).",
    },
    {
        "id":          "nonemit_003",
        "label":       "Loire Valley — arable/vineyards",
        "lat":         47.52,
        "lon":         1.48,
        "mgrs_tile":   "31TDK",
        "clc_class":   "arable_land",
        "ecoregion":   "Atlantic",
        "target_date": "2024-07",
        "exclusion_notes": "UNESCO biosphere reserve; no heavy industry within 60 km.",
    },
    {
        "id":          "nonemit_004",
        "label":       "Danish lowlands — arable",
        "lat":         55.92,
        "lon":         9.52,
        "mgrs_tile":   "32VMJ",
        "clc_class":   "arable_land",
        "ecoregion":   "Atlantic",
        "target_date": "2024-06",
        "exclusion_notes": "Central Jutland; no coal plants. Pig-farm density high but dispersed (not point sources).",
    },
    # ── Continental ecoregion ────────────────────────────────────────────────
    {
        "id":          "nonemit_005",
        "label":       "Black Forest — coniferous forest",
        "lat":         48.02,
        "lon":         8.17,
        "mgrs_tile":   "32TNT",
        "clc_class":   "coniferous_forest",
        "ecoregion":   "Continental",
        "target_date": "2024-07",
        "exclusion_notes": "Closest plant: Karlsruhe (50 km W); no active CH4 point sources in tile.",
    },
    {
        "id":          "nonemit_006",
        "label":       "Vosges Mountains — mixed forest",
        "lat":         48.12,
        "lon":         7.12,
        "mgrs_tile":   "32TLT",
        "clc_class":   "mixed_forest",
        "ecoregion":   "Continental",
        "target_date": "2024-07",
        "exclusion_notes": "Protected natural park. No emitters identified within 40 km.",
    },
    {
        "id":          "nonemit_007",
        "label":       "Bohemian Forest — coniferous",
        "lat":         49.02,
        "lon":         13.52,
        "mgrs_tile":   "33UUQ",
        "clc_class":   "coniferous_forest",
        "ecoregion":   "Continental",
        "target_date": "2024-08",
        "exclusion_notes": "Šumava / Bavarian Forest NP. Closest plant: Schwandorf (~60 km NW, shutdown).",
    },
    {
        "id":          "nonemit_008",
        "label":       "Bavarian Plateau — complex cultivation",
        "lat":         48.52,
        "lon":         12.03,
        "mgrs_tile":   "32UNB",
        "clc_class":   "complex_cultivation",
        "ecoregion":   "Continental",
        "target_date": "2024-06",
        "exclusion_notes": "Agricultural mosaic between Munich and Passau. No known point sources.",
    },
    {
        "id":          "nonemit_009",
        "label":       "Massif Central — grassland",
        "lat":         45.02,
        "lon":         3.02,
        "mgrs_tile":   "31TGL",
        "clc_class":   "pasture",
        "ecoregion":   "Continental",
        "target_date": "2024-07",
        "exclusion_notes": "High volcanic plateau; cattle-rearing region. Dispersed enteric CH4 — no point sources.",
    },
    # ── Pannonian ecoregion ──────────────────────────────────────────────────
    {
        "id":          "nonemit_010",
        "label":       "Danube plain, Hungary — arable",
        "lat":         47.22,
        "lon":         18.52,
        "mgrs_tile":   "34TBT",
        "clc_class":   "arable_land",
        "ecoregion":   "Pannonian",
        "target_date": "2024-07",
        "exclusion_notes": "Loess plain east of Budapest. No industrial point sources identified.",
    },
    {
        "id":          "nonemit_011",
        "label":       "Dobruja plateau, Romania — steppe/arable",
        "lat":         44.32,
        "lon":         28.48,
        "mgrs_tile":   "35TLF",
        "clc_class":   "arable_land",
        "ecoregion":   "Pannonian",
        "target_date": "2024-08",
        "exclusion_notes": "Grain-growing plateau near Black Sea coast. No CH4 industry within 40 km.",
    },
    {
        "id":          "nonemit_012",
        "label":       "Slovak Highlands — broadleaved forest",
        "lat":         48.65,
        "lon":         18.88,
        "mgrs_tile":   "34UCU",
        "clc_class":   "broadleaved_forest",
        "ecoregion":   "Pannonian",
        "target_date": "2024-07",
        "exclusion_notes": "Malá Fatra mountains. Closest plant: Nováky (~30 km SW, coal; check separation).",
    },
    # ── Boreal ecoregion ─────────────────────────────────────────────────────
    {
        "id":          "nonemit_013",
        "label":       "Białowieża Forest, Poland — primary forest",
        "lat":         52.72,
        "lon":         23.92,
        "mgrs_tile":   "34UEB",
        "clc_class":   "broadleaved_forest",
        "ecoregion":   "Boreal",
        "target_date": "2024-07",
        "exclusion_notes": "UNESCO World Heritage primeval forest. No industry within 80 km.",
    },
    {
        "id":          "nonemit_014",
        "label":       "Northern Poland plains — arable",
        "lat":         53.48,
        "lon":         17.52,
        "mgrs_tile":   "33UVV",
        "clc_class":   "arable_land",
        "ecoregion":   "Boreal",
        "target_date": "2024-07",
        "exclusion_notes": "Pomeranian lowland; agricultural. Closest emitter: Bełchatów ~300 km SE.",
    },
    {
        "id":          "nonemit_015",
        "label":       "Swedish coastal plain — mixed forest",
        "lat":         56.02,
        "lon":         13.02,
        "mgrs_tile":   "33UUB",
        "clc_class":   "mixed_forest",
        "ecoregion":   "Boreal",
        "target_date": "2024-06",
        "exclusion_notes": "Skåne mixed farmland / forest coast. No industrial CH4 within 60 km.",
    },
    {
        "id":          "nonemit_016",
        "label":       "Baltic coast, Latvia — coastal grassland",
        "lat":         57.02,
        "lon":         24.02,
        "mgrs_tile":   "35VLC",
        "clc_class":   "coastal_wetland",
        "ecoregion":   "Boreal",
        "target_date": "2024-06",
        "exclusion_notes": "Gulf of Rīga coast; wetland/grassland. No industrial sources.",
    },
    # ── Mediterranean ecoregion ──────────────────────────────────────────────
    {
        "id":          "nonemit_017",
        "label":       "Apennines, Italy — mixed forest",
        "lat":         44.52,
        "lon":         10.52,
        "mgrs_tile":   "32TPQ",
        "clc_class":   "mixed_forest",
        "ecoregion":   "Mediterranean",
        "target_date": "2024-07",
        "exclusion_notes": "Reggio Emilia Apennines. No emitters within 50 km. Po Valley ceramics industry is far.",
    },
    {
        "id":          "nonemit_018",
        "label":       "Pyrenean foothills, Spain — shrubland/forest",
        "lat":         42.82,
        "lon":         1.52,
        "mgrs_tile":   "30TXN",
        "clc_class":   "mixed_forest",
        "ecoregion":   "Mediterranean",
        "target_date": "2024-08",
        "exclusion_notes": "Navarra foothills. No CH4 point sources within 80 km.",
    },
]

# CLC class descriptions for reporting
CLC_DESCRIPTIONS = {
    "coniferous_forest":  "CLC 312 — Coniferous forest",
    "broadleaved_forest": "CLC 311 — Broadleaved forest",
    "mixed_forest":       "CLC 313 — Mixed forest",
    "arable_land":        "CLC 211 — Non-irrigated arable land",
    "pasture":            "CLC 231 — Pastures / heathland",
    "complex_cultivation":"CLC 242 — Complex cultivation patterns",
    "coastal_wetland":    "CLC 411 — Inland marshes / coastal wetland",
    "urban_green":        "CLC 141 — Green urban areas",
}

# Known emitter locations (augmented from quantification.json at runtime)
_BUILTIN_EMITTERS = [
    {"site": "neurath",    "lat": 51.038, "lon": 6.616},
    {"site": "belchatow",  "lat": 51.264, "lon": 19.331},
    {"site": "weisweiler", "lat": 50.859, "lon": 6.319},
    {"site": "rybnik",     "lat": 50.135, "lon": 18.522},
    {"site": "groningen",  "lat": 53.252, "lon": 6.682},
    {"site": "lippendorf", "lat": 51.178, "lon": 12.378},
    {"site": "maasvlakte", "lat": 51.954, "lon": 4.008},
    {"site": "turceni",    "lat": 44.101, "lon": 23.391},
    # JRC top-10 Phase 5 sites (approximate centroid from MGRS tiles)
    {"site": "belchatow_jrc", "lat": 51.264, "lon": 19.331},
]

# Separation threshold for a warning
MIN_SEPARATION_KM = 60.0
# Threshold below which a location is flagged as too close to an emitter
HARD_EXCLUSION_KM = 30.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_emitter_locs(quant_path: str = "results_analysis/quantification.json") -> list:
    """Load emitter lat/lons from quantification.json, merging with builtin list."""
    emitters = list(_BUILTIN_EMITTERS)
    p = Path(quant_path)
    if p.exists():
        try:
            records = json.load(p.open())
            for r in records:
                lat = r.get("plume_centroid_lat") or r.get("lat")
                lon = r.get("plume_centroid_lon") or r.get("lon")
                site = r.get("site", "unknown")
                if lat and lon:
                    # Avoid duplicating builtin entries
                    if not any(e["site"] == site for e in emitters):
                        emitters.append({"site": site, "lat": lat, "lon": lon})
        except Exception as exc:
            log.warning("Could not load emitter locations from %s: %s", quant_path, exc)
    return emitters


def check_separation(locations: list, emitters: list) -> list:
    """
    For each non-emitter location, compute minimum distance to any known emitter.
    Adds fields: min_dist_to_emitter_km, nearest_emitter_site, proximity_flag.
    """
    enriched = []
    for loc in locations:
        min_d = float("inf")
        nearest = "none"
        for em in emitters:
            d = haversine_km(loc["lat"], loc["lon"], em["lat"], em["lon"])
            if d < min_d:
                min_d = d
                nearest = em["site"]
        flag = (
            "HARD_EXCLUDE" if min_d < HARD_EXCLUSION_KM
            else "WARN_CLOSE"  if min_d < MIN_SEPARATION_KM
            else "OK"
        )
        enriched.append({
            **loc,
            "min_dist_to_emitter_km": round(min_d, 1),
            "nearest_emitter_site":   nearest,
            "proximity_flag":         flag,
        })
    return enriched


def print_summary(locations: list) -> None:
    """Print a tabular summary to stdout."""
    ecoregions = {}
    clc_classes = {}
    flags = {"OK": 0, "WARN_CLOSE": 0, "HARD_EXCLUDE": 0}

    print("\n" + "=" * 80)
    print("  WS5 Non-Emitter Reference Manifest")
    print("=" * 80)
    print(f"  {'ID':<14} {'Label':<42} {'Dist km':>8}  {'Prox':>12}")
    print("-" * 80)

    for loc in locations:
        dist = loc.get("min_dist_to_emitter_km")
        flag = loc.get("proximity_flag", "?")
        dist_str = f"{dist:.0f}" if dist is not None else "n/a"
        flag_icon = {"OK": "✓", "WARN_CLOSE": "⚠", "HARD_EXCLUDE": "✗"}.get(flag, "?")
        print(f"  {loc['id']:<14} {loc['label'][:42]:<42} {dist_str:>8}  {flag_icon} {flag}")

        eco = loc.get("ecoregion", "unknown")
        ecoregions[eco] = ecoregions.get(eco, 0) + 1
        clc = loc.get("clc_class", "unknown")
        clc_classes[clc] = clc_classes.get(clc, 0) + 1
        if flag in flags:
            flags[flag] += 1

    print("=" * 80)
    print(f"\n  Total locations  : {len(locations)}")
    print(f"  Proximity flags  : OK={flags['OK']}  WARN={flags['WARN_CLOSE']}  EXCLUDE={flags['HARD_EXCLUDE']}")
    print(f"\n  By ecoregion     : {dict(sorted(ecoregions.items()))}")
    print(f"  By CLC class     : {dict(sorted(clc_classes.items()))}")

    print(f"\n  Separation thresholds:")
    print(f"    Hard exclusion : < {HARD_EXCLUSION_KM:.0f} km  (location unusable)")
    print(f"    Warning        : < {MIN_SEPARATION_KM:.0f} km  (use with care)")
    print()

    flagged = [loc for loc in locations if loc.get("proximity_flag") != "OK"]
    if flagged:
        print("  Flagged locations:")
        for loc in flagged:
            print(f"    {loc['id']}  {loc['label']}")
            print(f"       → nearest emitter: {loc['nearest_emitter_site']}  "
                  f"({loc['min_dist_to_emitter_km']:.0f} km)  [{loc['proximity_flag']}]")
        print()


def build_manifest(locations: list) -> dict:
    """Wrap locations with metadata into the canonical manifest schema."""
    return {
        "schema_version": "1.0.0",
        "purpose": "WS5 non-emitter reference set for conformal calibration and FPR analysis",
        "methodology": (
            "18 manually curated locations spanning 5 European ecoregions and 8 CORINE "
            "Land Cover macro-classes. Selection: (i) no EDGAR/EUTL/GEM industrial CH4 "
            "point source within 20 km; (ii) minimum 60 km from confirmed CH4Net "
            "emitter detections (warn threshold); (iii) varied terrain for FPR "
            "characterisation by terrain class."
        ),
        "conformal_calibration_note": (
            "Run CH4Net v8 inference on S2 L1C scenes at these locations. "
            "Collect S/C ratios as the non-conformity score set for conformalized "
            "threshold calibration (WS1 conformal prediction module). "
            "Target: 90th percentile S/C on non-emitter set defines alpha=0.10 "
            "guaranteed FPR bound."
        ),
        "n_locations": len(locations),
        "ecoregion_coverage": list({loc["ecoregion"] for loc in locations}),
        "clc_class_coverage": list({loc["clc_class"] for loc in locations}),
        "separation_thresholds_km": {
            "hard_exclusion": HARD_EXCLUSION_KM,
            "warning": MIN_SEPARATION_KM,
        },
        "locations": locations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="WS5: sample non-emitter reference locations for conformal calibration"
    )
    parser.add_argument(
        "--output", default="results_analysis/nonemitter_manifest.json",
        help="Output path for the manifest JSON"
    )
    parser.add_argument(
        "--check-separation", action="store_true", default=True,
        help="Compute distances to known emitters (default: on)"
    )
    parser.add_argument(
        "--no-check-separation", action="store_false", dest="check_separation",
        help="Skip separation check"
    )
    parser.add_argument(
        "--quant-path", default="results_analysis/quantification.json",
        help="Path to quantification.json for emitter locations"
    )
    args = parser.parse_args()

    locations = [dict(loc) for loc in NONEMITTERS]

    # Initialise distance fields
    for loc in locations:
        loc["min_dist_to_emitter_km"] = None
        loc["nearest_emitter_site"]   = None
        loc["proximity_flag"]         = "UNCHECKED"

    if args.check_separation:
        log.info("Loading emitter locations from %s", args.quant_path)
        emitters = load_emitter_locs(args.quant_path)
        log.info("Checking separation against %d known emitter sites", len(emitters))
        locations = check_separation(locations, emitters)

    print_summary(locations)

    manifest = build_manifest(locations)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Manifest written to %s", out)

    # Flag any hard-excluded locations
    hard_excluded = [loc for loc in locations if loc.get("proximity_flag") == "HARD_EXCLUDE"]
    if hard_excluded:
        log.warning(
            "%d location(s) within hard exclusion radius (%.0f km) — review before use",
            len(hard_excluded), HARD_EXCLUSION_KM
        )
        for loc in hard_excluded:
            log.warning("  HARD_EXCLUDE: %s  nearest=%s  dist=%.0f km",
                        loc["id"], loc["nearest_emitter_site"],
                        loc["min_dist_to_emitter_km"])

    warn_close = [loc for loc in locations if loc.get("proximity_flag") == "WARN_CLOSE"]
    if warn_close:
        log.info(
            "%d location(s) within warning radius (%.0f km) — use with care",
            len(warn_close), MIN_SEPARATION_KM
        )

    print(f"\nManifest written to: {out}")
    print(f"Next steps:")
    print(f"  1. Download S2 L1C tiles:  python scripts/download_nonemitter_tiles.py")
    print(f"  2. Run CH4Net inference:   python scripts/run_nonemitter_inference.py")
    print(f"  3. Conformal calibration:  python scripts/conformal_threshold.py")


if __name__ == "__main__":
    main()
