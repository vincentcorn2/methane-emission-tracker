"""
run_analysis.py
===============
Two back-to-back analyses, no downloads required.

PART 1 — Ring Profile Regression
  Fits a log-linear decay model to the concentric-ring mean probability
  data for every site in results_validation/summary.json.
  A genuine point-source should show negative slope (prob decays with distance).
  Outputs a gradient score per site that is more informative than S/C ratio.

PART 2 — CEMF + IME Quantification on Groningen & Maasvlakte
  Uses cached .npy band files + detection GeoTIFFs to estimate emission
  flow rates (kg/h) and annualised IRA liability for the two cleanest detections.
"""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────
# PART 1: RING PROFILE REGRESSION
# ─────────────────────────────────────────────────────────────

def fit_ring_gradient(ring_profile):
    """
    Fit log-linear model: log(mean_prob) = a + b * distance_km
    b < 0  →  prob decays with distance (point-source signal)
    b > 0  →  prob rises with distance (terrain artefact)

    Returns dict with slope, r2, and a normalised gradient score.
    """
    distances = []
    log_probs  = []
    for ring in ring_profile:
        mp = ring.get("mean_prob")
        if mp and mp > 0:
            mid = (ring["inner_km"] + ring["outer_km"]) / 2
            distances.append(mid)
            log_probs.append(np.log(mp))

    if len(distances) < 3:
        return None

    x = np.array(distances)
    y = np.array(log_probs)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    slope = ((x - xm) @ (y - ym)) / ((x - xm) @ (x - xm))
    intercept = ym - slope * xm
    y_hat = intercept + slope * x
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Gradient score: -slope * 1000 (positive = good decay, negative = bad)
    gradient_score = -slope * 1000

    # Near/far ratio: prob at innermost ring / prob at outermost ring
    near_far_ratio = np.exp(y[0]) / np.exp(y[-1]) if len(y) >= 2 else None

    return dict(
        slope=round(slope, 6),
        intercept=round(intercept, 6),
        r2=round(r2, 3),
        gradient_score=round(gradient_score, 3),
        near_far_ratio=round(near_far_ratio, 3) if near_far_ratio else None,
        n_rings=n,
    )


def run_ring_regression():
    print("=" * 70)
    print("PART 1 — RING PROFILE REGRESSION ANALYSIS")
    print("=" * 70)
    print("Model: log(mean_prob) = a + b·distance_km")
    print("  b < 0  →  decay (✓ point-source signal)")
    print("  b > 0  →  rising (✗ terrain artefact)")
    print()

    with open("results_validation/summary.json") as f:
        raw = json.load(f)

    # Deduplicate
    seen, sites = set(), []
    for d in raw:
        if d["site"] not in seen:
            seen.add(d["site"])
            sites.append(d)

    results = []
    for d in sites:
        rings = d.get("ring_profile", [])
        fit = fit_ring_gradient(rings)
        if fit is None:
            continue
        sc = d.get("sc", {})
        sc_ratio = sc.get("sc_ratio") if isinstance(sc, dict) else None
        prec = d.get("precision_10km", {})
        prec_val = prec.get("precision") if isinstance(prec, dict) else None
        results.append(dict(
            site=d["site"],
            site_type=d.get("site_type", ""),
            sc_ratio=sc_ratio,
            precision_10km=prec_val,
            total_events=d.get("total_events", 0),
            **fit,
        ))

    # Sort by gradient_score descending
    results.sort(key=lambda x: x["gradient_score"], reverse=True)

    # Print table
    hdr = f"{'Site':<26} {'Type':<16} {'Grad▼':>7} {'NF-ratio':>9} {'R²':>5} {'S/C':>6} {'Prec':>6} {'verdict'}"
    print(hdr)
    print("─" * 90)
    for r in results:
        stype = r["site_type"]
        gs    = r["gradient_score"]
        nf    = r["near_far_ratio"]
        r2    = r["r2"]
        sc    = r["sc_ratio"]
        pr    = r["precision_10km"]

        # Verdict
        if stype == "control":
            if gs > 1.0:
                verdict = "⚠ FP (control fires)"
            else:
                verdict = "✓ quiet"
        elif stype in ("emitter", "emitter_miss"):
            if gs > 2.0 and nf and nf > 1.5:
                verdict = "✓✓ strong gradient"
            elif gs > 0.5 or (nf and nf > 1.2):
                verdict = "✓ weak gradient"
            else:
                verdict = "✗ no gradient"
        else:
            verdict = "—"

        sc_str = f"{sc:.3f}" if sc else "  —  "
        pr_str = f"{pr:.2f}" if pr else "  —  "
        nf_str = f"{nf:.3f}" if nf else "  —  "
        print(f"{r['site']:<26} {stype:<16} {gs:>7.3f} {nf_str:>9} {r2:>5.3f} {sc_str:>6} {pr_str:>6}  {verdict}")

    print()
    print("KEY INSIGHT:")
    emitter_scores  = [r["gradient_score"] for r in results if r["site_type"] == "emitter"]
    control_scores  = [r["gradient_score"] for r in results if r["site_type"] == "control"]
    if emitter_scores and control_scores:
        print(f"  Emitter gradient scores : mean={np.mean(emitter_scores):.3f}  max={max(emitter_scores):.3f}  min={min(emitter_scores):.3f}")
        print(f"  Control gradient scores : mean={np.mean(control_scores):.3f}  max={max(control_scores):.3f}  min={min(control_scores):.3f}")
        overlap = sum(1 for c in control_scores if c > min(emitter_scores))
        print(f"  Controls above worst emitter: {overlap}/{len(control_scores)}")

    # Save results
    os.makedirs("results_analysis", exist_ok=True)
    with open("results_analysis/ring_regression.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → results_analysis/ring_regression.json")
    return results


# ─────────────────────────────────────────────────────────────
# PART 2: CEMF + IME QUANTIFICATION
# ─────────────────────────────────────────────────────────────

SITES_TO_QUANTIFY = [
    dict(
        name="groningen",
        lat=53.252, lon=6.682,
        npy="data/npy_cache/S2A_MSIL1C_20240817T105031_N0511_R051_T31UGV_20240817T125303.npy",
        geo="data/npy_cache/S2A_MSIL1C_20240817T105031_N0511_R051_T31UGV_20240817T125303_geo.json",
        tif="results_validation/groningen/detection_T31UGV_2024-08-17.tif",
        scene_id="S2A_T31UGV_20240817",
        timestamp="2024-08-17T10:50:31Z",
        wind_ms=5.5,   # typical SW wind over NL in August; ERA5 fallback
        wind_source="climatological_NL_Aug",
        radius_km=10,  # restrict quantification to near-plant events
    ),
    dict(
        name="maasvlakte",
        lat=51.951, lon=4.004,
        npy="data/npy_cache/S2B_MSIL1C_20240825T105619_N0511_R094_T31UET_20240825T130043.npy",
        geo="data/npy_cache/S2B_MSIL1C_20240825T105619_N0511_R094_T31UET_20240825T130043_geo.json",
        tif="results_validation/maasvlakte/detection_T31UET_2024-08-25.tif",
        scene_id="S2B_T31UET_20240825",
        timestamp="2024-08-25T10:56:19Z",
        wind_ms=4.5,   # typical N-NW coastal wind over Rotterdam in August
        wind_source="climatological_NL_Aug",
        radius_km=10,
    ),
]

# Band order in .npy (B10 excluded): B01-B09, B8A, B11, B12
B11_IDX = 10
B12_IDX = 11
THRESHOLD = 0.18
PIXEL_SIZE_M = 10.0  # resampled to 10m grid


def latlon_to_pixel(lat, lon, geo_meta):
    """Convert geographic coordinates to pixel (row, col) using affine transform."""
    transform = geo_meta["transform"]  # [a, b, c, d, e, f] where c=x_origin, f=y_origin
    # transform: [pixel_width, 0, x_origin, 0, pixel_height, y_origin]
    # or as a list: [a, b, c, d, e, f]
    if isinstance(transform, dict):
        a, b, c = transform["a"], transform["b"], transform["c"]
        d, e, f = transform["d"], transform["e"], transform["f"]
    else:
        a, b, c, d, e, f = transform[0], transform[1], transform[2], transform[3], transform[4], transform[5]

    # Invert affine to get pixel coords
    # x = a*col + b*row + c  →  col = (x - c - b*row) / a  (b=0 for north-up)
    # y = d*col + e*row + f  →  row = (y - f - d*col) / e  (d=0 for north-up)
    from pyproj import Transformer
    crs = geo_meta.get("crs", "EPSG:32631")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)

    col = (x - c) / a
    row = (y - f) / e
    return int(round(row)), int(round(col))


def make_radius_mask(shape, center_row, center_col, radius_km, pixel_size_m):
    """Binary mask of pixels within radius_km of center pixel."""
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    dist_m = np.sqrt(((rows - center_row) * pixel_size_m) ** 2 +
                     ((cols - center_col) * pixel_size_m) ** 2)
    return dist_m <= (radius_km * 1000)


def run_quantification():
    print()
    print("=" * 70)
    print("PART 2 — CEMF + IME QUANTIFICATION")
    print("  Sites: Groningen, Maasvlakte")
    print("  Using cached .npy band files — no downloads")
    print("=" * 70)

    try:
        from pyproj import Transformer
        from src.quantification.cemf import run_cemf
        from src.quantification.ime import CEMFIntegratedMassEnhancement
    except ImportError as e:
        print(f"  Import error: {e}")
        return
    # PIL fallback for rasterio (rasterio not available in sandbox)
    try:
        import rasterio
        _use_rasterio = True
    except ImportError:
        from PIL import Image as _PIL_Image
        import warnings as _warnings
        _use_rasterio = False

    all_results = []

    for site in SITES_TO_QUANTIFY:
        name = site["name"]
        print(f"\n{'─'*60}")
        print(f"  Site: {name.upper()}")
        print(f"{'─'*60}")

        # ── Load band array
        print(f"  Loading .npy ({os.path.getsize(site['npy'])/1e9:.1f} GB)...", end="", flush=True)
        bands = np.load(site["npy"])   # (H, W, 12)
        print(f" shape={bands.shape}")

        b11 = bands[:, :, B11_IDX].astype(np.float32) / 255.0   # convert DN→TOA reflectance for CEMF
        b12 = bands[:, :, B12_IDX].astype(np.float32) / 255.0   # convert DN→TOA reflectance for CEMF

        # ── Load geo metadata
        with open(site["geo"]) as f:
            geo_meta = json.load(f)

        # ── Load detection GeoTIFF as probability map
        print(f"  Loading detection GeoTIFF...")
        if _use_rasterio:
            with rasterio.open(site["tif"]) as src:
                prob_map = src.read(1).astype(np.float32)
        else:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                prob_map = np.array(_PIL_Image.open(site["tif"]), dtype=np.float32)

        print(f"  prob_map shape={prob_map.shape}  max={prob_map.max():.3f}  pct>{THRESHOLD}={100*(prob_map>THRESHOLD).mean():.3f}%")

        # ── Full-tile binary mask at threshold
        full_mask = (prob_map >= THRESHOLD).astype(np.uint8)

        # ── Near-plant mask (restrict to radius_km around plant)
        try:
            ctr_row, ctr_col = latlon_to_pixel(site["lat"], site["lon"], geo_meta)
            print(f"  Plant pixel: row={ctr_row}, col={ctr_col}")
            radius_mask = make_radius_mask(prob_map.shape, ctr_row, ctr_col,
                                           site["radius_km"], PIXEL_SIZE_M)
            near_mask = (full_mask & radius_mask).astype(np.uint8)
            print(f"  Near-plant events (within {site['radius_km']} km): {near_mask.sum()} pixels "
                  f"({near_mask.sum() * PIXEL_SIZE_M**2 / 1e6:.2f} km²)")
        except Exception as e:
            print(f"  Warning: could not compute radius mask ({e}), using full tile")
            near_mask = full_mask

        if near_mask.sum() == 0:
            print(f"  ⚠ No detected pixels near plant — skipping quantification")
            continue

        # ── Resize b11/b12 to match prob_map if needed
        h, w = prob_map.shape
        if b11.shape != (h, w):
            from scipy.ndimage import zoom
            zy, zx = h / b11.shape[0], w / b11.shape[1]
            print(f"  Resampling bands from {b11.shape} to ({h},{w})...")
            b11 = zoom(b11, (zy, zx), order=1)
            b12 = zoom(b12, (zy, zx), order=1)

        # ── Run CEMF on near-plant mask
        print(f"  Running CEMF...")
        cemf_result = run_cemf(
            b11=b11,
            b12=b12,
            mask=near_mask,
            scene_id=site["scene_id"],
            timestamp=site["timestamp"],
        )

        if not cemf_result.retrieval_valid:
            print(f"  ⚠ CEMF retrieval invalid: {cemf_result.warning}")
            print(f"  Falling back to geometric IME proxy...")

        print(f"  CEMF result:")
        print(f"    Plume pixels:   {cemf_result.n_plume_pixels:,}")
        print(f"    Retrieval valid: {cemf_result.retrieval_valid}")
        if cemf_result.retrieval_valid:
            print(f"    Total mass:     {cemf_result.total_mass_kg:.2f} kg")

        # ── Run IME
        ime = CEMFIntegratedMassEnhancement(pixel_size_m=PIXEL_SIZE_M)

        if cemf_result.retrieval_valid and cemf_result.total_mass_kg > 0:
            result = ime.estimate_from_cemf(
                cemf_result=cemf_result,
                wind_speed_ms=site["wind_ms"],
                wind_source=site["wind_source"],
            )
        else:
            # Geometric fallback
            result = ime.estimate(
                plume_mask=near_mask,
                band_11=b11,
                band_12=b12,
                wind_speed_ms=site["wind_ms"],
            )

        print(f"\n  ── EMISSION ESTIMATE ──────────────────────────────")
        print(f"  Methodology:        {result.methodology}")
        print(f"  Wind speed used:    {site['wind_ms']} m/s ({site['wind_source']})")
        print(f"  Flow rate:          {result.flow_rate_kgh:.1f} kg CH₄/hr")
        print(f"  Uncertainty range:  {result.flow_rate_lower_kgh:.1f} – {result.flow_rate_upper_kgh:.1f} kg/hr  (±{50 if 'CEMF' not in result.methodology else 40}%)")
        if result.annual_tonnes:
            print(f"  Annual equivalent:  {result.annual_tonnes:,.0f} tonnes CH₄/yr")
        if result.ira_waste_charge_usd:
            print(f"  IRA liability:      ${result.ira_waste_charge_usd:,.0f} USD/yr  (@$1,500/tonne, 2026)")
        if result.plume_length_m:
            print(f"  Plume length:       {result.plume_length_m/1000:.1f} km")

        all_results.append(dict(
            site=name,
            methodology=result.methodology,
            wind_ms=site["wind_ms"],
            flow_rate_kgh=result.flow_rate_kgh,
            flow_rate_lower_kgh=result.flow_rate_lower_kgh,
            flow_rate_upper_kgh=result.flow_rate_upper_kgh,
            annual_tonnes=result.annual_tonnes,
            ira_usd=result.ira_waste_charge_usd,
            plume_pixels=cemf_result.n_plume_pixels,
            cemf_valid=cemf_result.retrieval_valid,
            total_mass_kg=cemf_result.total_mass_kg if cemf_result.retrieval_valid else None,
        ))

    # Save
    if all_results:
        with open("results_analysis/quantification.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Saved → results_analysis/quantification.json")

    # Cross-check against TROPOMI literature for Groningen
    print(f"\n{'─'*60}")
    print("  LITERATURE CROSS-CHECK (Groningen gas field)")
    print("  TROPOMI-derived estimates (Schneising et al. 2020):")
    print("  ~50,000 t/yr total Groningen field CH4  (~5,700 kg/hr continuous)")
    print("  Individual well clusters: 100–2,000 kg/hr depending on activity")
    print("  Our estimate covers only the near-plant S2 tile area (10 km radius)")
    print("  → expect our estimate to be a fraction of the full-field total")

    return all_results


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    regression_results = run_ring_regression()
    quant_results      = run_quantification()

    print()
    print("=" * 70)
    print("DONE — results saved to results_analysis/")
    print("=" * 70)
