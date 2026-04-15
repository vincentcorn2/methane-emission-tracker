"""
run_quant_fixed.py
==================
Re-runs CEMF+IME quantification with the corrected sensitivity coefficient
(4e-7 per Varon 2021) without needing rasterio or pyproj.
Uses PIL for GeoTIFF reading and a pure-Python UTM projection.
Supports any UTM zone — zone is auto-detected from geo_meta CRS string.
"""

import json, math, warnings
import numpy as np
from PIL import Image

# ── Suppress PIL decompression bomb warning for 10980×10980 tiles
Image.MAX_IMAGE_PIXELS = None

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from src.quantification.cemf import run_cemf
from src.quantification.ime import CEMFIntegratedMassEnhancement


# ── Generic UTM projection (WGS84, any zone) ─────────────────────────────────
def latlon_to_utm(lat_deg, lon_deg, utm_zone: int):
    """
    Convert WGS84 lat/lon to UTM easting/northing for any zone.
    Central meridian = (zone - 1) * 6 - 180 + 3 degrees.
    Northern hemisphere only (no southern offset needed for Europe).
    """
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    lon0 = math.radians((utm_zone - 1) * 6 - 180 + 3)   # central meridian

    a  = 6378137.0
    f  = 1 / 298.257223563
    b  = a * (1 - f)
    e2 = 1 - (b/a)**2
    k0 = 0.9996
    E0 = 500000.0

    N   = a / math.sqrt(1 - e2 * math.sin(lat)**2)
    T   = math.tan(lat)**2
    C   = e2 / (1 - e2) * math.cos(lat)**2
    A_  = math.cos(lat) * (lon - lon0)
    e2p = e2 / (1 - e2)
    n   = (a - b) / (a + b)
    A0  = a * (1 - n + 5/4*(n**2 - n**3) + 81/64*n**4)
    B0  = 3*a/2 * (n - n**2 + 7/8*(n**3 - n**4))
    C0  = 15*a/16 * (n**2 - n**3 + 51/32*n**4)
    D0  = 35*a/48 * (n**3 - n**4)
    M   = A0*lat - B0*math.sin(2*lat) + C0*math.sin(4*lat) - D0*math.sin(6*lat)

    easting  = k0*N*(A_ + (1-T+C)*A_**3/6 + (5-18*T+T**2+72*C-58*e2p)*A_**5/120) + E0
    northing = k0*(M + N*math.tan(lat)*(A_**2/2 + (5-T+9*C+4*C**2)*A_**4/24 +
                                          (61-58*T+T**2+600*C-330*e2p)*A_**6/720))
    return easting, northing


def utm_zone_from_crs(crs_str: str) -> int:
    """Extract UTM zone number from an EPSG CRS string like 'EPSG:32633'."""
    # EPSG:326ZZ → zone ZZ (northern hemisphere)
    import re
    m = re.search(r'326(\d{2})', crs_str)
    if m:
        return int(m.group(1))
    # Fallback: zone 31 (Netherlands tiles)
    return 31


def latlon_to_pixel(lat, lon, geo_meta):
    """Convert lat/lon to pixel (row, col) using the tile's own UTM projection."""
    t    = geo_meta["transform"]          # [a, b, c, d, e, f]
    a, b, c = t[0], t[1], t[2]
    d, e, f = t[3], t[4], t[5]
    zone = utm_zone_from_crs(geo_meta.get("crs", "EPSG:32631"))
    x, y = latlon_to_utm(lat, lon, zone)
    col  = (x - c) / a
    row  = (y - f) / e
    return int(round(row)), int(round(col))


def make_radius_mask(shape, center_row, center_col, radius_km, pixel_size_m):
    rows, cols = np.ogrid[:shape[0], :shape[1]]
    dist_m = np.sqrt(((rows - center_row) * pixel_size_m)**2 +
                     ((cols - center_col) * pixel_size_m)**2)
    return dist_m <= (radius_km * 1000)


# ── Site configs ─────────────────────────────────────────────────────────────
SITES = [
    dict(
        name="groningen",
        lat=53.252, lon=6.682,
        npy="data/npy_cache/S2A_MSIL1C_20240817T105031_N0511_R051_T31UGV_20240817T125303.npy",
        geo="data/npy_cache/S2A_MSIL1C_20240817T105031_N0511_R051_T31UGV_20240817T125303_geo.json",
        tif="results_validation/groningen/detection_T31UGV_2024-08-17.tif",
        scene_id="S2A_T31UGV_20240817",
        timestamp="2024-08-17T10:50:31Z",
        wind_ms=5.5,
        wind_source="ERA5 climatology",
        radius_km=10,
    ),
    dict(
        name="maasvlakte",
        lat=51.951, lon=4.004,
        npy="data/npy_cache/S2B_MSIL1C_20240825T105619_N0511_R094_T31UET_20240825T130043.npy",
        geo="data/npy_cache/S2B_MSIL1C_20240825T105619_N0511_R094_T31UET_20240825T130043_geo.json",
        tif="results_validation/maasvlakte/detection_T31UET_2024-08-25.tif",
        scene_id="S2B_T31UET_20240825",
        timestamp="2024-08-25T10:56:19Z",
        wind_ms=4.5,
        wind_source="ERA5 climatology",
        radius_km=10,
        note="v8 non-detection (S/C=0.210) — CEMF result is terrain contrast, not methane",
    ),
    # ── Confirmed detections from scale-up (2026-04-10) ──────────────────────
    dict(
        name="belchatow",
        lat=51.266, lon=19.315,
        npy="data/npy_cache/S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.npy",
        geo="data/npy_cache/S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611_geo.json",
        tif="results_bitemporal/belchatow/original_S2B_MSIL1C_20240824T094549_N0511_R079_T34UCB_20240824T115611.tif",
        scene_id="S2B_T34UCB_20240824",
        timestamp="2024-08-24T09:45:49Z",
        wind_ms=3.5,           # ERA5 climatological fallback — replace with live ERA5
        wind_source="climatological_fallback",
        radius_km=15,          # larger radius: plant+associated gas handling infrastructure
        note="v8 DETECT: S/C=27.303 (classic, comparable to Weisweiler 23.4). "
             "Europe's #1 CO2 emitter. UTM Zone 34N (EPSG:32634).",
    ),
    dict(
        name="lippendorf",
        lat=51.178, lon=12.378,
        npy="data/npy_cache/S2B_MSIL1C_20240922T101629_N0511_R065_T33UUS_20240922T140318.npy",
        geo="data/npy_cache/S2B_MSIL1C_20240922T101629_N0511_R065_T33UUS_20240922T140318_geo.json",
        tif="results_bitemporal/lippendorf/original_S2B_MSIL1C_20240922T101629_N0511_R065_T33UUS_20240922T140318.tif",
        scene_id="S2B_T33UUS_20240922",
        timestamp="2024-09-22T10:16:29Z",
        wind_ms=3.5,           # ERA5 climatological fallback — replace with live ERA5
        wind_source="climatological_fallback",
        radius_km=12,
        note="v8 DETECT: S/C=155.362, CFAR=yes (thresh_ratio=2.914). "
             "1782 MW lignite x2, strongest signal in dataset. UTM Zone 33N (EPSG:32633).",
    ),
]

B11_IDX    = 10
B12_IDX    = 11
THRESHOLD  = 0.18
PIXEL_SIZE = 10.0

all_results = []

for site in SITES:
    name = site["name"]
    print(f"\n{'='*60}")
    print(f"  Site: {name.upper()}")
    print(f"{'='*60}")

    # Load bands
    print(f"  Loading .npy...", end="", flush=True)
    bands = np.load(site["npy"])
    print(f" shape={bands.shape}")

    b11 = bands[:, :, B11_IDX].astype(np.float32) / 255.0
    b12 = bands[:, :, B12_IDX].astype(np.float32) / 255.0

    # Load geo metadata
    with open(site["geo"]) as f:
        geo_meta = json.load(f)

    # Load detection TIF via PIL
    print(f"  Loading detection GeoTIFF...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prob_map = np.array(Image.open(site["tif"]), dtype=np.float32)
    print(f"  prob_map shape={prob_map.shape}  max={prob_map.max():.4f}")

    full_mask = (prob_map >= THRESHOLD).astype(np.uint8)

    # Radius mask around plant centroid
    try:
        ctr_row, ctr_col = latlon_to_pixel(site["lat"], site["lon"], geo_meta)
        print(f"  Plant pixel: row={ctr_row}, col={ctr_col}")
        radius_mask = make_radius_mask(prob_map.shape, ctr_row, ctr_col,
                                       site["radius_km"], PIXEL_SIZE)
        near_mask = (full_mask & radius_mask).astype(np.uint8)
        print(f"  Near-plant plume pixels (within {site['radius_km']} km): {near_mask.sum()}")
    except Exception as ex:
        print(f"  Warning: radius mask failed ({ex}), using full tile")
        near_mask = full_mask

    if near_mask.sum() == 0:
        print(f"  No detected pixels near plant — skipping")
        continue

    # Resize b11/b12 to match prob_map if needed
    h, w = prob_map.shape
    if b11.shape != (h, w):
        from scipy.ndimage import zoom
        zy, zx = h / b11.shape[0], w / b11.shape[1]
        print(f"  Resampling bands {b11.shape} → ({h},{w})...")
        b11 = zoom(b11, (zy, zx), order=1)
        b12 = zoom(b12, (zy, zx), order=1)

    # CEMF
    print(f"  Running CEMF (sensitivity=4e-7, Varon 2021)...")
    cemf_result = run_cemf(
        b11=b11, b12=b12, mask=near_mask,
        scene_id=site["scene_id"], timestamp=site["timestamp"],
    )

    print(f"  Retrieval valid: {cemf_result.retrieval_valid}")
    if cemf_result.retrieval_valid:
        print(f"  Total column mass:  {cemf_result.total_mass_kg:.3f} kg")

    # IME
    ime = CEMFIntegratedMassEnhancement(pixel_size_m=PIXEL_SIZE)
    if cemf_result.retrieval_valid and cemf_result.total_mass_kg > 0:
        result = ime.estimate_from_cemf(
            cemf_result=cemf_result,
            wind_speed_ms=site["wind_ms"],
            wind_source=site["wind_source"],
        )
    else:
        result = ime.estimate(
            plume_mask=near_mask, band_11=b11, band_12=b12,
            wind_speed_ms=site["wind_ms"],
        )

    print(f"\n  ── CORRECTED EMISSION ESTIMATE ─────────────────────")
    print(f"  Flow rate:         {result.flow_rate_kgh:.2f} kg CH₄/hr")
    print(f"  Uncertainty:       {result.flow_rate_lower_kgh:.2f} – {result.flow_rate_upper_kgh:.2f} kg/hr (±40%)")
    if result.annual_tonnes:
        print(f"  Annual equivalent: {result.annual_tonnes:,.1f} t CH₄/yr")
    if result.ira_waste_charge_usd:
        print(f"  IRA liability:     ${result.ira_waste_charge_usd:,.0f}/yr")

    all_results.append(dict(
        site=name,
        methodology=result.methodology,
        wind_ms=site["wind_ms"],
        flow_rate_kgh=round(result.flow_rate_kgh, 4),
        flow_rate_lower_kgh=round(result.flow_rate_lower_kgh, 4),
        flow_rate_upper_kgh=round(result.flow_rate_upper_kgh, 4),
        annual_tonnes=result.annual_tonnes,
        ira_usd=result.ira_waste_charge_usd,
        plume_pixels=cemf_result.n_plume_pixels,
        cemf_valid=cemf_result.retrieval_valid,
        total_mass_kg=cemf_result.total_mass_kg,
        cemf_sensitivity_coeff="4e-7 (Varon 2021)",
    ))

# Save corrected results
out = "results_analysis/quantification.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n\nSaved corrected quantification → {out}")
