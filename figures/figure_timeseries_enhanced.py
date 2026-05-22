"""
Bełchatów detection figure — Panel A restructured as 3 sub-panels:
  A1: S/C CFAR time series with winter shading + TROPOMI annotation
  A2: ClimateTrace monthly CH4 (t/month) vs CH4Net quantified flow (kg/h → t/day)
  A3: Sentinel-2 RGB map of mine with polygon, S/C crop box, detection centroid
  B:  Rybnik CM displacement + CFAR bar (unchanged)
  C:  Seasonal detection counts (unchanged)
"""

import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pathlib import Path

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'axes.linewidth': 0.8,
})
COLORS = {
    'above': '#0072B2', 'sub': '#BBBBBB', 'thresh': '#D55E00',
    'ct': '#009E73', 'quant': '#E69F00', 'winter': '#8888BB',
}
TAU = 3.5796
ROOT = Path(__file__).parent.parent

# ── helpers ──────────────────────────────────────────────────────────────────
def wgs84_to_utm34n(lat_deg, lon_deg):
    a=6378137.0; e2=0.00669437999014; k0=0.9996; lon0=21.0
    lat=math.radians(lat_deg); lon=math.radians(lon_deg); lon0r=math.radians(lon0)
    N=a/math.sqrt(1-e2*math.sin(lat)**2); T=math.tan(lat)**2
    A_=math.cos(lat)*(lon-lon0r); e4=e2**2; e6=e2**3
    M=a*((1-e2/4-3*e4/64-5*e6/256)*lat-(3*e2/8+3*e4/32+45*e6/1024)*math.sin(2*lat)
         +(15*e4/256+45*e6/1024)*math.sin(4*lat)-(35*e6/3072)*math.sin(6*lat))
    x=500000+k0*N*(A_+(1-T)*A_**3/6+(5-18*T+T**2)*A_**5/120)
    y=k0*(M+N*math.tan(lat)*(A_**2/2+(5-T+9*T)*A_**4/24+(61-58*T+T**2)*A_**6/720))
    return x, y

def utm_to_px(utm_x, utm_y, origin_x=300000, origin_y=5700000, res=10):
    """Return (col, row) pixel in the T34UCB tile."""
    return (utm_x - origin_x) / res, (origin_y - utm_y) / res

def pct_stretch(arr, lo=2, hi=98):
    p_lo, p_hi = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)

# ── Data ─────────────────────────────────────────────────────────────────────
with open(ROOT / 'results_analysis/belchatow_annual_timeseries_mbsp.json') as f:
    bel_data = json.load(f)

dates, sc_scores, is_detect, partial = [], [], [], []
flow_dates, flow_kgh, flow_lo, flow_hi = [], [], [], []

for r in bel_data['records']:
    dt = r.get('acquisition_date') or r.get('search', {}).get('acquisition_date')
    if not dt: continue
    det = r.get('detection', {}); q = r.get('quantification', {})
    if 'sc_cfar' not in det: continue
    dates.append(dt); sc_scores.append(det['sc_cfar'])
    is_detect.append(det.get('cfar_detect', False))
    partial.append(r.get('partial_swath', False))
    if det.get('cfar_detect') and q.get('flow_rate_kgh') is not None:
        flow_dates.append(dt)
        flow_kgh.append(q['flow_rate_kgh'])
        flow_lo.append(q['flow_rate_lower_kgh'])
        flow_hi.append(q['flow_rate_upper_kgh'])

df = pd.DataFrame({'date': dates, 'sc_cfar': sc_scores,
                   'detect': is_detect, 'partial': partial})
df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
df = df[~df['partial']].copy()
df['quarter'] = df['date'].dt.quarter

fd = pd.to_datetime(flow_dates, format='mixed', utc=True)
flow_kgh = np.array(flow_kgh)
flow_lo  = np.array(flow_lo)
flow_hi  = np.array(flow_hi)
# Convert kg/h → t/month using actual days in the detection month
# (apples-to-apples with ClimateTrace monthly totals)
import calendar
days_in_month = np.array([calendar.monthrange(d.year, d.month)[1] for d in fd])
to_t_month = 24 * days_in_month / 1000   # array, one per detection
flow_t    = flow_kgh * to_t_month
flow_t_lo = flow_lo  * to_t_month
flow_t_hi = flow_hi  * to_t_month

# ClimateTrace
ct_raw = pd.read_csv(ROOT / 'data/16168_climate_trace_ch4.csv')
ct = ct_raw[ct_raw['gas'] == 'ch4'].copy()
def parse_ct(s):
    p = str(s).split('-')
    return pd.Timestamp(f"{p[0]}-{int(p[1]):02d}-{int(p[2]):02d}", tz='UTC')
ct['date'] = ct['start_time'].apply(parse_ct)
ct = ct.sort_values('date')

# Rybnik
with open(ROOT / 'results_analysis/rybnik_centroid_vs_wind.json') as f:
    ryb = json.load(f)

# ── S2 map crop (2021-09-09 npy — good clear summer image) ──────────────────
# Use the 2021-09-09 tile for the map (it was the requested date; not detected
# but the RGB is fine for geographic context — we're showing location, not signal)
npy_path = ROOT/'data/npy_cache/S2B_MSIL1C_20210909T095029_N0500_R079_T34UCB_20230117T015854.npy'
s2 = np.load(npy_path)

# Crop covering the mine polygon + padding (zoomed out more for label clearance)
# Mine polygon px: rows 1987-2460, cols 6721-8823
PAD_R, PAD_C = 300, 550
r0, r1 = 1987 - PAD_R, 2460 + PAD_R
c0, c1 = 6721 - PAD_C, 8823 + PAD_C
map_crop = s2[r0:r1, c0:c1, :].astype(np.float32)

rgb = np.stack([pct_stretch(map_crop[:,:,3]),   # B04 Red
                pct_stretch(map_crop[:,:,2]),   # B03 Green
                pct_stretch(map_crop[:,:,1])],  # B02 Blue
               axis=-1)

# Key pixel positions relative to crop
def crop_px(lat, lon):
    """Pixel (col-in-crop, row-in-crop) for a lat/lon."""
    x, y = wgs84_to_utm34n(lat, lon)
    gc, gr = utm_to_px(x, y)
    return gc - c0, gr - r0   # col offset, row offset

# Mine polygon corners (in crop-px)
corners_latlon = [(51.2570, 19.0970), (51.2566, 19.3900),
                  (51.2190, 19.3996), (51.2185, 19.0990)]
corners_px = [crop_px(la, lo) for la, lo in corners_latlon]
mine_xs = [p[0] for p in corners_px] + [corners_px[0][0]]
mine_ys = [p[1] for p in corners_px] + [corners_px[0][1]]

# S/C crop box centred at 51.242°N, 19.275°E, 100×100 px at 10m = 1km
mc_x, mc_y = crop_px(51.242, 19.275)
box_half = 50   # 500m radius = 100px × 10m/px
box_x = [mc_x-box_half, mc_x+box_half, mc_x+box_half, mc_x-box_half, mc_x-box_half]
box_y = [mc_y-box_half, mc_y-box_half, mc_y+box_half, mc_y+box_half, mc_y-box_half]

# Detection centroid & ClimateTrace pin
det_x, det_y = crop_px(51.2495, 19.2227)
ct_x,  ct_y  = crop_px(51.2420, 19.2754)

# ── Figure layout ─────────────────────────────────────────────────────────────
# Left column: 3 stacked sub-panels for Panel A
# Right: B (map+bar) and C (seasonal)
fig = plt.figure(figsize=(11.0, 7.5))
gs_root = gridspec.GridSpec(1, 2, width_ratios=[1.55, 1.0],
                             wspace=0.30, left=0.07, right=0.97,
                             top=0.94, bottom=0.08)

# Left column = 3 rows for A1, A2, A3
gs_A = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_root[0],
                                         hspace=0.42,
                                         height_ratios=[1.2, 0.9, 1.0])
axA1 = fig.add_subplot(gs_A[0])
axA2 = fig.add_subplot(gs_A[1])
axA3 = fig.add_subplot(gs_A[2])

# Right column = B (top 60%) + C (bottom 40%)
gs_R = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_root[1],
                                          hspace=0.38, height_ratios=[1.4, 1.0])
# B splits map + bar
gs_B = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_R[0],
                                          height_ratios=[2.5, 1], hspace=0.28)
axB_map = fig.add_subplot(gs_B[0])
axB_bar = fig.add_subplot(gs_B[1])
axC = fig.add_subplot(gs_R[1])

# ══════════════════════════════════════════════════════════════════════════════
# A1 — S/C CFAR timeseries
# ══════════════════════════════════════════════════════════════════════════════
for yr in range(2019, 2026):
    axA1.axvspan(pd.Timestamp(f'{yr}-01-01', tz='UTC'),
                 pd.Timestamp(f'{yr}-03-31', tz='UTC'),
                 alpha=0.09, color=COLORS['winter'], lw=0, zorder=0)
    axA1.axvspan(pd.Timestamp(f'{yr}-10-01', tz='UTC'),
                 pd.Timestamp(f'{yr}-12-31', tz='UTC'),
                 alpha=0.09, color=COLORS['winter'], lw=0, zorder=0)

ab = df['detect'] == True; su = ~ab
axA1.scatter(df.loc[su, 'date'], df.loc[su, 'sc_cfar'],
             facecolors='none', edgecolors=COLORS['sub'], s=12, lw=0.5,
             label='Sub-threshold', zorder=3)
axA1.scatter(df.loc[ab, 'date'], df.loc[ab, 'sc_cfar'],
             color=COLORS['above'], s=14, label='Detection', zorder=4)
axA1.axhline(TAU, color=COLORS['thresh'], ls='--', lw=1.2,
             label=f'τ = {TAU}', zorder=5)

# TROPOMI: vertical tick at date only — not on S/C value scale
tropomi_dt = pd.Timestamp('2021-09-09', tz='UTC')
axA1.axvline(tropomi_dt, color='#AA3399', lw=1.0, ls=':', zorder=6, alpha=0.9)
axA1.text(tropomi_dt + pd.Timedelta(days=35), 1.8e-3,
          'TROPOMI\n+12.7 ppb\n2021-09-09',
          fontsize=4.8, color='#AA3399', va='bottom', ha='left',
          bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, lw=0.4))

# Outlier annotation
axA1.annotate('2019-08-31\nS/C = 802',
              xy=(pd.Timestamp('2019-08-31', tz='UTC'), 802.6),
              xytext=(pd.Timestamp('2021-08-01', tz='UTC'), 200),
              fontsize=5.5, color=COLORS['above'],
              arrowprops=dict(arrowstyle='->', lw=0.7, color=COLORS['above'],
                              connectionstyle='arc3,rad=0.25'), va='center')

axA1.set_yscale('log'); axA1.set_ylim(1e-3, 3000)
axA1.set_ylabel('S/C ratio (CFAR)', fontsize=8)
axA1.set_title('A1. KWB Bełchatów — CH₄Net detection signal',
                fontweight='bold', loc='left', fontsize=8.5)
axA1.xaxis.set_major_locator(mdates.YearLocator())
axA1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axA1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
axA1.tick_params(axis='x', which='minor', length=2)

win_p = mpatches.Patch(color=COLORS['winter'], alpha=0.4, label='Winter (Q1/Q4)')
h1, l1 = axA1.get_legend_handles_labels()
axA1.legend(h1 + [win_p], l1 + ['Winter (Q1/Q4)'],
            fontsize=5.5, loc='upper right', framealpha=0.85,
            ncol=2, handlelength=1.2)

# ══════════════════════════════════════════════════════════════════════════════
# A2 — Emissions comparison: ClimateTrace vs CH4Net quantified
# ══════════════════════════════════════════════════════════════════════════════
# ClimateTrace as step line
ct_vis = ct[ct['date'] >= pd.Timestamp('2021-01-01', tz='UTC')]
axA2.step(ct_vis['date'], ct_vis['emissions_quantity'],
          where='post', color=COLORS['ct'], lw=1.3, alpha=0.85,
          label='ClimateTrace CH₄ inventory', zorder=3)
axA2.fill_between(ct_vis['date'], 0, ct_vis['emissions_quantity'],
                   step='post', color=COLORS['ct'], alpha=0.15, zorder=2)

# CH4Net quantified detections as scatter + error bars (t/month)
# Only show those with CT coverage (2021+)
vis_mask = fd >= pd.Timestamp('2021-01-01', tz='UTC')
if vis_mask.any():
    yerr_lo = (flow_t - flow_t_lo)[vis_mask]
    yerr_hi = (flow_t_hi - flow_t)[vis_mask]
    axA2.errorbar(fd[vis_mask], flow_t[vis_mask],
                  yerr=[yerr_lo, yerr_hi],
                  fmt='o', color=COLORS['quant'], ms=5, lw=0.8, capsize=2,
                  label='CH₄Net quantified (±30%)', zorder=5)

# All quantified detections (pre-2021 too) as lighter background markers
all_vis = fd < pd.Timestamp('2021-01-01', tz='UTC')
if all_vis.any():
    axA2.scatter(fd[all_vis], flow_t[all_vis], marker='o',
                 color=COLORS['quant'], alpha=0.4, s=18, zorder=4,
                 label='CH₄Net quantified (pre-CT)')

axA2.set_ylabel('CH₄ flux (t month⁻¹)', fontsize=8)
axA2.set_title('A2. Monthly CH₄ flux — ClimateTrace inventory vs. CH₄Net detections',
                fontweight='bold', loc='left', fontsize=8.5)
axA2.xaxis.set_major_locator(mdates.YearLocator())
axA2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axA2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
axA2.tick_params(axis='x', which='minor', length=2)
axA2.set_xlim(axA1.get_xlim())
axA2.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'{x/1000:.1f}k' if x >= 1000 else f'{x:.0f}'))
axA2.legend(fontsize=5.5, loc='upper right', framealpha=0.85, handlelength=1.5)
axA2.set_ylim(bottom=0)
axA2.text(0.01, 0.96,
          'CH₄Net values: instantaneous rate × hours in detection month (if emission were continuous)',
          transform=axA2.transAxes, fontsize=5.0, color='#555555', va='top')

# ══════════════════════════════════════════════════════════════════════════════
# A3 — Sentinel-2 RGB map with mine geometry
# ══════════════════════════════════════════════════════════════════════════════
axA3.imshow(rgb, origin='upper', interpolation='bilinear', aspect='equal')

# Mine quantification polygon (orange dashed)
axA3.plot(mine_xs, mine_ys, color='#FF8C00', lw=1.4, ls='--',
          label='Mine quantification polygon (KWB Bełchatów)')

# Corner coordinate labels — offset slightly outside the polygon
corner_labels = [
    # (crop_px col, crop_px row, ha, va, lat_str, lon_str)
    (corners_px[0][0], corners_px[0][1], 'right', 'bottom', '51.257°N', '19.097°E'),  # NW
    (corners_px[1][0], corners_px[1][1], 'left',  'bottom', '51.257°N', '19.390°E'),  # NE
    (corners_px[2][0], corners_px[2][1], 'left',  'top',    '51.219°N', '19.400°E'),  # SE
    (corners_px[3][0], corners_px[3][1], 'right', 'top',    '51.219°N', '19.099°E'),  # SW
]
for cx, cy, ha, va, lat_s, lon_s in corner_labels:
    ox = -6 if ha == 'right' else 6
    oy = -4 if va == 'bottom' else 4
    axA3.text(cx + ox, cy + oy, f'{lat_s}\n{lon_s}',
              ha=ha, va=va, fontsize=4.2, color='#FF8C00',
              fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55, lw=0))

# S/C analysis crop box (blue solid)
axA3.plot(box_x, box_y, color=COLORS['above'], lw=1.4, ls='-',
          label='S/C site crop (1 km × 1 km)')

# Detection centroid (red star)
axA3.plot(det_x, det_y, '*', color='#CC0000', ms=9, zorder=6,
          label='CH₄Net detection centroid')

# ClimateTrace pin
axA3.plot(ct_x, ct_y, 's', color=COLORS['ct'], ms=6, zorder=6,
          label='ClimateTrace source pin')

# Scale bar — 500 px = 5 km at 10m/px
sh = rgb.shape
sb_x0, sb_x1 = sh[1]*0.03, sh[1]*0.03 + 500
sb_y = sh[0]*0.93
axA3.plot([sb_x0, sb_x1], [sb_y, sb_y], 'w-', lw=2.5)
axA3.text((sb_x0+sb_x1)/2, sb_y - 12, '5 km',
          ha='center', va='bottom', fontsize=6, color='white', fontweight='bold')

axA3.set_xticks([]); axA3.set_yticks([])
axA3.set_title('A3. Sentinel-2 RGB  ·  2021-09-09  ·  T34UCB  ·  mine geometry overlay',
                fontweight='bold', loc='left', fontsize=8.0, pad=4)

# N arrow (top-right)
axA3.annotate('N', xy=(0.975, 0.90), xytext=(0.975, 0.80),
              xycoords='axes fraction', textcoords='axes fraction',
              ha='center', fontsize=7, fontweight='bold', color='white',
              arrowprops=dict(arrowstyle='->', color='white', lw=1.2))

# Legend bottom-LEFT to keep SE corner clear
axA3.legend(fontsize=5.5, loc='lower left', framealpha=0.88,
            handlelength=1.5, markerscale=0.85,
            facecolor='#111111', labelcolor='white',
            edgecolor='#444444')

# ── Control-crop locator inset — geographic map style, fully in top-right corner ──
# Uses real lat/lon coordinates with graticule grid (looks like a map, no tiles needed)
# Site: 51.242°N, 19.275°E; offsets N/S=0.20°, E=0.30°, W=0.39°
site_lat, site_lon = 51.242, 19.275
ctrl_offsets = {
    'N': (site_lat + 0.20, site_lon),
    'S': (site_lat - 0.20, site_lon),
    'E': (site_lat,         site_lon + 0.30),
    'W': (site_lat,         site_lon - 0.39),
}
# Lon/lat bounds for inset
lon_min, lon_max = site_lon - 0.50, site_lon + 0.41
lat_min, lat_max = site_lat - 0.27, site_lat + 0.27

# Inset fully in top-right corner
ax_cc = axA3.inset_axes([0.70, 0.52, 0.295, 0.465])
ax_cc.set_facecolor('#EDF4FB')   # light blue-grey — "water/terrain" feel

# Draw graticule grid (0.1° spacing)
import numpy as _np
for glon in _np.arange(math.floor(lon_min*10)/10, lon_max, 0.1):
    ax_cc.axvline(glon, color='#C8D8E8', lw=0.4, zorder=1)
for glat in _np.arange(math.floor(lat_min*10)/10, lat_max, 0.1):
    ax_cc.axhline(glat, color='#C8D8E8', lw=0.4, zorder=1)

# Terrain fill (very light green to suggest land)
ax_cc.set_facecolor('#EEF2E8')

# Control crop boxes — 1km ≈ 0.009° lat, 0.014° lon at 51°N
box_dlat = 0.0045   # half-box in lat
box_dlon = 0.0067   # half-box in lon
for name, (clat, clon) in ctrl_offsets.items():
    rect = mpatches.Rectangle(
        (clon - box_dlon, clat - box_dlat), 2*box_dlon, 2*box_dlat,
        edgecolor='#2255CC', facecolor='#99BBFF', alpha=0.6, lw=0.9,
        linestyle='--', zorder=4)
    ax_cc.add_patch(rect)
    # label outside the box
    lbl_offsets = {'N': (0, 0.015), 'S': (0, -0.018),
                   'E': (0.025, 0), 'W': (-0.025, 0)}
    dx, dy = lbl_offsets[name]
    ax_cc.text(clon + dx, clat + dy, f'Ctrl {name}',
               ha='center', va='center', fontsize=3.8,
               color='#1133AA', fontweight='bold', zorder=5)

# Site crop box (solid blue, smaller)
site_rect = mpatches.Rectangle(
    (site_lon - box_dlon, site_lat - box_dlat), 2*box_dlon, 2*box_dlat,
    edgecolor=COLORS['above'], facecolor='#88BBFF', alpha=0.9, lw=1.2, zorder=5)
ax_cc.add_patch(site_rect)
ax_cc.text(site_lon, site_lat, 'KWB\nBełchatów', ha='center', va='center',
           fontsize=3.5, color='#002255', fontweight='bold', zorder=6)

# Dotted lines from site to each control
for name, (clat, clon) in ctrl_offsets.items():
    ax_cc.plot([site_lon, clon], [site_lat, clat],
               color='#888888', lw=0.5, ls=':', zorder=2)

# Axis ticks: 0.2° spacing
ax_cc.set_xlim(lon_min, lon_max)
ax_cc.set_ylim(lat_min, lat_max)
ax_cc.set_xticks([19.0, 19.2, 19.4, 19.6])
ax_cc.set_yticks([51.0, 51.2, 51.4])
ax_cc.tick_params(labelsize=3.5, pad=1, length=2, colors='#333333')
ax_cc.set_xlabel('Lon (°E)', fontsize=3.8, labelpad=1)
ax_cc.set_ylabel('Lat (°N)', fontsize=3.8, labelpad=1)
for sp in ax_cc.spines.values():
    sp.set_edgecolor('#AABBCC'); sp.set_linewidth(0.7)
ax_cc.set_title('S/C control crop layout', fontsize=4.2, pad=2,
                 color='#222222',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, lw=0.4))

# ══════════════════════════════════════════════════════════════════════════════
# B — Rybnik (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
pin_lon, pin_lat = ryb['cm_pin']['lon'], ryb['cm_pin']['lat']
cen_lon, cen_lat = ryb['centroid']['centroid_lon'], ryb['centroid']['centroid_lat']
u_e, v_e = ryb['era5_wind']['era5_u_ms'], ryb['era5_wind']['era5_v_ms']

def deg_to_km(dlon, dlat, ref_lat):
    return (dlon*111.32*math.cos(math.radians(ref_lat)), dlat*111.32)

dx_cen, dy_cen = deg_to_km(cen_lon-pin_lon, cen_lat-pin_lat, pin_lat)

axB_map.set_facecolor('#F0F0F0')
axB_map.axhline(0, color='white', lw=0.5); axB_map.axvline(0, color='white', lw=0.5)
axB_map.plot(0, 0, 'k^', ms=6, label='Mine pin', zorder=5)
axB_map.plot(dx_cen, dy_cen, 'r*', ms=8, label='CH₄Net centroid', zorder=5)
ws = 0.30
axB_map.quiver(0, 0, u_e*ws, v_e*ws, color='#E69F00', scale_units='xy', scale=1,
               width=0.04, label=f'ERA5 wind ({ryb["era5_wind"]["wind_speed_ms"]:.1f} m/s)',
               zorder=4)
dir_cm = ryb['cm_wind_one_day_prior']['wind_dir_deg']
spd_cm = ryb['cm_wind_one_day_prior']['wind_speed_ms']
u_cm = -spd_cm*math.sin(math.radians(dir_cm))
v_cm = -spd_cm*math.cos(math.radians(dir_cm))
axB_map.quiver(0, 0, u_cm*ws, v_cm*ws, color='#CC79A7', scale_units='xy', scale=1,
               width=0.04, label='Wind D-1 (172° reversal)', zorder=4, alpha=0.85)
axB_map.annotate('', xy=(dx_cen, dy_cen), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', lw=0.8, color='#555555', ls='dashed'))
axB_map.text(dx_cen/2-0.05, dy_cen/2+0.12, f'{ryb["displacement_m"]/1000:.1f} km',
             fontsize=5.5, ha='center', color='#333333')
m = 2.3
axB_map.set_xlim(-m, m); axB_map.set_ylim(-m, m)
axB_map.set_xlabel('Δ lon (km)', fontsize=7); axB_map.set_ylabel('Δ lat (km)', fontsize=7)
axB_map.tick_params(labelsize=6)
axB_map.set_title('B. Rybnik CM — centroid displacement vs. wind',
                   fontweight='bold', loc='left', fontsize=8.5)
axB_map.text(0.02, 0.04, 'TROPOMI: no confirmed enhancement\n(Silesian industrial fringe)',
             transform=axB_map.transAxes, fontsize=5.0, va='bottom',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, lw=0.4))
axB_map.legend(fontsize=5.2, loc='upper right', framealpha=0.85, handlelength=1.2)

axB_bar.bar(['Raw S/C', 'CFAR'], [5.48, 1.48],
            color=['#999999', '#CCCCCC'], edgecolor='white', lw=0.5)
axB_bar.axhline(TAU, color=COLORS['thresh'], ls='--', lw=1.0)
axB_bar.set_ylabel('S/C ratio', fontsize=7)
axB_bar.set_ylim(0, 7)
axB_bar.tick_params(labelsize=6.5)
axB_bar.text(0.97, TAU+0.12, f'τ = {TAU:.2f}', ha='right', fontsize=5.2,
             color=COLORS['thresh'], transform=axB_bar.get_yaxis_transform())
axB_bar.text(1, 1.65, 'Suppressed\n(CFAR)', ha='center', fontsize=5.0, color='#555555')

# ══════════════════════════════════════════════════════════════════════════════
# C — Seasonal detection counts (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
all_q = df['quarter'].value_counts()
det_q = df[df['detect']]['quarter'].value_counts()
q_lab   = ['Q1\n(Jan–Mar)', 'Q2\n(Apr–Jun)', 'Q3\n(Jul–Sep)', 'Q4\n(Oct–Dec)']
q_dets  = [det_q.get(i, 0) for i in range(1, 5)]
q_total = [all_q.get(i, 0) for i in range(1, 5)]
bar_c = [COLORS['winter'], COLORS['above'], COLORS['above'], COLORS['winter']]
bar_a = [0.55, 0.90, 0.90, 0.55]
for i in range(4):
    axC.bar(i, q_dets[i], color=bar_c[i], alpha=bar_a[i], width=0.72, zorder=2)
axC.bar(range(4), q_total, color='none', edgecolor='#555555', lw=0.8, width=0.72, zorder=3)
axC.set_xticks(range(4)); axC.set_xticklabels(q_lab, fontsize=7)
axC.set_ylabel('# overpasses', fontsize=8)
axC.set_title('C. Structural Ceiling\n(seasonal detection rates)',
               fontweight='bold', loc='left', fontsize=8.5)
p1 = mpatches.Patch(color=COLORS['above'], alpha=0.9, label='Detections (Q2/Q3)')
p2 = mpatches.Patch(color=COLORS['winter'], alpha=0.55, label='Winter (Q1/Q4)')
p3 = mpatches.Patch(fc='none', ec='#555555', lw=0.8, label='Total obs.')
axC.legend(handles=[p1, p2, p3], fontsize=6, loc='upper right', framealpha=0.85)

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path('/sessions/brave-gallant-mayer/mnt/Downloads/figure_timeseries_enhanced.png')
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig('figure_timeseries_enhanced.png', dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {out}')
plt.close()
