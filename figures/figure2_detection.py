"""
Figure 2 — CH₄Net detection evidence  (layout v3)

  LEFT — KWB Bełchatów
    Top row:  A (S/C CFAR scatter, ~75% width)  |  E (seasonal ceiling, ~25%)
    Bot row:  B (mine RGB detail)  |  C (wide S2 control-crop context)

  RIGHT — Rybnik CM
    D  : centroid displacement vs. ERA5 wind
    D2 : CFAR suppression bar
"""
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
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
TAU  = 3.5796
ROOT = Path(__file__).parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────
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
    return (utm_x - origin_x) / res, (origin_y - utm_y) / res

def pct_stretch(arr, lo=2, hi=98):
    p_lo, p_hi = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)

# ── data ──────────────────────────────────────────────────────────────────────
with open(ROOT / 'results_analysis/belchatow_annual_timeseries_mbsp.json') as f:
    bel_data = json.load(f)

dates, sc_scores, is_detect, partial = [], [], [], []
for r in bel_data['records']:
    dt = r.get('acquisition_date') or r.get('search', {}).get('acquisition_date')
    if not dt: continue
    det = r.get('detection', {})
    if 'sc_cfar' not in det: continue
    dates.append(dt); sc_scores.append(det['sc_cfar'])
    is_detect.append(det.get('cfar_detect', False))
    partial.append(r.get('partial_swath', False))

df = pd.DataFrame({'date': dates, 'sc_cfar': sc_scores,
                   'detect': is_detect, 'partial': partial})
df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
df = df[~df['partial']].copy()
df['quarter'] = df['date'].dt.quarter

last_date = df['date'].max()
xlim_l = pd.Timestamp('2018-10-01', tz='UTC')
xlim_r = last_date + pd.Timedelta(days=60)

with open(ROOT / 'results_analysis/rybnik_centroid_vs_wind.json') as f:
    ryb = json.load(f)

# ── S2 imagery ─────────────────────────────────────────────────────────────────
print("Loading S2 tile …")
npy_path = (ROOT / 'data/npy_cache/'
            'S2B_MSIL1C_20210909T095029_N0500_R079_T34UCB_20230117T015854.npy')
s2 = np.load(npy_path)
print(f"  shape: {s2.shape}")

# Panel B — square-ish mine detail crop
# Mine polygon: rows ~1987-2460 (473 tall), cols ~6721-8823 (2102 wide)
# Add large vertical padding + minimal horizontal padding → ~4:3 (H:W) crop
PAD_R, PAD_C = 620, 60
r0b, r1b = 1987 - PAD_R, 2460 + PAD_R   # rows 1367–3080  → 1713 tall
c0b, c1b = 6721 - PAD_C, 8823 + PAD_C   # cols 6661–8883  → 2222 wide
# Aspect (H:W) = 1713/2222 = 0.77  ≈ 4:3  ✓
map_crop = s2[r0b:r1b, c0b:c1b, :].astype(np.float32)
rgb_B = np.stack([pct_stretch(map_crop[:,:,3]),
                  pct_stretch(map_crop[:,:,2]),
                  pct_stretch(map_crop[:,:,1])], axis=-1)

def crop_px_B(lat, lon):
    x, y = wgs84_to_utm34n(lat, lon)
    gc, gr = utm_to_px(x, y)
    return gc - c0b, gr - r0b

# Panel C — wide context crop (rows 0-4700, cols 4900-10300)
CROP_R0, CROP_R1 = 0,    4700
CROP_C0, CROP_C1 = 4900, 10300
DS = 5
print("Extracting wide crop …")
wide_raw = s2[CROP_R0:CROP_R1, CROP_C0:CROP_C1, :].astype(np.float32)
rgb_C    = np.stack([pct_stretch(wide_raw[:,:,3]),
                     pct_stretch(wide_raw[:,:,2]),
                     pct_stretch(wide_raw[:,:,1])], axis=-1)
rgb_C_ds = rgb_C[::DS, ::DS]
C_H, C_W = rgb_C_ds.shape[:2]
print(f"  wide-crop display: {C_W}×{C_H}")

def crop_px_C(lat, lon):
    x, y = wgs84_to_utm34n(lat, lon)
    gc, gr = utm_to_px(x, y)
    return (gc - CROP_C0) / DS, (gr - CROP_R0) / DS

# ── geometry ──────────────────────────────────────────────────────────────────
corners_latlon = [(51.2570, 19.0970), (51.2566, 19.3900),
                  (51.2190, 19.3996), (51.2185, 19.0990)]

# Panel B
corners_px_B = [crop_px_B(la, lo) for la, lo in corners_latlon]
mine_xs_B    = [p[0] for p in corners_px_B] + [corners_px_B[0][0]]
mine_ys_B    = [p[1] for p in corners_px_B] + [corners_px_B[0][1]]
mc_x_B, mc_y_B = crop_px_B(51.242, 19.275)
BH = 50
box_x_B = [mc_x_B-BH, mc_x_B+BH, mc_x_B+BH, mc_x_B-BH, mc_x_B-BH]
box_y_B = [mc_y_B-BH, mc_y_B-BH, mc_y_B+BH, mc_y_B+BH, mc_y_B-BH]
det_x_B, det_y_B = crop_px_B(51.2495, 19.2227)
ct_x_B,  ct_y_B  = crop_px_B(51.2420, 19.2754)

# Panel C
corners_px_C = [crop_px_C(la, lo) for la, lo in corners_latlon]
mine_xs_C    = [p[0] for p in corners_px_C] + [corners_px_C[0][0]]
mine_ys_C    = [p[1] for p in corners_px_C] + [corners_px_C[0][1]]
mine_cen_x_C = sum(p[0] for p in corners_px_C) / 4
mine_cen_y_C = sum(p[1] for p in corners_px_C) / 4

site_lat, site_lon = 51.242, 19.275
site_px_C          = crop_px_C(site_lat, site_lon)
sx, sy             = crop_px_C(51.2495, 19.2227)   # detection centroid

ctrl_centres = {
    'N': (site_lat + 0.20, site_lon),
    'S': (site_lat - 0.20, site_lon),
    'E': (site_lat,         site_lon + 0.30),
    'W': (site_lat,         site_lon - 0.39),
}
ctrl_px_C     = {k: crop_px_C(la, lo) for k, (la, lo) in ctrl_centres.items()}
BOX_HALF_C    = 1000 / (10 * DS)   # 20 display px = 1 km half-side
ctrl_dist_str = {'N': '22 km N', 'S': '22 km S', 'E': '33 km E', 'W': '43 km W'}

towns = {'Bełchatów': (51.361, 19.360), 'Kamieńsk': (51.216, 19.491),
         'Radomsko':  (51.062, 19.450), 'Pajęczno': (51.157, 18.964)}
town_px_C = {nm: crop_px_C(la, lo) for nm, (la, lo) in towns.items()}

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.0, 9.2))

# Two main columns: Bełchatów (left, wider) | Rybnik (right)
gs_main = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[3.0, 1],
    left=0.05, right=0.98, top=0.91, bottom=0.06,
    wspace=0.26)

# ── Bełchatów column ──────────────────────────────────────────────────────────
gs_bel = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_main[0],
    height_ratios=[1.0, 1.35], hspace=0.30)

# Top: A (scatter) + E (seasonal) side by side
gs_bel_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_bel[0],
    width_ratios=[2.8, 1.0], wspace=0.22)
axA = fig.add_subplot(gs_bel_top[0])
axE = fig.add_subplot(gs_bel_top[1])

# Bottom: B (mine detail) + C (wide context)
gs_bel_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_bel[1],
    width_ratios=[1.0, 1.3], wspace=0.05)
axB = fig.add_subplot(gs_bel_bot[0])
axC = fig.add_subplot(gs_bel_bot[1])

# ── Rybnik column ─────────────────────────────────────────────────────────────
gs_ryb = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_main[1],
    height_ratios=[2.0, 0.9], hspace=0.36)
axD     = fig.add_subplot(gs_ryb[0])
axD_bar = fig.add_subplot(gs_ryb[1])

# Section header labels — anchored to figure fraction coordinates
# Bełchatów spans ~left 0.05 to ~0.77 of figure; Rybnik ~0.79 to 0.98
fig.text(0.40, 0.955, 'KWB Bełchatów',
         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1A1A1A',
         bbox=dict(boxstyle='round,pad=0.30', fc='#EEF3FA', ec='#AABBDD', lw=0.8))
fig.text(0.895, 0.955, 'Rybnik CM',
         ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1A1A1A',
         bbox=dict(boxstyle='round,pad=0.30', fc='#FAF0EE', ec='#DDAABB', lw=0.8))

# ══════════════════════════════════════════════════════════════════════════════
# A — S/C CFAR time series (Bełchatów)
# ══════════════════════════════════════════════════════════════════════════════
for yr in range(2019, 2026):
    axA.axvspan(pd.Timestamp(f'{yr}-01-01', tz='UTC'),
                pd.Timestamp(f'{yr}-03-31', tz='UTC'),
                alpha=0.09, color=COLORS['winter'], lw=0, zorder=0)
    axA.axvspan(pd.Timestamp(f'{yr}-10-01', tz='UTC'),
                pd.Timestamp(f'{yr}-12-31', tz='UTC'),
                alpha=0.09, color=COLORS['winter'], lw=0, zorder=0)

ab = df['detect'] == True; su = ~ab
axA.scatter(df.loc[su, 'date'], df.loc[su, 'sc_cfar'],
            facecolors='none', edgecolors=COLORS['sub'], s=12, lw=0.5,
            label='Sub-threshold', zorder=3)
axA.scatter(df.loc[ab, 'date'], df.loc[ab, 'sc_cfar'],
            color=COLORS['above'], s=15, label='Detection', zorder=4)
axA.axhline(TAU, color=COLORS['thresh'], ls='--', lw=1.2,
            label=f'CFAR threshold τ = {TAU}', zorder=5)

tropomi_dt = pd.Timestamp('2021-09-09', tz='UTC')
axA.axvline(tropomi_dt, color='#AA3399', lw=1.0, ls=':', zorder=6, alpha=0.9)
# Annotation to the LEFT of the line so it doesn't crowd the right side
axA.text(tropomi_dt - pd.Timedelta(days=38), 1.5e-3,
         'TROPOMI\n+12.7 ppb\n2021-09-09',
         fontsize=4.5, color='#AA3399', va='bottom', ha='right',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, lw=0.4))

axA.annotate('2019-08-31\nS/C = 802',
             xy=(pd.Timestamp('2019-08-31', tz='UTC'), 802.6),
             xytext=(pd.Timestamp('2021-06-01', tz='UTC'), 200),
             fontsize=5.5, color=COLORS['above'],
             arrowprops=dict(arrowstyle='->', lw=0.7, color=COLORS['above'],
                             connectionstyle='arc3,rad=0.22'), va='center')

axA.set_yscale('log'); axA.set_ylim(1e-3, 3000)
axA.set_xlim(xlim_l, xlim_r)
axA.set_ylabel('S/C ratio (CFAR-normalised)', fontsize=8)
axA.set_title('A.  CH₄Net detection signal', fontweight='bold', loc='left', fontsize=8.5)
axA.xaxis.set_major_locator(mdates.YearLocator())
axA.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axA.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
axA.tick_params(axis='x', which='minor', length=2)

win_p = mpatches.Patch(color=COLORS['winter'], alpha=0.4, label='Winter (Q1/Q4)')
h1, l1 = axA.get_legend_handles_labels()
axA.legend(h1 + [win_p], l1 + ['Winter (Q1/Q4)'],
           fontsize=5.5, loc='lower left', framealpha=0.85,
           ncol=2, handlelength=1.2)

# ══════════════════════════════════════════════════════════════════════════════
# E — Seasonal detection ceiling  (Bełchatów only — clearly labelled)
# ══════════════════════════════════════════════════════════════════════════════
all_q = df['quarter'].value_counts()
det_q = df[df['detect']]['quarter'].value_counts()
q_lab   = ['Q1', 'Q2', 'Q3', 'Q4']
q_dets  = [det_q.get(i, 0) for i in range(1, 5)]
q_total = [all_q.get(i, 0) for i in range(1, 5)]
bar_c   = [COLORS['winter'], COLORS['above'], COLORS['above'], COLORS['winter']]
bar_a   = [0.55, 0.90, 0.90, 0.55]
for i in range(4):
    axE.bar(i, q_dets[i], color=bar_c[i], alpha=bar_a[i], width=0.72, zorder=2)
axE.bar(range(4), q_total, color='none', edgecolor='#555555', lw=0.8, width=0.72, zorder=3)
axE.set_xticks(range(4)); axE.set_xticklabels(q_lab, fontsize=7)
axE.set_ylabel('# overpasses', fontsize=7)
axE.set_title('E.  Seasonal ceiling\n(KWB Bełchatów)',
               fontweight='bold', loc='left', fontsize=7.5)
p1 = mpatches.Patch(color=COLORS['above'], alpha=0.9,   label='Detections')
p2 = mpatches.Patch(color=COLORS['winter'], alpha=0.55, label='Winter')
p3 = mpatches.Patch(fc='none', ec='#555555', lw=0.8,   label='All obs.')
axE.legend(handles=[p1, p2, p3], fontsize=5.0, loc='upper right', framealpha=0.85)
# Note that seasonal data is Bełchatów only; no comparable Rybnik quarterly data
axE.text(0.03, 0.03, 'Bełchatów site only\n(no Rybnik quarterly data)',
         transform=axE.transAxes, fontsize=4.5, color='#666666', va='bottom')

# ══════════════════════════════════════════════════════════════════════════════
# B — Mine S2 RGB detail  (restored undistorted aspect ratio)
# ══════════════════════════════════════════════════════════════════════════════
axB.imshow(rgb_B, origin='upper', interpolation='bilinear', aspect='equal')

axB.plot(mine_xs_B, mine_ys_B, color='#FF8C00', lw=1.5, ls='--', zorder=5,
         label='Mine polygon (KWB Bełchatów)')

corner_labels_B = [
    (corners_px_B[0][0], corners_px_B[0][1], 'right', 'bottom', '51.257°N', '19.097°E'),
    (corners_px_B[1][0], corners_px_B[1][1], 'left',  'bottom', '51.257°N', '19.390°E'),
    (corners_px_B[2][0], corners_px_B[2][1], 'left',  'top',    '51.219°N', '19.400°E'),
    (corners_px_B[3][0], corners_px_B[3][1], 'right', 'top',    '51.219°N', '19.099°E'),
]
for cx, cy, ha, va, lat_s, lon_s in corner_labels_B:
    ox = -5 if ha == 'right' else 5
    oy = -4 if va == 'bottom' else 4
    axB.text(cx + ox, cy + oy, f'{lat_s}\n{lon_s}',
             ha=ha, va=va, fontsize=3.8, color='#FF8C00', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.12', fc='black', alpha=0.55, lw=0))

axB.plot(box_x_B, box_y_B, color=COLORS['above'], lw=1.5, ls='-', zorder=5,
         label='S/C site crop (1 km²)')
axB.plot(det_x_B, det_y_B, '*', color='#CC0000', ms=9, zorder=6,
         label='Detection centroid')
axB.plot(ct_x_B, ct_y_B, 's', color=COLORS['ct'], ms=6, zorder=6,
         label='ClimateTrace source pin')

# Scale bar: 500 tile px = 5 km
sh = rgb_B.shape
sb_x0, sb_x1 = sh[1]*0.03, sh[1]*0.03 + 500
sb_y = sh[0] * 0.95
axB.plot([sb_x0, sb_x1], [sb_y, sb_y], 'w-', lw=2.5)
axB.text((sb_x0+sb_x1)/2, sb_y - 18, '5 km',
         ha='center', va='bottom', fontsize=6, color='white', fontweight='bold')

axB.set_xticks([]); axB.set_yticks([])
axB.set_title('B.  Sentinel-2 RGB  ·  2021-09-09',
               fontweight='bold', loc='left', fontsize=8.0, pad=4)
axB.annotate('N', xy=(0.97, 0.93), xytext=(0.97, 0.86),
             xycoords='axes fraction', textcoords='axes fraction',
             ha='center', fontsize=7, fontweight='bold', color='white',
             arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
axB.legend(fontsize=4.8, loc='lower left', framealpha=0.88,
           handlelength=1.4, markerscale=0.8,
           facecolor='#111111', labelcolor='white', edgecolor='#444444')

# ══════════════════════════════════════════════════════════════════════════════
# C — Wide S2 context: mine polygon + control-crop boxes
# ══════════════════════════════════════════════════════════════════════════════
axC.imshow(rgb_C_ds, origin='upper', interpolation='bilinear', aspect='auto',
           extent=[0, C_W, C_H, 0])

# Mine polygon
axC.plot(mine_xs_C, mine_ys_C, color='#FF8C00', lw=2.0, ls='--', zorder=5)
axC.text(mine_cen_x_C, mine_cen_y_C, 'KWB\nBełchatów',
         ha='center', va='center', fontsize=5.5, color='#FF8C00', fontweight='bold',
         zorder=7, bbox=dict(boxstyle='round,pad=0.2', fc='#00000077', lw=0))

# S/C site indicator — filled semi-transparent square at detection centroid,
# clearly distinct from the hollow-dashed control boxes
site_box = mpatches.Rectangle(
    (sx - BOX_HALF_C, sy - BOX_HALF_C), 2*BOX_HALF_C, 2*BOX_HALF_C,
    edgecolor=COLORS['above'], facecolor=COLORS['above'],
    alpha=0.40, lw=2.2, zorder=6)
axC.add_patch(site_box)
axC.text(sx, sy + BOX_HALF_C + 10, 'S/C site',
         ha='center', va='top', fontsize=4.5, color='#88CCFF',
         fontweight='bold', zorder=8)

# Control boxes + connector lines
ctrl_lbl_off = {'N': (0, -30), 'S': (0, 30), 'E': (36, 0), 'W': (-36, 0)}
for name, (cx_c, cy_c) in ctrl_px_C.items():
    axC.plot([sx, cx_c], [sy, cy_c], color='white', lw=0.8, ls=':', zorder=3, alpha=0.55)

    if name == 'N' and cy_c < 0:
        # Partial U-box: left side, bottom edge, right side (no top — outside tile)
        bottom = cy_c + BOX_HALF_C
        left   = cx_c - BOX_HALF_C
        right  = cx_c + BOX_HALF_C
        for xs, ys in [([left, left], [0, bottom]),
                       ([right, right], [0, bottom]),
                       ([left, right], [bottom, bottom])]:
            axC.plot(xs, ys, color='#66AAFF', lw=1.6, ls='--', zorder=5)
        axC.annotate('', xy=(cx_c, 1), xytext=(cx_c, bottom + 10),
                     arrowprops=dict(arrowstyle='->', color='#66AAFF', lw=1.3))
        lbl_y = bottom + 32
    else:
        axC.add_patch(mpatches.Rectangle(
            (cx_c - BOX_HALF_C, cy_c - BOX_HALF_C),
            2*BOX_HALF_C, 2*BOX_HALF_C,
            edgecolor='#66AAFF', facecolor='none', lw=1.6, ls='--', zorder=5))
        lbl_y = np.clip(cy_c + ctrl_lbl_off[name][1], 8, C_H - 8)

    lbl_x = np.clip(cx_c + ctrl_lbl_off[name][0], 8, C_W - 8)
    axC.text(lbl_x, lbl_y, f'Ctrl {name}\n({ctrl_dist_str[name]})',
             ha='center', va='center', fontsize=5.0, color='white', fontweight='bold',
             zorder=7,
             bbox=dict(boxstyle='round,pad=0.22', fc='#00336699', lw=0.8, ec='#66AAFF'))

# Town labels
town_ha  = {'Bełchatów': 'right', 'Kamieńsk': 'left',
            'Radomsko': 'center', 'Pajęczno': 'center'}
town_off = {'Bełchatów': (-6, 0), 'Kamieńsk': (6, 0),
            'Radomsko': (0, 9),   'Pajęczno': (0, 9)}
for nm, (tx, ty) in town_px_C.items():
    if not (8 < tx < C_W - 8 and 8 < ty < C_H - 8):
        continue
    axC.plot(tx, ty, 'o', ms=3.5, color='white',
             markeredgecolor='#888888', markeredgewidth=0.5, zorder=6)
    dx_t, dy_t = town_off[nm]
    axC.text(tx + dx_t, ty + dy_t, nm, ha=town_ha[nm], va='center',
             fontsize=5.2, color='white', fontweight='bold', zorder=7,
             bbox=dict(boxstyle='round,pad=0.15', fc='#00000066', lw=0))

# Scale bar: 10 km = 200 display px
sb_x0c = C_W * 0.04; sb_x1c = sb_x0c + 200; sb_yc = C_H * 0.96
axC.plot([sb_x0c, sb_x1c], [sb_yc, sb_yc], 'w-', lw=3.0)
axC.text((sb_x0c+sb_x1c)/2, sb_yc - 9, '10 km',
         ha='center', va='bottom', fontsize=6.5, color='white', fontweight='bold')
axC.annotate('N', xy=(0.97, 0.945), xytext=(0.97, 0.875),
             xycoords='axes fraction', textcoords='axes fraction',
             ha='center', fontsize=8, fontweight='bold', color='white',
             arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

axC.set_xlim(0, C_W); axC.set_ylim(C_H, 0)
axC.set_xticks([]); axC.set_yticks([])
axC.set_title('C.  S/C control-crop layout  ·  2021-09-09',
               fontweight='bold', loc='left', fontsize=8.0, pad=4)

h_mine = Line2D([0],[0], color='#FF8C00', lw=1.8, ls='--', label='Mine polygon')
h_site = mpatches.Patch(facecolor=COLORS['above'], alpha=0.45, label='S/C site crop')
h_ctrl = Line2D([0],[0], color='#66AAFF', lw=1.4, ls='--', label='Control crops (4×)')
axC.legend(handles=[h_mine, h_site, h_ctrl], fontsize=5.5, loc='lower right',
           framealpha=0.90, handlelength=1.8,
           facecolor='#111111', labelcolor='white', edgecolor='#444444')

# ══════════════════════════════════════════════════════════════════════════════
# D — Rybnik centroid displacement vs. ERA5 wind
# ══════════════════════════════════════════════════════════════════════════════
pin_lon, pin_lat = ryb['cm_pin']['lon'], ryb['cm_pin']['lat']
cen_lon, cen_lat = ryb['centroid']['centroid_lon'], ryb['centroid']['centroid_lat']
u_e, v_e = ryb['era5_wind']['era5_u_ms'], ryb['era5_wind']['era5_v_ms']

def deg_to_km(dlon, dlat, ref_lat):
    return (dlon * 111.32 * math.cos(math.radians(ref_lat)), dlat * 111.32)

dx_cen, dy_cen = deg_to_km(cen_lon - pin_lon, cen_lat - pin_lat, pin_lat)

axD.set_facecolor('#F0F0F0')
axD.axhline(0, color='white', lw=0.5); axD.axvline(0, color='white', lw=0.5)
axD.plot(0, 0, 'k^', ms=7, label='Mine pin', zorder=5)
axD.plot(dx_cen, dy_cen, 'r*', ms=9, label='CH₄Net centroid', zorder=5)
ws = 0.30
axD.quiver(0, 0, u_e*ws, v_e*ws, color='#E69F00', scale_units='xy', scale=1,
           width=0.04, zorder=4,
           label=f'ERA5 wind ({ryb["era5_wind"]["wind_speed_ms"]:.1f} m/s)')
dir_cm = ryb['cm_wind_one_day_prior']['wind_dir_deg']
spd_cm = ryb['cm_wind_one_day_prior']['wind_speed_ms']
u_cm   = -spd_cm * math.sin(math.radians(dir_cm))
v_cm   = -spd_cm * math.cos(math.radians(dir_cm))
axD.quiver(0, 0, u_cm*ws, v_cm*ws, color='#CC79A7', scale_units='xy', scale=1,
           width=0.04, zorder=4, alpha=0.85, label='Wind D−1 (172° reversal)')
axD.annotate('', xy=(dx_cen, dy_cen), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', lw=0.9, color='#555555', ls='dashed'))
axD.text(dx_cen/2 - 0.07, dy_cen/2 + 0.14, f'{ryb["displacement_m"]/1000:.1f} km',
         fontsize=5.5, ha='center', color='#333333')
m = 2.3
axD.set_xlim(-m, m); axD.set_ylim(-m, m)
axD.set_xlabel('Δ lon (km)', fontsize=7); axD.set_ylabel('Δ lat (km)', fontsize=7)
axD.tick_params(labelsize=6)
axD.set_title('D.  Centroid displacement vs. wind',
               fontweight='bold', loc='left', fontsize=8.0)
axD.text(0.02, 0.04, 'No TROPOMI hit\n(Silesian industrial fringe)',
         transform=axD.transAxes, fontsize=4.8, va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85, lw=0.4))
axD.legend(fontsize=4.8, loc='upper right', framealpha=0.85, handlelength=1.2)

# ══════════════════════════════════════════════════════════════════════════════
# D2 — Rybnik CFAR suppression bar
# ══════════════════════════════════════════════════════════════════════════════
axD_bar.bar(['Raw S/C', 'CFAR'], [5.48, 1.48],
            color=['#999999', '#CCCCCC'], edgecolor='white', lw=0.5)
axD_bar.axhline(TAU, color=COLORS['thresh'], ls='--', lw=1.0)
axD_bar.set_ylabel('S/C ratio', fontsize=7)
axD_bar.set_ylim(0, 7)
axD_bar.tick_params(labelsize=6)
axD_bar.text(0.97, TAU + 0.15, f'τ = {TAU:.2f}', ha='right', fontsize=5.0,
             color=COLORS['thresh'], transform=axD_bar.get_yaxis_transform())
axD_bar.text(1, 1.65, 'Suppressed\n(CFAR)', ha='center', fontsize=4.8, color='#555555')
axD_bar.set_title('D2.  CFAR suppression',
                   fontweight='bold', loc='left', fontsize=7.5, pad=2)

# ── save ──────────────────────────────────────────────────────────────────────
out  = Path('/sessions/brave-gallant-mayer/mnt/Downloads/figure2_detection.png')
out2 = ROOT / 'figures/figure2_detection.png'
print("Saving …")
plt.savefig(out,  dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(out2, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved → {out}")
plt.close()
