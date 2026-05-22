"""
Figure 2 — KWB Bełchatów CH₄Net detection evidence

  A : S/C CFAR scatter time series (winter shading, TROPOMI annotation)
  B : Seasonal detection ceiling  (KWB Bełchatów)
  C : Sentinel-2 RGB mine detail  (polygon, site crop, centroid, CT pin)
  D : Wide S2 RGB context — mine polygon + 4 control-crop boxes
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
    'font.size': 11,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.titlesize': 12,
    'font.family': 'serif',
    'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
})

COLORS = {
    'above':  '#0072B2',
    'sub':    '#BBBBBB',
    'thresh': '#D55E00',
    'ct':     '#009E73',
    'winter': '#8888BB',
}
TAU  = 3.5796
ROOT = Path(__file__).parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────
def wgs84_to_utm34n(lat_deg, lon_deg):
    a=6378137.0; e2=0.00669437999014; k0=0.9996; lon0=21.0
    lat=math.radians(lat_deg); lon=math.radians(lon_deg); lon0r=math.radians(lon0)
    N=a/math.sqrt(1-e2*math.sin(lat)**2); T=math.tan(lat)**2
    A_=math.cos(lat)*(lon-lon0r); e4=e2**2; e6=e2**3
    M=a*((1-e2/4-3*e4/64-5*e6/256)*lat
         -(3*e2/8+3*e4/32+45*e6/1024)*math.sin(2*lat)
         +(15*e4/256+45*e6/1024)*math.sin(4*lat)
         -(35*e6/3072)*math.sin(6*lat))
    x=500000+k0*N*(A_+(1-T)*A_**3/6+(5-18*T+T**2)*A_**5/120)
    y=k0*(M+N*math.tan(lat)*(A_**2/2+(5-T+9*T)*A_**4/24+(61-58*T+T**2)*A_**6/720))
    return x, y

def utm_to_px(utm_x, utm_y, origin_x=300000, origin_y=5700000, res=10):
    return (utm_x - origin_x) / res, (origin_y - utm_y) / res   # col, row

def pct_stretch(arr, lo=2, hi=98):
    p_lo, p_hi = np.percentile(arr, lo), np.percentile(arr, hi)
    return np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)

# ── detection data ─────────────────────────────────────────────────────────────
# Two sources, both using the mine centroid crop (51.242°N, 19.275°E):
#   (A) belchatow_annual_timeseries_mbsp.json  → 2019–2020 acquisitions
#       cfar_detect flags are buggy; use sc_cfar > τ directly.
#   (B) production_rule_audit.json, _source='belchatow_annual_timeseries'
#       → 2021–2024 acquisitions; DETECTION classification is authoritative.
#   historical_backfill records EXCLUDED: used power-station crop (51.266°N, 19.315°E).

TAU_CHECK = TAU   # 3.5796

rows = []

# ── (A) mbsp json 2019–2020 ────────────────────────────────────────────────────
with open(ROOT / 'results_analysis/belchatow_annual_timeseries_mbsp.json') as f:
    mbsp = json.load(f)
for r in mbsp['records']:
    dt = (r.get('acquisition_date')
          or r.get('search', {}).get('acquisition_date')
          or r.get('month', ''))
    if not dt or dt[:4] not in ('2019', '2020'):
        continue
    if r.get('partial_swath'):
        continue
    det = r.get('detection', {}) or {}
    sc = det.get('sc_cfar')
    if sc is None:
        continue
    rows.append({
        'date':       dt[:10],
        'sc_cfar':    sc,
        'detect':     sc > TAU_CHECK,
        'suppressed': False,
    })

# ── (B) production_rule_audit 2021–2024 ────────────────────────────────────────
with open(ROOT / 'results_analysis/production_rule_audit.json') as f:
    audit = json.load(f)
for r in audit['records']:
    if r['site'] != 'belchatow':
        continue
    if r.get('_source') != 'belchatow_annual_timeseries':
        continue
    cls = r.get('classification', '')
    if cls in ('NO_COVERAGE', 'NOT_OK'):
        continue
    sc = r.get('sc_cfar')
    if sc is None:
        continue
    raw_date = r['date']
    date_str = (raw_date + '-15') if len(raw_date) == 7 else raw_date
    rows.append({
        'date':       date_str,
        'sc_cfar':    sc,
        'detect':     cls == 'DETECTION',
        'suppressed': cls == 'CFAR_SUPPRESSED',
    })

df = pd.DataFrame(rows)
df['date'] = pd.to_datetime(df['date'], utc=True)
df = df.drop_duplicates(subset=['date', 'sc_cfar']).copy()
df['quarter'] = df['date'].dt.quarter

# Scatter masks
ab   = df['detect']
sub  = ~df['detect'] & ~df['suppressed']
supp = df['suppressed']

xlim_l = pd.Timestamp('2018-10-01', tz='UTC')
xlim_r = df['date'].max() + pd.Timedelta(days=60)

# ── S2 imagery ─────────────────────────────────────────────────────────────────
print("Loading S2 tile …")
npy_path = (ROOT / 'data/npy_cache/'
            'S2B_MSIL1C_20210909T095029_N0500_R079_T34UCB_20230117T015854.npy')
s2 = np.load(npy_path)
print(f"  shape: {s2.shape}")

# Panel C — mine detail: PAD_R=620 + PAD_C=60 → ~4:3 crop (1713 × 2222 px)
PAD_R, PAD_C = 620, 60
r0c, r1c = 1987 - PAD_R, 2460 + PAD_R   # rows 1367–3080
c0c, c1c = 6721 - PAD_C, 8823 + PAD_C   # cols 6661–8883
map_crop  = s2[r0c:r1c, c0c:c1c, :].astype(np.float32)
rgb_C = np.stack([pct_stretch(map_crop[:,:,3]),
                  pct_stretch(map_crop[:,:,2]),
                  pct_stretch(map_crop[:,:,1])], axis=-1)

def crop_px_C(lat, lon):
    x, y = wgs84_to_utm34n(lat, lon)
    gc, gr = utm_to_px(x, y)
    return gc - c0c, gr - r0c

# Panel D — wide context: rows 0-4700, cols 4900-10300 (~47 × 54 km), DS=5
CROP_R0, CROP_R1 = 0,    4700
CROP_C0, CROP_C1 = 4900, 10300
DS = 5
print("Extracting wide crop …")
wide_raw = s2[CROP_R0:CROP_R1, CROP_C0:CROP_C1, :].astype(np.float32)
rgb_D    = np.stack([pct_stretch(wide_raw[:,:,3]),
                     pct_stretch(wide_raw[:,:,2]),
                     pct_stretch(wide_raw[:,:,1])], axis=-1)
rgb_D_ds = rgb_D[::DS, ::DS]
D_H, D_W = rgb_D_ds.shape[:2]   # 940 × 1080 display px
print(f"  wide-crop display: {D_W}×{D_H}")

# (pad imagery computed in Panel D section using already-stretched rgb_D_ds)

def crop_px_D(lat, lon):
    x, y = wgs84_to_utm34n(lat, lon)
    gc, gr = utm_to_px(x, y)
    return (gc - CROP_C0) / DS, (gr - CROP_R0) / DS

# ── geometry ──────────────────────────────────────────────────────────────────
corners_latlon = [(51.2570, 19.0970), (51.2566, 19.3900),
                  (51.2190, 19.3996), (51.2185, 19.0990)]

# Panel C
corners_px_C = [crop_px_C(la, lo) for la, lo in corners_latlon]
mine_xs_C    = [p[0] for p in corners_px_C] + [corners_px_C[0][0]]
mine_ys_C    = [p[1] for p in corners_px_C] + [corners_px_C[0][1]]
mc_x_C, mc_y_C = crop_px_C(51.242, 19.275)
BH = 50
box_x_C = [mc_x_C-BH, mc_x_C+BH, mc_x_C+BH, mc_x_C-BH, mc_x_C-BH]
box_y_C = [mc_y_C-BH, mc_y_C-BH, mc_y_C+BH, mc_y_C+BH, mc_y_C-BH]
det_x_C, det_y_C = crop_px_C(51.2495, 19.2227)
ct_x_C,  ct_y_C  = crop_px_C(51.2420, 19.2754)

corner_labels_C = [
    (corners_px_C[0][0], corners_px_C[0][1], 'right', 'bottom', '51.257°N', '19.097°E'),
    (corners_px_C[1][0], corners_px_C[1][1], 'left',  'bottom', '51.257°N', '19.390°E'),
    (corners_px_C[2][0], corners_px_C[2][1], 'left',  'top',    '51.219°N', '19.400°E'),
    (corners_px_C[3][0], corners_px_C[3][1], 'right', 'top',    '51.219°N', '19.099°E'),
]

# Panel D
corners_px_D = [crop_px_D(la, lo) for la, lo in corners_latlon]
mine_xs_D    = [p[0] for p in corners_px_D] + [corners_px_D[0][0]]
mine_ys_D    = [p[1] for p in corners_px_D] + [corners_px_D[0][1]]
mine_cen_x_D = sum(p[0] for p in corners_px_D) / 4
mine_cen_y_D = sum(p[1] for p in corners_px_D) / 4

site_lat, site_lon = 51.242, 19.275
sx_D, sy_D = crop_px_D(51.2495, 19.2227)   # detection centroid = site indicator

ctrl_centres = {
    'N': (site_lat + 0.20, site_lon),
    'S': (site_lat - 0.20, site_lon),
    'E': (site_lat,         site_lon + 0.30),
    'W': (site_lat,         site_lon - 0.39),
}
ctrl_px_D     = {k: crop_px_D(la, lo) for k, (la, lo) in ctrl_centres.items()}
BOX_HALF_D    = 1000 / (10 * DS)   # 1 km half-side = 20 display px
ctrl_dist_str = {'N': '22 km N', 'S': '22 km S', 'E': '33 km E', 'W': '43 km W'}

towns = {'Bełchatów': (51.361, 19.360), 'Kamieńsk': (51.216, 19.491),
         'Radomsko':  (51.062, 19.450), 'Pajęczno': (51.157, 18.964)}
town_px_D = {nm: crop_px_D(la, lo) for nm, (la, lo) in towns.items()}

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.0, 9.0))

gs_root = gridspec.GridSpec(
    2, 1, figure=fig,
    height_ratios=[1.0, 1.4],
    left=0.06, right=0.98, top=0.95, bottom=0.06,
    hspace=0.28)

# Top row: A (scatter) + B (seasonal) side by side
gs_top = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_root[0],
    width_ratios=[2.8, 1.0], wspace=0.22)
axA = fig.add_subplot(gs_top[0])
axB = fig.add_subplot(gs_top[1])

# Bottom row: C (mine RGB) + D (wide control)
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs_root[1],
    width_ratios=[1.0, 1.35], wspace=0.05)
axC = fig.add_subplot(gs_bot[0])
axD = fig.add_subplot(gs_bot[1])

# ══════════════════════════════════════════════════════════════════════════════
# A — S/C CFAR time series
# ══════════════════════════════════════════════════════════════════════════════
for yr in range(2019, 2026):
    axA.axvspan(pd.Timestamp(f'{yr}-01-01', tz='UTC'),
                pd.Timestamp(f'{yr}-03-31', tz='UTC'),
                alpha=0.09, color=COLORS['winter'], lw=0, zorder=0)
    axA.axvspan(pd.Timestamp(f'{yr}-10-01', tz='UTC'),
                pd.Timestamp(f'{yr}-12-31', tz='UTC'),
                alpha=0.09, color=COLORS['winter'], lw=0, zorder=0)

axA.scatter(df.loc[sub, 'date'], df.loc[sub, 'sc_cfar'],
            facecolors='none', edgecolors=COLORS['sub'], s=14, lw=0.5,
            label='Sub-threshold', zorder=3)
axA.scatter(df.loc[ab, 'date'], df.loc[ab, 'sc_cfar'],
            color=COLORS['above'], s=18, label='Detection', zorder=4)
if supp.any():
    axA.scatter(df.loc[supp, 'date'], df.loc[supp, 'sc_cfar'],
                facecolors='none', edgecolors=COLORS['thresh'], s=22,
                marker='^', lw=1.0, label='CFAR-suppressed', zorder=4)
axA.axhline(TAU, color=COLORS['thresh'], ls='--', lw=1.3,
            label=f'CFAR threshold τ = {TAU}', zorder=5)

tropomi_dt = pd.Timestamp('2021-09-09', tz='UTC')
axA.axvline(tropomi_dt, color='#AA3399', lw=2.0, ls=':', zorder=6, alpha=0.9)

axA.annotate('2019-08-31  sc_cfar = 803',
             xy=(pd.Timestamp('2019-08-31', tz='UTC'), 802.6),
             xytext=(pd.Timestamp('2021-06-01', tz='UTC'), 200),
             fontsize=10.5, color=COLORS['above'],
             arrowprops=dict(arrowstyle='->', lw=0.8, color=COLORS['above'],
                             connectionstyle='arc3,rad=0.22'), va='center')

axA.set_yscale('log'); axA.set_ylim(1e-3, 3000)
axA.set_xlim(xlim_l, xlim_r)
axA.set_ylabel('S/C ratio (CFAR-normalised)')
axA.set_title('A.  KWB Bełchatów — CH₄Net detection signal (S/C CFAR)',
               fontweight='bold', loc='left')
axA.xaxis.set_major_locator(mdates.YearLocator())
axA.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axA.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
axA.tick_params(axis='x', which='minor', length=2)

win_p    = mpatches.Patch(color=COLORS['winter'], alpha=0.4, label='Winter (Q1/Q4)')
tropo_h  = Line2D([0], [0], color='#AA3399', lw=2.0, ls=':', label='TROPOMI +12.7 ppb (2021-09-09)')
h1, l1 = axA.get_legend_handles_labels()
axA.legend(h1 + [win_p, tropo_h], l1 + ['Winter (Q1/Q4)', 'TROPOMI +12.7 ppb (2021-09-09)'],
           fontsize=10.5, loc='lower left', framealpha=0.85,
           ncol=2, handlelength=1.2)
n_det = int(ab.sum())
axA.text(0.99, 0.01,
         f'n = {n_det} detections (mine centroid crop, 51.242°N, 19.275°E)',
         transform=axA.transAxes, fontsize=11.5, color='#555555',
         va='bottom', ha='right')

# ══════════════════════════════════════════════════════════════════════════════
# B — Seasonal detection ceiling  (KWB Bełchatów)
# ══════════════════════════════════════════════════════════════════════════════
all_q = df['quarter'].value_counts()
det_q = df[df['detect']]['quarter'].value_counts()
q_lab   = ['Q1\n(Jan–Mar)', 'Q2\n(Apr–Jun)', 'Q3\n(Jul–Sep)', 'Q4\n(Oct–Dec)']
q_dets  = [det_q.get(i, 0) for i in range(1, 5)]
q_total = [all_q.get(i, 0) for i in range(1, 5)]
bar_c   = [COLORS['winter'], COLORS['above'], COLORS['above'], COLORS['winter']]
bar_a   = [0.55, 0.90, 0.90, 0.55]
for i in range(4):
    axB.bar(i, q_dets[i], color=bar_c[i], alpha=bar_a[i], width=0.72, zorder=2)
axB.bar(range(4), q_total, color='none', edgecolor='#555555', lw=0.8, width=0.72, zorder=3)
axB.set_xticks(range(4)); axB.set_xticklabels(q_lab, fontsize=11.5)
axB.set_ylabel('# overpasses')
axB.set_title('B.  Seasonal detection ceiling\n(KWB Bełchatów)',
               fontweight='bold', loc='left')
p1 = mpatches.Patch(color=COLORS['above'], alpha=0.9,   label='Detections (Q2/Q3)')
p2 = mpatches.Patch(color=COLORS['winter'], alpha=0.55, label='Winter (Q1/Q4)')
p3 = mpatches.Patch(fc='none', ec='#555555', lw=0.8,   label='Total obs.')
axB.legend(handles=[p1, p2, p3], fontsize=10.5, loc='upper right', framealpha=0.85)

# ══════════════════════════════════════════════════════════════════════════════
# C — Mine S2 RGB detail  (undistorted 4:3 crop, aspect='equal')
# ══════════════════════════════════════════════════════════════════════════════
axC.imshow(rgb_C, origin='upper', interpolation='bilinear', aspect='equal')

axC.plot(mine_xs_C, mine_ys_C, color='#FF8C00', lw=1.6, ls='--', zorder=5,
         label='Mine polygon (KWB Bełchatów)')

for cx, cy, ha, va, lat_s, lon_s in corner_labels_C:
    ox = -5 if ha == 'right' else 5
    oy = -4 if va == 'bottom' else 4
    axC.text(cx + ox, cy + oy, f'{lat_s}\n{lon_s}',
             ha=ha, va=va, fontsize=6.5, color='#FF8C00', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.14', fc='black', alpha=0.55, lw=0))

axC.plot(box_x_C, box_y_C, color=COLORS['above'], lw=1.6, ls='-', zorder=5,
         label='S/C site crop (1 km²)')
axC.plot(det_x_C, det_y_C, '*', color='#CC0000', ms=10, zorder=6,
         label='Detection centroid')
axC.plot(ct_x_C, ct_y_C, 's', color=COLORS['ct'], ms=7, zorder=6,
         label='ClimateTrace source pin')

sh = rgb_C.shape
sb_x0, sb_x1 = sh[1]*0.03, sh[1]*0.03 + 500
sb_y = sh[0] * 0.95
axC.plot([sb_x0, sb_x1], [sb_y, sb_y], 'w-', lw=2.5)
axC.text((sb_x0+sb_x1)/2, sb_y - 20, '5 km',
         ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')

axC.set_xticks([]); axC.set_yticks([])
axC.set_title('C.  Sentinel-2 RGB  ·  2021-09-09  ·  mine geometry',
               fontweight='bold', loc='left')
axC.annotate('N', xy=(0.97, 0.93), xytext=(0.97, 0.86),
             xycoords='axes fraction', textcoords='axes fraction',
             ha='center', fontsize=10, fontweight='bold', color='white',
             arrowprops=dict(arrowstyle='->', color='white', lw=1.3))
axC.legend(fontsize=10, loc='lower left', framealpha=0.88,
           handlelength=1.4, markerscale=0.85,
           facecolor='#111111', labelcolor='white', edgecolor='#444444')

# ══════════════════════════════════════════════════════════════════════════════
# D — Wide S2 context: mine polygon + 4 control-crop boxes
#     Labels placed OUTSIDE each box (above/at edges) so box interiors stay clear
# ══════════════════════════════════════════════════════════════════════════════
axD.imshow(rgb_D_ds, origin='upper', interpolation='bilinear', aspect='auto',
           extent=[0, D_W, D_H, 0])

# Mine polygon + centre label
axD.plot(mine_xs_D, mine_ys_D, color='#FF8C00', lw=2.0, ls='--', zorder=5)
axD.text(mine_cen_x_D, mine_cen_y_D + 55, 'KWB\nBełchatów',
         ha='center', va='top', fontsize=11, color='#FF8C00', fontweight='bold',
         zorder=7, bbox=dict(boxstyle='round,pad=0.22', fc='#00000088', lw=0))

# S/C site: filled semi-transparent box at detection centroid
axD.add_patch(mpatches.Rectangle(
    (sx_D - BOX_HALF_D, sy_D - BOX_HALF_D), 2*BOX_HALF_D, 2*BOX_HALF_D,
    edgecolor=COLORS['above'], facecolor=COLORS['above'],
    alpha=0.40, lw=2.2, zorder=6))
axD.text(sx_D, sy_D + BOX_HALF_D + 11, 'S/C site',
         ha='center', va='top', fontsize=10.5, color='#88CCFF',
         fontweight='bold', zorder=8)

# Helper: label style for ctrl boxes
lbl_kw = dict(fontsize=10.5, color='white', fontweight='bold', zorder=8,
              bbox=dict(boxstyle='round,pad=0.25', fc='#002244BB',
                        lw=0.8, ec='#66AAFF'))

for name, (cx_ctrl, cy_ctrl) in ctrl_px_D.items():
    # thin dotted line from site to ctrl centre (context only, behind boxes)
    axD.plot([sx_D, cx_ctrl], [sy_D, cy_ctrl],
             color='white', lw=0.7, ls=':', zorder=3, alpha=0.5)

    # Draw the dashed box for all crops — N may sit slightly above row 0
    # (tile top); the extended ylim below accommodates it cleanly.
    axD.add_patch(mpatches.Rectangle(
        (cx_ctrl - BOX_HALF_D, cy_ctrl - BOX_HALF_D),
        2*BOX_HALF_D, 2*BOX_HALF_D,
        edgecolor='#66AAFF', facecolor='none', lw=1.6, ls='--', zorder=5))

    # Place label OUTSIDE the box so box interior stays visible.
    # N crop sits partially above the tile top edge — box is clipped naturally
    # by the axes boundary; label goes to the RIGHT of the box so it stays
    # on imagery and doesn't collide with the panel title above.
    if name == 'N':
        lbl_x = cx_ctrl + BOX_HALF_D + 8   # right of box
        lbl_y = np.clip(cy_ctrl, 8, D_H - 8)
        ha, va = 'left', 'center'
    elif name == 'S':
        lbl_x = cx_ctrl
        lbl_y = cy_ctrl + BOX_HALF_D + 8   # below box
        ha, va = 'center', 'top'
    else:   # W, E — label above box
        lbl_x = cx_ctrl
        lbl_y = cy_ctrl - BOX_HALF_D - 8
        ha, va = 'center', 'bottom'

    lbl_x = np.clip(lbl_x, 8, D_W - 8)
    if name != 'N':
        lbl_y = np.clip(lbl_y, 8, D_H - 8)
    axD.text(lbl_x, lbl_y, f'Ctrl {name}\n({ctrl_dist_str[name]})',
             ha=ha, va=va, **lbl_kw)

# Town labels
town_ha  = {'Bełchatów': 'right', 'Kamieńsk': 'left',
            'Radomsko':  'center', 'Pajęczno': 'center'}
town_off = {'Bełchatów': (-7, 0), 'Kamieńsk': (7, 0),
            'Radomsko':  (0, 10), 'Pajęczno': (0, 10)}
for nm, (tx, ty) in town_px_D.items():
    if nm == 'Radomsko':   # sits on top of the legend — omit
        continue
    if not (8 < tx < D_W - 8 and 8 < ty < D_H - 8):
        continue
    axD.plot(tx, ty, 'o', ms=4, color='white',
             markeredgecolor='#888888', markeredgewidth=0.5, zorder=6)
    dx_t, dy_t = town_off[nm]
    axD.text(tx + dx_t, ty + dy_t, nm, ha=town_ha[nm], va='center',
             fontsize=10.5, color='white', fontweight='bold', zorder=7,
             bbox=dict(boxstyle='round,pad=0.15', fc='#00000066', lw=0))

# Scale bar: 10 km = 200 display px
sb_x0d = D_W * 0.04; sb_x1d = sb_x0d + 200; sb_yd = D_H * 0.96
axD.plot([sb_x0d, sb_x1d], [sb_yd, sb_yd], 'w-', lw=3.0)
axD.text((sb_x0d+sb_x1d)/2, sb_yd - 10, '10 km',
         ha='center', va='bottom', fontsize=11.5, color='white', fontweight='bold')
axD.annotate('N', xy=(0.97, 0.945), xytext=(0.97, 0.875),
             xycoords='axes fraction', textcoords='axes fraction',
             ha='center', fontsize=11, fontweight='bold', color='white',
             arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

# N control crop sits just above the tile's northern edge — no imagery there.
# Extend ylim above row 0 so the box shows fully; fill the strip with the
# sampled mean colour of the top image rows so it blends rather than going black.
N_PAD = int(BOX_HALF_D * 2 + 30)   # display px — enough for box + label clearance
# Fill the above-tile strip with mirrored tile imagery for visual continuity.
# Taking the top N_PAD rows of rgb_D_ds and flipping vertically creates a
# seamless mirror at y=0 (tile northern boundary) with identical colour stretch.
rgb_pad_strip = rgb_D_ds[:N_PAD, :, :][::-1, :, :]
axD.imshow(rgb_pad_strip, origin='upper', interpolation='bilinear', aspect='auto',
           extent=[0, D_W, 0, -N_PAD])
axD.set_xlim(0, D_W); axD.set_ylim(D_H, -N_PAD)
# Thin tile-edge indicator so the boundary is honest but unobtrusive
axD.axhline(0, color='white', lw=0.6, ls=':', alpha=0.35, zorder=4)
axD.set_xticks([]); axD.set_yticks([])
axD.set_title('D.  S/C control-crop layout  ·  S2 RGB context  ·  2021-09-09',
               fontweight='bold', loc='left')

h_mine = Line2D([0],[0], color='#FF8C00', lw=1.8, ls='--', label='Mine polygon')
h_site = mpatches.Patch(facecolor=COLORS['above'], alpha=0.45, label='S/C site crop')
h_ctrl = Line2D([0],[0], color='#66AAFF', lw=1.4, ls='--', label='Control crops (4×)')
axD.legend(handles=[h_mine, h_site, h_ctrl], fontsize=10.5, loc='lower right',
           framealpha=0.90, handlelength=1.8,
           facecolor='#111111', labelcolor='white', edgecolor='#444444')

# ── save ──────────────────────────────────────────────────────────────────────
out  = Path('/sessions/brave-gallant-mayer/mnt/Downloads/figure2_belchatow.png')
out2 = ROOT / 'figures/figure2_belchatow.png'
print("Saving …")
plt.savefig(out,  dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(out2, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved → {out}")
plt.close()
