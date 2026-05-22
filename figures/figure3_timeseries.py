"""
Figure 3 — Bełchatów Emission Rate Time Series & Recovery Assessment

Panel A: Emission rate per confirmed overpass (2019–2024).
         30 quantification-supporting detections (sc_cfar > τ = 3.5796).
         Error bars = IME uncertainty bounds. Detection-weighted mean + 95% CI band.
         TROPOMI co-detection marked.

Panel B: Annualised estimate vs Climate TRACE 2021–2024 mean inventory.
         14.1% recovery ratio with 95% CI. CT year-to-year spread shown (±1 SD).
         Note: Climate TRACE rates confidence for this asset as low.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from datetime import datetime
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'pdf.fonttype': 42,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.65,
    'ytick.major.width': 0.65,
})

ROOT   = Path(__file__).parent.parent
INK    = '#1A1A1A'
B_DARK = '#004080'
B_FILL = '#D6E8F8'
G_DARK = '#1A6B3C'
G_FILL = '#D4EDDA'
TAU    = 3.5796

# ── Recomputed summary statistics (30-record corrected dataset) ───────────────
MEAN_KGH     =  476     # detection-weighted mean (30 records, June 2022 excluded)
CI_LO_KGH    =  341     # 95% CI lower (t-dist, df=29)
CI_HI_KGH    =  612     # 95% CI upper
ANNUAL_T     = 4174     # annualised estimate (t CH₄/yr)
ANNUAL_LO    = 2987
ANNUAL_HI    = 5360
TROPOMI_DATE = datetime(2021, 9, 9)
TROPOMI_ENH  = 12.74    # ppb

# Climate TRACE: 2021-2024 mean (excludes 2025-2026 incomplete years)
CT_YEARS = {2021: 36667, 2022: 39736, 2023: 29636, 2024: 29636}
CT_MEAN  = int(round(sum(CT_YEARS.values()) / len(CT_YEARS)))   # 33 919 t
RECOVERY_PCT    = 14.1   # vs CT-2024 (29,636)
RECOVERY_LO     = 10.1
RECOVERY_HI     = 18.1

# ── Build combined detection dataset ─────────────────────────────────────────
# 2021-2024: powerstation-crop run (01_mbsp) — gives paper-consistent mean
fp01 = ROOT / 'results_analysis/timeseries/belchatow/01_belchatow_powerstation_coords_750px_crop_2019-2024_mbsp.json'
with open(fp01) as f:
    d01 = json.load(f)

seen   = set()
quant  = []

for r in d01['records']:
    det = r.get('detection', {})
    q   = r.get('quantification', {})
    sc  = det.get('sc_cfar', 0) or 0
    acq = r.get('search', {}).get('acquisition_date', '') or r.get('acquisition_date', '')
    if not acq or r.get('partial_swath'):
        continue
    try:
        dt = datetime.fromisoformat(acq.replace('Z', ''))
    except Exception:
        continue
    rate = q.get('flow_rate_kgh')
    if sc > TAU and rate is not None:
        key = dt.strftime('%Y-%m-%d')
        if key not in seen:
            seen.add(key)
            # June 2022 (sc_cfar=4.64): above τ but excluded from quant per §6.3
            if dt.year == 2022 and dt.month == 6:
                continue
            quant.append({'dt': dt, 'sc_cfar': sc, 'rate': rate,
                          'lower': q.get('flow_rate_lower_kgh', rate * 0.70),
                          'upper': q.get('flow_rate_upper_kgh', rate * 1.30)})

# 2019-2020: mine-centroid run (main mbsp)
with open(ROOT / 'results_analysis/belchatow_annual_timeseries_mbsp.json') as f:
    mbsp = json.load(f)

for r in mbsp['records']:
    det = r.get('detection', {})
    q   = r.get('quantification', {})
    sc  = det.get('sc_cfar', 0) or 0
    acq = (r.get('search', {}).get('acquisition_date', '')
           or r.get('acquisition_date', ''))
    if not acq or acq[:4] not in ('2019', '2020') or r.get('partial_swath'):
        continue
    rate = q.get('flow_rate_kgh')
    if sc > TAU and rate is not None:
        try:
            dt = datetime.fromisoformat(acq.replace('Z', ''))
        except Exception:
            continue
        key = dt.strftime('%Y-%m-%d')
        if key not in seen:
            seen.add(key)
            quant.append({'dt': dt, 'sc_cfar': sc, 'rate': rate,
                          'lower': q.get('flow_rate_lower_kgh', rate * 0.70),
                          'upper': q.get('flow_rate_upper_kgh', rate * 1.30)})

quant.sort(key=lambda x: x['dt'])
n_det = len(quant)

dates  = np.array([q['dt']      for q in quant])
rates  = np.array([q['rate']    for q in quant])
lowers = np.array([q['lower']   for q in quant])
uppers = np.array([q['upper']   for q in quant])
cfars  = np.array([q['sc_cfar'] for q in quant])

# Climate TRACE year-to-year spread (sample std, n=4)
ct_vals = np.array(list(CT_YEARS.values()), dtype=float)
CT_STD  = float(np.std(ct_vals, ddof=1))   # ≈ 5,102 t

# sc_cfar colormap for Panel A scatter
cfar_clamp = np.clip(cfars, 1.0, 1000)
cnorm      = LogNorm(vmin=1.0, vmax=1000)
cmap       = plt.cm.plasma

# ── Canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.0, 5.8))
gs  = gridspec.GridSpec(
    1, 2,
    width_ratios=[2.5, 1.0],
    left=0.075, right=0.97,
    bottom=0.135, top=0.905,
    wspace=0.34,
)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — emission rate scatter
# ══════════════════════════════════════════════════════════════════════════════

# — mean + 95% CI band —
x_lo = datetime(2018, 10, 1)
x_hi = datetime(2025, 3, 1)
axA.axhspan(CI_LO_KGH, CI_HI_KGH, color='#D55E00', alpha=0.12, zorder=2,
            label=f'95% CI on mean ({CI_LO_KGH}–{CI_HI_KGH} kg/hr)')
axA.axhline(MEAN_KGH, color='#D55E00', lw=1.6, ls='--', zorder=3)

# — TROPOMI vertical marker —
axA.axvline(TROPOMI_DATE, color='#AA3399', lw=1.8, ls=':', zorder=4, alpha=0.9)
axA.annotate(
    f'TROPOMI +{TROPOMI_ENH:.1f} ppb',
    xy=(TROPOMI_DATE, CI_HI_KGH * 1.12),
    xytext=(8, 0), textcoords='offset points',
    ha='left', va='bottom', fontsize=9.0, color='#AA3399',
    bbox=dict(fc='white', ec='#AA3399', lw=0.7, pad=2.2, alpha=0.9),
    zorder=6,
)

# — error bars —
err_lo = rates - lowers
err_hi = uppers - rates
axA.errorbar(
    dates, rates,
    yerr=[err_lo, err_hi],
    fmt='none', ecolor='#AAAAAA', elinewidth=0.9,
    capsize=2.5, capthick=0.7, zorder=5,
)

# — scatter: colour = sc_cfar signal strength —
sc_obj = axA.scatter(
    dates, rates,
    c=cfar_clamp, norm=cnorm, cmap=cmap,
    s=52, edgecolors=INK, linewidths=0.7,
    zorder=6,
)

# — mean label —
axA.text(x_hi, MEAN_KGH * 1.06,
         f'Mean {MEAN_KGH} kg/hr',
         ha='right', va='bottom', fontsize=9.5,
         color='#D55E00', zorder=8)

# — n= label —
axA.text(0.02, 0.97,
         f'n = {n_det} detections  (sc$_{{\\rm cfar}}$ > τ = {TAU})',
         transform=axA.transAxes, ha='left', va='top',
         fontsize=9.5, color='#444444')

# — colorbar: horizontal, sits below the n= label —
cax = fig.add_axes([0.082, 0.755, 0.20, 0.013])
cb  = ColorbarBase(cax, cmap=cmap, norm=cnorm, orientation='horizontal')
cb.set_label(
    'Dot colour: sc$_{\\rm cfar}$ — CFAR-normalised S/C ratio (log scale)',
    fontsize=8.5, labelpad=3
)
cb.ax.tick_params(labelsize=7.5, top=True, labeltop=True, bottom=False, labelbottom=False)
cb.set_ticks([1, 3, 10, 100, 1000])
cb.set_ticklabels(['1', '3', '10', '100', '≥1000'])

# — legend —
mean_line  = Line2D([0], [0], color='#D55E00', lw=1.6, ls='--',
                    label=f'Mean {MEAN_KGH} kg/hr (30-record, corrected)')
ci_patch   = mpatches.Patch(fc='#D55E00', alpha=0.18, ec='none',
                             label=f'95% CI {CI_LO_KGH}–{CI_HI_KGH} kg/hr')
tropo_line = Line2D([0], [0], color='#AA3399', lw=1.8, ls=':',
                    label='TROPOMI dual-sensor confirm (2021-09-09)')
axA.legend(handles=[mean_line, ci_patch, tropo_line],
           fontsize=9.5, loc='upper right', framealpha=0.88,
           ncol=1, handlelength=1.6, edgecolor='#CCCCCC')

# — axes —
axA.set_yscale('log')
axA.set_ylim(35, 6500)
axA.set_xlim(x_lo, x_hi)
axA.xaxis.set_major_locator(mdates.YearLocator())
axA.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axA.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
axA.tick_params(axis='x', labelsize=10.0)
axA.tick_params(axis='y', labelsize=9.5, which='both')
axA.set_xlabel('Overpass date', fontsize=10.5)
axA.set_ylabel('Emission rate (kg CH$_4$ hr$^{-1}$)', fontsize=10.5)
axA.set_title('A   Emission rate per confirmed overpass — KWB Bełchatów',
              fontsize=11.0, fontweight='bold', loc='left', pad=5)

# light year separators
for yr in range(2019, 2026):
    axA.axvline(datetime(yr, 1, 1), color='#EEEEEE', lw=0.55, zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — recovery comparison
# ══════════════════════════════════════════════════════════════════════════════

labels = [
    'This study\n(Sentinel-2)',
    'Climate TRACE\n2021–2024\nmean inventory†',
]
values  = [ANNUAL_T, CT_MEAN]
y_pos   = [1,        0]
fcs     = [B_FILL,   G_FILL]
ecs     = [B_DARK,   G_DARK]

for y, v, fc, ec in zip(y_pos, values, fcs, ecs):
    axB.barh(y, v, height=0.45, color=fc, edgecolor=ec, lw=1.4, zorder=2)

# value labels: blue (this study) right of CI upper; green (CT) left of SD lower
axB.text(ANNUAL_HI + 700,       1, f'{ANNUAL_T:,}', va='center', ha='left',
         fontsize=10.0, color=B_DARK, fontweight='bold')
axB.text(CT_MEAN - CT_STD - 700, 0, f'{CT_MEAN:,}', va='center', ha='right',
         fontsize=10.0, color=G_DARK, fontweight='bold')

# CI error bar on this-study bar
axB.errorbar(ANNUAL_T, 1,
             xerr=[[ANNUAL_T - ANNUAL_LO], [ANNUAL_HI - ANNUAL_T]],
             fmt='none', ecolor=B_DARK, elinewidth=1.4,
             capsize=5, capthick=1.2, zorder=3)

# ±1 SD error bar on Climate TRACE bar (year-to-year variability, n=4)
axB.errorbar(CT_MEAN, 0,
             xerr=CT_STD,
             fmt='none', ecolor=G_DARK, elinewidth=1.4,
             capsize=5, capthick=1.2, zorder=3)

# individual CT year dots
ct_y_offsets = {2021: 0.19, 2022: 0.25, 2023: -0.10, 2024: -0.19}
for yr, val in CT_YEARS.items():
    axB.plot(val, 0 + ct_y_offsets[yr], 'o',
             color=G_DARK, ms=4.5, alpha=0.65, zorder=4)
    axB.text(val, 0 + ct_y_offsets[yr],
             f' {yr}', va='center', ha='left',
             fontsize=8.0, color=G_DARK, alpha=0.8)

# recovery annotation (vs CT-2024 to match paper's primary comparison)
ct_2024 = 29636
mid_x = (ANNUAL_T + ct_2024) / 2
axB.annotate('', xy=(ANNUAL_T, 0.55), xytext=(ct_2024, 0.55),
             arrowprops=dict(arrowstyle='<->', color='#666666', lw=1.0))
axB.text(mid_x, 0.60,
         f'{RECOVERY_PCT}% of CT-2024\n(95% CI: {RECOVERY_LO}–{RECOVERY_HI}%)',
         ha='center', va='bottom', fontsize=9.0, color='#555555', style='italic')

# low-confidence note (dagger footnote)
axB.text(0.02, 0.04,
         '† Climate TRACE confidence rating for this asset: low',
         transform=axB.transAxes, ha='left', va='bottom',
         fontsize=8.0, color='#888888', style='italic')

# t/yr units label on each bar
for y, v, ec in zip(y_pos, values, ecs):
    pass  # already labelled

# axes
axB.set_xlim(0, (CT_MEAN + CT_STD) * 1.22)
axB.set_ylim(-0.65, 1.60)
axB.set_yticks(y_pos)
axB.set_yticklabels(labels, fontsize=9.5)
axB.set_xlabel('CH$_4$ (t yr$^{-1}$)', fontsize=10.5)
axB.set_title('B   Annualised estimate vs. inventory',
              fontsize=11.0, fontweight='bold', loc='left', pad=5)
axB.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
axB.tick_params(axis='x', labelsize=9.0, rotation=15)
axB.spines['right'].set_visible(False)
axB.spines['top'].set_visible(False)

# ── Save ──────────────────────────────────────────────────────────────────────
out1 = Path('/sessions/brave-gallant-mayer/mnt/Downloads/figure3_timeseries.png')
out2 = ROOT / 'figures/figure3_timeseries.png'
print(f'n_det = {n_det}  |  rates: mean={rates.mean():.0f}, median={np.median(rates):.0f}')
print('Saving …')
plt.savefig(out1, dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(out2, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {out1}')
plt.close()
