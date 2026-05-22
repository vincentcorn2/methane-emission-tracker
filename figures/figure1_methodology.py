"""
Figure 1 — CH₄Net pipeline methodology flow diagram
Detection path: top → bottom (left column)
Quantification path: bottom → top (right column)
NO exits go LEFT; YES arrows are implied (unlabelled).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
    'pdf.fonttype': 42,
})

# ── colours ───────────────────────────────────────────────────────────────────
B_DARK  = '#004080'
B_FILL  = '#D6E8F8'
O_DARK  = '#7A3800'
O_FILL  = '#FEF0DC'
SLATE   = '#3D5166'
GREY_FC = '#F0F0F0'
GREY_EC = '#888888'
INK     = '#1A1A1A'

# ── canvas ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.6, 9.6))
ax  = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0, 10)
ax.set_ylim(-0.3, 14.8)
ax.axis('off')

# ── font sizes (all node text at same size; only headers + confirmed are bold) ──
FS      = 9.5   # all node labels
FS_EXIT = 8.4   # small exit stubs
FS_LBL  = 8.8   # NO labels

# ── helpers ───────────────────────────────────────────────────────────────────

def pbox(cx, cy, w, h, text, fc, ec=INK, lw=1.4, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.12', fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=FS,
            fontweight='bold' if bold else 'normal', color=tc,
            multialignment='center', linespacing=1.35, zorder=3)


def diam(cx, cy, w, h, text, fc=B_FILL, ec=B_DARK):
    pts = np.array([(cx, cy+h/2), (cx+w/2, cy),
                    (cx, cy-h/2), (cx-w/2, cy)])
    ax.add_patch(plt.Polygon(pts, fc=fc, ec=ec, lw=1.6, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=FS,
            fontweight='normal', color=INK,           # same weight as pbox
            multialignment='center', linespacing=1.35, zorder=3)


def cyl(cx, cy, w, h, text):
    eh       = h * 0.30
    y_bot    = cy - h/2
    body_top = cy + h/2 - eh/2
    ax.add_patch(Rectangle(
        (cx - w/2, y_bot), w, body_top - y_bot + eh/4,
        fc=SLATE, ec='none', lw=0, zorder=2))
    ax.add_patch(Ellipse(
        (cx, body_top), w, eh, fc=SLATE, ec='none', lw=0, zorder=2))
    theta = np.linspace(0, np.pi, 80)
    ax.plot(cx + (w/2)*np.cos(theta), body_top + (eh/2)*np.sin(theta),
            color=INK, lw=1.4, zorder=4)
    ax.plot([cx-w/2, cx-w/2], [y_bot, body_top], color=INK, lw=1.4, zorder=4)
    ax.plot([cx+w/2, cx+w/2], [y_bot, body_top], color=INK, lw=1.4, zorder=4)
    ax.plot([cx-w/2, cx+w/2], [y_bot,    y_bot], color=INK, lw=1.4, zorder=4)
    ax.text(cx, (y_bot + body_top) / 2, text,
            ha='center', va='center', fontsize=FS,    # same size as pbox
            fontweight='normal', color='white',        # same weight as pbox
            multialignment='center', linespacing=1.35, zorder=5)


def varr(x, y1, y2, color=INK, lw=1.4):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=5)


def harr(y, x1, x2, color=INK, lw=1.4):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=5)


# ── geometry ──────────────────────────────────────────────────────────────────
LX  = 3.10    # detection column centre
RX  = 7.30    # quantification column centre
BW  = 3.80    # process box width — narrowed so neither column crosses the divider
BH  = 0.90    # process box height
CW  = 3.80    # cylinder width
CH  = 1.00    # cylinder height
DW  = 2.80    # diamond width  (slightly expanded)
DH  = 1.25    # diamond height (slightly expanded)
EW  = 1.20    # exit node width
EH  = 0.70    # exit node height
EX  = 0.80    # exit node centre x  (left edge ≈ 0.20 — no clip)

# detection path: top → bottom
Y0 = 12.55   # Copernicus archive  [cyl]
Y1 = 11.10   # Download image      [pbox]
Y2 =  9.62   # Cloud-free?         [diamond]
Y3 =  8.05   # Full site covered?  [diamond]
Y4 =  6.55   # Run CH4Net          [pbox]
Y5 =  5.18   # Facility stand out? [pbox]
Y6 =  3.68   # Strong enough?      [diamond]
Y7 =  2.28   # Detection confirmed [pbox]

# quantification path: bottom → top
Q0 =  2.28   # ERA5                [cyl]   same row as Y7
Q1 =  4.72   # Measure column      [pbox]
Q2 =  7.18   # Convert to rate     [pbox]
Q3 =  9.62   # Save to log         [cyl]

# NO label sits just inside the left tip of each diamond — away from exit boxes
DIA_L  = LX - DW/2    # left tip x = 1.70

# ── column headers ────────────────────────────────────────────────────────────
ax.text(LX, 14.30, 'Detection Path', ha='center', va='center',
        fontsize=12, fontweight='bold', color=B_DARK)
ax.text(RX, 14.30, 'Quantification Path', ha='center', va='center',
        fontsize=12, fontweight='bold', color=O_DARK)

# faint separator
ax.plot([5.12, 5.12], [0.05, 13.70], color='#DDDDDD', lw=0.8, ls='--', zorder=1)

# ── DETECTION PATH ─────────────────────────────────────────────────────────────

cyl(LX, Y0, CW, CH, 'Copernicus satellite archive\n(Sentinel-2 imagery)')
varr(LX, Y0 - CH/2, Y1 + BH/2, color=B_DARK)

pbox(LX, Y1, BW, BH,
     'Download satellite image\nfor target overpass date',
     fc=B_FILL, ec=B_DARK)
varr(LX, Y1 - BH/2, Y2 + DH/2, color=B_DARK)

# ① Cloud-free?
diam(LX, Y2, DW, DH, 'Is the image\ncloud-free?')
harr(Y2, DIA_L, EX + EW/2, color=GREY_EC)
ax.text(DIA_L + 0.03, Y2 + 0.22, 'NO', ha='left', va='bottom',
        fontsize=FS_LBL, color=GREY_EC, fontstyle='italic', zorder=6)
pbox(EX, Y2, EW, EH, 'Cloudy —\nskip',
     fc=GREY_FC, ec=GREY_EC, lw=1.1, tc='#555555')
varr(LX, Y2 - DH/2, Y3 + DH/2, color=B_DARK)

# ② Full site covered?
diam(LX, Y3, DW, DH, 'Did the satellite photograph\nthe full site?')
harr(Y3, DIA_L, EX + EW/2, color=GREY_EC)
ax.text(DIA_L + 0.03, Y3 + 0.22, 'NO', ha='left', va='bottom',
        fontsize=FS_LBL, color=GREY_EC, fontstyle='italic', zorder=6)
pbox(EX, Y3, EW, EH, 'Partial —\nskip',
     fc=GREY_FC, ec=GREY_EC, lw=1.1, tc='#555555')
varr(LX, Y3 - DH/2, Y4 + BH/2, color=B_DARK)

pbox(LX, Y4, BW, BH,
     'Run CH4Net methane detector\n→ per-pixel probability map',
     fc=B_FILL, ec=B_DARK)
varr(LX, Y4 - BH/2, Y5 + BH/2, color=B_DARK)

pbox(LX, Y5, BW, BH,
     'Does the facility stand out\nfrom its surroundings?  (S/C ratio)',
     fc=B_FILL, ec=B_DARK)
varr(LX, Y5 - BH/2, Y6 + DH/2, color=B_DARK)

# ③ Signal strong enough?
diam(LX, Y6, DW, DH, 'Signal strong enough\nto confirm a detection?')
harr(Y6, DIA_L, EX + EW/2, color=GREY_EC)
ax.text(DIA_L + 0.03, Y6 + 0.22, 'NO', ha='left', va='bottom',
        fontsize=FS_LBL, color=GREY_EC, fontstyle='italic', zorder=6)
pbox(EX, Y6, EW, EH, 'Below\nthreshold',
     fc=GREY_FC, ec=GREY_EC, lw=1.1, tc='#555555')
varr(LX, Y6 - DH/2, Y7 + BH/2, color=B_DARK)

# Detection confirmed — only bold node in the flow
pbox(LX, Y7, BW, BH,
     'Detection confirmed',
     fc='white', ec=INK, lw=2.0, bold=True, tc=INK)

# ── JUNCTION arrow ────────────────────────────────────────────────────────────
harr(Y7, LX + BW/2, RX - CW/2, color=O_DARK, lw=1.8)

# ── QUANTIFICATION PATH (bottom → top) ────────────────────────────────────────

cyl(RX, Q0, CW, CH, 'ERA5 climate archive\n(wind speed at overpass time)')
varr(RX, Q0 + CH/2, Q1 - BH/2, color=O_DARK)

pbox(RX, Q1, BW, BH,
     'Estimate how much methane\nis above the mine area',
     fc=O_FILL, ec=O_DARK)
varr(RX, Q1 + BH/2, Q2 - BH/2, color=O_DARK)

pbox(RX, Q2, BW, BH,
     'Convert plume size + wind speed\nto emission rate  (kg CH₄/hr)',
     fc=O_FILL, ec=O_DARK)
varr(RX, Q2 + BH/2, Q3 - CH/2, color=O_DARK)

cyl(RX, Q3, CW, CH, 'Save result to\ndetection log')

# ── LEGEND — in the empty space above Q3, under "Quantification Path" header ──
leg_handles = [
    mpatches.Patch(fc=B_FILL,  ec=B_DARK,  lw=1.4, label='Detection step'),
    mpatches.Patch(fc=O_FILL,  ec=O_DARK,  lw=1.4, label='Quantification step'),
    mpatches.Patch(fc=SLATE,   ec=INK,     lw=1.4, label='Data source'),
    mpatches.Patch(fc=GREY_FC, ec=GREY_EC, lw=1.2, label='Excluded / no detection'),
]
# place legend centred in the top-right dead zone (above Q3, below header)
ax.legend(handles=leg_handles,
          loc='center', bbox_to_anchor=(0.73, 0.835),
          fontsize=8.8, framealpha=0.93, ncol=1,
          handlelength=1.3, handleheight=0.9,
          edgecolor='#BBBBBB', columnspacing=1.0)

# ── save ──────────────────────────────────────────────────────────────────────
ROOT = Path('/sessions/brave-gallant-mayer/mnt/methane-api')
out  = Path('/sessions/brave-gallant-mayer/mnt/Downloads/figure1_methodology.png')
out2 = ROOT / 'figures/figure1_methodology.png'
print('Saving …')
plt.savefig(out,  dpi=180, bbox_inches='tight', facecolor='white')
plt.savefig(out2, dpi=180, bbox_inches='tight', facecolor='white')
print(f'Saved → {out}')
plt.close()
