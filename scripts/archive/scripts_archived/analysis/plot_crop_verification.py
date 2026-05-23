"""
scripts/analysis/plot_crop_verification.py
Crop-verification map — pulls constants DIRECTLY from belchatow_annual_timeseries.py
so the map and the actual run code are guaranteed to be in sync.

Shows exactly where the S/C crop, each control crop, and the mine polygon will
fall when you run the timeseries. No hardcoded coords — everything is imported.

Run:
    conda activate methane
    python scripts/analysis/plot_crop_verification.py
    open results_analysis/belchatow_crop_verification.html
"""

import sys
import math
from pathlib import Path

# ── Import constants directly from the timeseries script ──────────────────────
_root = Path(__file__).resolve().parents[2]   # methane-api/
sys.path.insert(0, str(_root))                                  # makes `src` importable
sys.path.insert(0, str(_root / "scripts" / "detection"))        # apply_bitemporal_diff
sys.path.insert(0, str(_root / "scripts" / "timeseries"))       # belchatow_annual_timeseries
# detection path must be on sys.path BEFORE importing timeseries,
# because timeseries imports apply_bitemporal_diff at module level
from apply_bitemporal_diff import SC_CROP_PX
from belchatow_annual_timeseries import (
    LAT, LON,
    SC_OFFSET_N, SC_OFFSET_S, SC_OFFSET_E, SC_OFFSET_W,
    MINE_POLYGON_LATLON,
    SITE_NAME,
)

import folium
from folium import plugins

# ── Derived geometry ──────────────────────────────────────────────────────────
SC_CROP_DEG_LAT = (SC_CROP_PX * 10) / 2 / 111_000
SC_CROP_DEG_LON = (SC_CROP_PX * 10) / 2 / (111_000 * math.cos(math.radians(LAT)))

def km_to_deg_lat(km):      return km / 111.0
def km_to_deg_lon(km, lat): return km / (111.0 * math.cos(math.radians(lat)))

def bbox(center, half_lat, half_lon):
    return [[center[0]-half_lat, center[1]-half_lon],
            [center[0]+half_lat, center[1]+half_lon]]

# Centre of map: midpoint of mine polygon
poly_lats = [p[0] for p in MINE_POLYGON_LATLON]
poly_lons = [p[1] for p in MINE_POLYGON_LATLON]
map_center = [(min(poly_lats)+max(poly_lats))/2, (min(poly_lons)+max(poly_lons))/2]

m = folium.Map(location=map_center, zoom_start=11, tiles=None)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Satellite", show=False,
).add_to(m)

# ── Mine polygon (quantification boundary) ────────────────────────────────────
closed = MINE_POLYGON_LATLON + [MINE_POLYGON_LATLON[0]]
folium.Polygon(
    locations=closed,
    color="#E8851A", weight=2.5, dash_array="10 5",
    fill=True, fill_opacity=0.07,
    popup=folium.Popup(
        "<b>Mine polygon — quantification boundary</b><br>"
        "<i>Source: MINE_POLYGON_LATLON in belchatow_annual_timeseries.py</i><br><br>"
        + "".join(
            f"{'NW' if i==0 else 'NE' if i==1 else 'SE' if i==2 else 'SW'}: "
            f"{p[0]}°N, {p[1]}°E<br>"
            for i, p in enumerate(MINE_POLYGON_LATLON)
        ),
        max_width=280),
    tooltip="Mine quantification polygon (imported from timeseries script)",
).add_to(m)

# Corner labels
corner_labels = ["NW", "NE", "SE", "SW"]
for i, (plat, plon) in enumerate(MINE_POLYGON_LATLON):
    folium.Marker((plat, plon), icon=folium.DivIcon(
        html=f'<div style="font-size:8px;color:#E8851A;font-family:monospace;'
             f'background:rgba(255,255,255,0.88);padding:1px 4px;border-radius:2px;'
             f'white-space:nowrap;border:1px solid #E8851A;">'
             f'{corner_labels[i]} {plat}°N {plon}°E</div>',
        icon_size=(180, 14), icon_anchor=(0, 7),
    )).add_to(m)

# ── S/C site crop ─────────────────────────────────────────────────────────────
folium.Rectangle(
    bounds=bbox((LAT, LON), SC_CROP_DEG_LAT, SC_CROP_DEG_LON),
    color="#185FA5", weight=3,
    fill=True, fill_opacity=0.30,
    popup=folium.Popup(
        "<b>S/C site crop</b><br>"
        f"<i>Source: LAT={LAT}, LON={LON}, SC_CROP_PX={SC_CROP_PX}</i><br><br>"
        f"Centre: {LAT}°N, {LON}°E<br>"
        f"Size: {SC_CROP_PX*10}m × {SC_CROP_PX*10}m ({SC_CROP_PX}×{SC_CROP_PX} px at 10m)<br>"
        f"NW: {LAT+SC_CROP_DEG_LAT:.5f}°N, {LON-SC_CROP_DEG_LON:.5f}°E<br>"
        f"SE: {LAT-SC_CROP_DEG_LAT:.5f}°N, {LON+SC_CROP_DEG_LON:.5f}°E",
        max_width=280),
    tooltip=f"S/C site crop — {SC_CROP_PX}px = {SC_CROP_PX*10}m — centred {LAT}°N, {LON}°E",
).add_to(m)

# S/C crop corner labels
for lbl, coord in [
    ("NW", (LAT+SC_CROP_DEG_LAT, LON-SC_CROP_DEG_LON)),
    ("NE", (LAT+SC_CROP_DEG_LAT, LON+SC_CROP_DEG_LON)),
    ("SW", (LAT-SC_CROP_DEG_LAT, LON-SC_CROP_DEG_LON)),
    ("SE", (LAT-SC_CROP_DEG_LAT, LON+SC_CROP_DEG_LON)),
]:
    folium.Marker(coord, icon=folium.DivIcon(
        html=f'<div style="font-size:8px;color:#185FA5;font-family:monospace;'
             f'background:rgba(255,255,255,0.88);padding:1px 3px;border-radius:2px;'
             f'white-space:nowrap;">{coord[0]:.5f}°N {coord[1]:.5f}°E</div>',
        icon_size=(150, 14), icon_anchor=(75, 7),
    )).add_to(m)

# ── Control crops ─────────────────────────────────────────────────────────────
ctrl_dirs = [
    ("N", SC_OFFSET_N,    0,            "#185FA5"),
    ("S", -SC_OFFSET_S,   0,            "#185FA5"),
    ("E", 0,              SC_OFFSET_E,  "#185FA5"),
    ("W", 0,             -SC_OFFSET_W,  "#185FA5"),
]
offsets = {"N": SC_OFFSET_N, "S": SC_OFFSET_S, "E": SC_OFFSET_E, "W": SC_OFFSET_W}

for label, dlat, dlon, col in ctrl_dirs:
    cc = (LAT+dlat, LON+dlon)
    off = offsets[label]
    folium.Rectangle(
        bounds=bbox(cc, SC_CROP_DEG_LAT, SC_CROP_DEG_LON),
        color=col, weight=2, dash_array="5 4",
        fill=True, fill_opacity=0.15,
        popup=folium.Popup(
            f"<b>Control crop {label}</b><br>"
            f"<i>Source: SC_OFFSET_{label}={off}</i><br><br>"
            f"Centre: {cc[0]:.4f}°N, {cc[1]:.4f}°E<br>"
            f"Offset: {off}° (~{off*111:.0f} km) from site<br>"
            f"Size: {SC_CROP_PX*10}m × {SC_CROP_PX*10}m",
            max_width=260),
        tooltip=f"Control {label} — offset {off}° (~{off*111:.0f}km) — {cc[0]:.4f}°N {cc[1]:.4f}°E",
    ).add_to(m)
    folium.Marker(cc, icon=folium.DivIcon(
        html=f'<div style="font-size:9px;color:#185FA5;font-family:monospace;'
             f'background:rgba(255,255,255,0.88);padding:1px 3px;border-radius:2px;'
             f'white-space:nowrap;">ctrl {label} ({off}°)</div>',
        icon_size=(100, 14), icon_anchor=(50, 7),
    )).add_to(m)
    folium.PolyLine([(LAT, LON), cc], color=col, weight=1,
                    opacity=0.25, dash_array="3 6").add_to(m)

# ── Site centre marker ────────────────────────────────────────────────────────
folium.CircleMarker(
    (LAT, LON), radius=10, color="white", weight=2.5,
    fill=True, fill_color="#185FA5", fill_opacity=1.0,
    popup=folium.Popup(
        f"<b>S/C crop centre · S</b><br>"
        f"<code>{LAT}°N, {LON}°E</code><br><br>"
        f"Imported from belchatow_annual_timeseries.py<br>"
        f"<code>LAT, LON = {LAT}, {LON}</code>",
        max_width=240),
    tooltip=f"Site centre · {LAT}°N, {LON}°E",
).add_to(m)

# ── Controls ──────────────────────────────────────────────────────────────────
folium.LayerControl(position="bottomleft").add_to(m)
plugins.MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
plugins.MousePosition(
    position="bottomright", separator=" | ", prefix="",
    lat_formatter="function(n){return n.toFixed(5)+'°N';}",
    lng_formatter="function(n){return n.toFixed(5)+'°E';}",
).add_to(m)

# ── Legend ────────────────────────────────────────────────────────────────────
legend = f"""
<div style="position:fixed;top:10px;right:10px;z-index:1000;
  background:rgba(10,10,10,0.90);padding:14px 16px;border-radius:10px;
  color:white;font-family:monospace;font-size:11px;min-width:310px;
  box-shadow:0 2px 12px rgba(0,0,0,0.5);">
  <div style="font-size:13px;font-weight:700;color:#FFD700;margin-bottom:8px;">
    Crop Verification — {SITE_NAME}
  </div>
  <div style="color:#aaa;margin-bottom:10px;font-size:10px;">
    All values imported live from belchatow_annual_timeseries.py
  </div>
  <b>S/C site crop</b><br>
  &nbsp; LAT={LAT}, LON={LON}<br>
  &nbsp; SC_CROP_PX={SC_CROP_PX} → {SC_CROP_PX*10}m × {SC_CROP_PX*10}m<br><br>
  <b>Control offsets</b><br>
  &nbsp; N: {SC_OFFSET_N}° (~{SC_OFFSET_N*111:.0f}km)<br>
  &nbsp; S: {SC_OFFSET_S}° (~{SC_OFFSET_S*111:.0f}km)<br>
  &nbsp; E: {SC_OFFSET_E}° (~{SC_OFFSET_E*111:.0f}km)<br>
  &nbsp; W: {SC_OFFSET_W}° (~{SC_OFFSET_W*111:.0f}km)<br><br>
  <b>Mine polygon corners</b><br>
  {''.join(f"&nbsp; {'NW' if i==0 else 'NE' if i==1 else 'SE' if i==2 else 'SW'}: {p[0]}°N {p[1]}°E<br>" for i, p in enumerate(MINE_POLYGON_LATLON))}
  <div style="border-top:1px solid #333;margin-top:8px;padding-top:6px;
    font-size:9px;color:#777;">
    Cursor coords bottom-right · Ruler top-left · Satellite bottom-left
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend))

out = Path("results_analysis/belchatow_crop_verification.html")
out.parent.mkdir(parents=True, exist_ok=True)
m.save(str(out))
print(f"Saved → {out}")
print(f"  S/C crop centre: {LAT}°N, {LON}°E  ({SC_CROP_PX}px = {SC_CROP_PX*10}m)")
print(f"  Control offsets: N={SC_OFFSET_N}° S={SC_OFFSET_S}° E={SC_OFFSET_E}° W={SC_OFFSET_W}°")
print(f"  Mine polygon: {len(MINE_POLYGON_LATLON)} corners, "
      f"lon {min(poly_lons):.3f}°–{max(poly_lons):.3f}°E  "
      f"lat {min(poly_lats):.3f}°–{max(poly_lats):.3f}°N")
print("Run:  open results_analysis/belchatow_crop_verification.html")
