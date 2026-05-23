"""
scripts/analysis/plot_belchatow_map.py
Definitive Bełchatów detection + quantification map.

Shows four distinct boundaries:
  1. S/C site crop   — 1km² centred on Climate TRACE KWB mine (51.242°N, 19.275°E)
  2. Control crops   — 4 × 1km² at 0.2° offset (~22km) — the background reference
  3. Mine polygon    — exact OSM mine boundary used for quantification
  4. Detection centroid — where CH4Net probability peaked

Run:
    conda activate methane
    pip install folium
    python scripts/analysis/plot_belchatow_map.py
    open results_analysis/belchatow_map.html
"""

import folium
from folium import plugins
import math

# ── EXACT COORDINATES ─────────────────────────────────────────────────────────
DET = (51.2495, 19.2227)   # CH4Net detection centroid (probability-weighted)
CT  = (51.2420, 19.2754)   # Climate TRACE mine asset 16168
SRC = (51.242,  19.275)    # S/C site crop centre — Climate TRACE KWB mine centroid
                            # Fixed from previous 51.266, 19.315 (power station)

# KWB Bełchatów mine polygon — exact OSM boundary corners
MINE_POLYGON = [
    (51.257,  19.097),   # NW
    (51.2566, 19.390),   # NE
    (51.219,  19.3996),  # SE
    (51.2185, 19.099),   # SW
    (51.257,  19.097),   # close polygon
]

WIND_FROM_DEG = 245.0
WIND_MS       = 2.19

CTRL_OFFSET     = 0.20   # degrees (~22 km) — N/S
CTRL_OFFSET_E   = 0.30   # degrees (~33 km) — E
CTRL_OFFSET_W   = 0.39   # degrees (~43 km) — W (E + ~10 km)

def km_to_deg_lat(km):      return km / 111.0
def km_to_deg_lon(km, lat): return km / (111.0 * math.cos(math.radians(lat)))

SC_CROP_DEG_LAT = (100 * 10) / 2 / 111_000   # half of 1km in degrees lat
SC_CROP_DEG_LON = (100 * 10) / 2 / (111_000 * math.cos(math.radians(SRC[0])))

def bbox(center, half_lat, half_lon):
    return [[center[0]-half_lat, center[1]-half_lon],
            [center[0]+half_lat, center[1]+half_lon]]

# ── MAP ───────────────────────────────────────────────────────────────────────
# Centre map on midpoint of mine polygon, zoom out enough to show full extent
m = folium.Map(location=[51.242, 19.230], zoom_start=10, tiles=None)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Satellite", show=False,
).add_to(m)

# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY 3: Mine polygon — quantification boundary
# Drawn first so other layers render on top
# ─────────────────────────────────────────────────────────────────────────────
folium.Polygon(
    locations=MINE_POLYGON,
    color="#E8851A", weight=2.5, dash_array="10 5",
    fill=True, fill_opacity=0.07,
    popup=folium.Popup(
        "<b>KWB Bełchatów Mine — quantification boundary</b><br>"
        "Exact OSM polygon used to bound CEMF+IME mass retrieval.<br><br>"
        "<table style='font-size:11px;'>"
        f"<tr><td>NW</td><td>51.2570°N, 19.0970°E</td></tr>"
        f"<tr><td>NE</td><td>51.2566°N, 19.3900°E</td></tr>"
        f"<tr><td>SE</td><td>51.2190°N, 19.3996°E</td></tr>"
        f"<tr><td>SW</td><td>51.2185°N, 19.0990°E</td></tr>"
        "</table><br>"
        "Width: ~21 km · Height: ~4.2 km<br>"
        "B11/B12 spectral mass retrieved within this polygon → Q (kg/h)",
        max_width=300),
    tooltip="Mine quantification polygon — click for corner coordinates",
).add_to(m)

# Corner coordinate labels on the polygon
for label, coord in [
    ("NW  51.2570°N 19.0970°E", (51.257,  19.097)),
    ("NE  51.2566°N 19.3900°E", (51.2566, 19.390)),
    ("SE  51.2190°N 19.3996°E", (51.219,  19.3996)),
    ("SW  51.2185°N 19.0990°E", (51.2185, 19.099)),
]:
    folium.Marker(coord, icon=folium.DivIcon(
        html=f'<div style="font-size:8px;color:#E8851A;font-family:monospace;'
             f'background:rgba(255,255,255,0.85);padding:1px 4px;border-radius:2px;'
             f'white-space:nowrap;border:1px solid rgba(232,133,26,0.4);">{label}</div>',
        icon_size=(200, 14), icon_anchor=(0, 7),
    )).add_to(m)

# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY 2: Control crops — 4 × 1km² at 0.2° offset
# ─────────────────────────────────────────────────────────────────────────────
ctrl_dirs = [
    ("N", CTRL_OFFSET,    0),
    ("S", -CTRL_OFFSET,   0),
    ("E", 0,              CTRL_OFFSET_E),
    ("W", 0,             -CTRL_OFFSET_W),
]

for label, dlat, dlon in ctrl_dirs:
    cc = (SRC[0]+dlat, SRC[1]+dlon)
    offset_used = {"N": CTRL_OFFSET, "S": CTRL_OFFSET, "E": CTRL_OFFSET_E, "W": CTRL_OFFSET_W}[label]
    offset_km   = offset_used * 111.0
    folium.Rectangle(
        bounds=bbox(cc, SC_CROP_DEG_LAT, SC_CROP_DEG_LON),
        color="#185FA5", weight=2, dash_array="5 4",
        fill=True, fill_opacity=0.15,
        popup=folium.Popup(
            f"<b>Control crop {label}</b><br>"
            f"Centre: {cc[0]:.4f}°N, {cc[1]:.4f}°E<br>"
            f"Size: 1 km × 1 km (100 px × 100 px at 10 m)<br>"
            f"Offset: {offset_used}° (~{offset_km:.0f} km) from source<br><br>"
            f"Mean CH4Net probability here = background reference.<br>"
            f"S/C = site mean ÷ mean(all valid control crops).<br>"
            f"<i>E/W offset = 1.5× N/S offset to clear mine footprint</i>",
            max_width=270),
        tooltip=f"Control crop {label} — background reference (click for coords)",
    ).add_to(m)
    folium.PolyLine([SRC, cc], color="#185FA5", weight=1,
                    opacity=0.25, dash_array="3 6").add_to(m)

# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY 1: S/C site crop — 1km² — centred on mine
# ─────────────────────────────────────────────────────────────────────────────
folium.Rectangle(
    bounds=bbox(SRC, SC_CROP_DEG_LAT, SC_CROP_DEG_LON),
    color="#185FA5", weight=3, dash_array=None,
    fill=True, fill_opacity=0.30,
    popup=folium.Popup(
        "<b>S/C site crop — analysis window</b><br>"
        f"Centre: {SRC[0]}°N, {SRC[1]}°E<br>"
        f"Size: 1 km × 1 km (100 px × 100 px at 10 m)<br><br>"
        "Mean CH4Net probability inside this box = the signal.<br>"
        "Divided by mean of 4 control crops = S/C ratio.<br>"
        f"S/C = <b>27.3×</b> on Aug 24 2024 → above τ = 3.58.<br><br>"
        "Centred on Climate TRACE KWB mine (corrected).<br>"
        "<i>Code: LAT, LON = 51.242, 19.275</i>",
        max_width=280),
    tooltip="S/C site crop — 1km² · centred on mine · S/C=27.3 (click for coords)",
).add_to(m)

# S/C crop corner labels
for lbl, coord in [
    ("NW", (SRC[0]+SC_CROP_DEG_LAT, SRC[1]-SC_CROP_DEG_LON)),
    ("NE", (SRC[0]+SC_CROP_DEG_LAT, SRC[1]+SC_CROP_DEG_LON)),
    ("SW", (SRC[0]-SC_CROP_DEG_LAT, SRC[1]-SC_CROP_DEG_LON)),
    ("SE", (SRC[0]-SC_CROP_DEG_LAT, SRC[1]+SC_CROP_DEG_LON)),
]:
    folium.Marker(coord, icon=folium.DivIcon(
        html=f'<div style="font-size:8px;color:#185FA5;font-family:monospace;'
             f'background:rgba(255,255,255,0.85);padding:1px 3px;border-radius:2px;'
             f'white-space:nowrap;">{coord[0]:.4f}°N {coord[1]:.4f}°E</div>',
        icon_size=(140, 14), icon_anchor=(70, 7),
    )).add_to(m)

# ─────────────────────────────────────────────────────────────────────────────
# Wind arrow from detection centroid
# ─────────────────────────────────────────────────────────────────────────────
wind_toward = (WIND_FROM_DEG + 180) % 360
dist = 2.5
tip = (DET[0] + km_to_deg_lat(dist * math.cos(math.radians(wind_toward))),
       DET[1] + km_to_deg_lon(dist * math.sin(math.radians(wind_toward)), DET[0]))
folium.PolyLine([DET, tip], color="#E8851A", weight=2.5, opacity=0.9).add_to(m)
folium.RegularPolygonMarker(
    tip, number_of_sides=3, radius=7,
    color="#E8851A", fill_color="#E8851A", fill_opacity=1.0,
    rotation=wind_toward - 90,
    tooltip=f"Wind FROM {WIND_FROM_DEG}° WSW → plume drifts ENE · {WIND_MS} m/s ERA5",
).add_to(m)

# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────
def pin(loc, color, label, popup_html, tip):
    folium.CircleMarker(loc, radius=11, color="white", weight=2.5,
        fill=True, fill_color=color, fill_opacity=1.0,
        popup=folium.Popup(popup_html, max_width=280),
        tooltip=tip).add_to(m)
    folium.Marker(loc, icon=folium.DivIcon(
        html=f'<div style="font-size:10px;font-weight:700;color:white;'
             f'text-align:center;margin-top:-16px;margin-left:-8px;'
             f'width:16px;line-height:22px;">{label}</div>',
        icon_size=(16,22))).add_to(m)

pin(DET, "#D85A30", "★",
    f"<b>Detection centroid</b><br>"
    f"<code>{DET[0]}°N, {DET[1]}°E</code><br><br>"
    f"Probability-weighted centroid of CH4Net mask across full T34UCB tile.<br>"
    f"Western edge of mine — plume drifts ENE on wind FROM {WIND_FROM_DEG}°.<br><br>"
    f"S/C = 27.3× · Aug 24 2024 · T34UCB",
    f"Detection centroid · {DET[0]}°N, {DET[1]}°E")

pin(SRC, "#185FA5", "S",
    f"<b>S/C site crop centre</b><br>"
    f"<code>{SRC[0]}°N, {SRC[1]}°E</code><br><br>"
    f"Climate TRACE KWB Bełchatów mine centroid (asset 16168).<br>"
    f"Corrected from previous 51.266°N, 19.315°E (power station).<br>"
    f"<i>belchatow_annual_timeseries.py · LAT, LON = 51.242, 19.275</i>",
    f"S/C crop centre · {SRC[0]}°N, {SRC[1]}°E · KWB mine")

pin(CT, "#3B6D11", "CT",
    f"<b>Climate TRACE mine centroid</b><br>"
    f"<code>{CT[0]}°N, {CT[1]}°E</code><br><br>"
    f"Asset 16168 · KWB Bełchatów · 29,636 t CH₄/yr (2024)<br>"
    f"Reference point — S/C crop now aligned to this location.",
    f"Climate TRACE · {CT[0]}°N, {CT[1]}°E")

# ─────────────────────────────────────────────────────────────────────────────
# Controls + mouse position
# ─────────────────────────────────────────────────────────────────────────────
folium.LayerControl(position="bottomleft").add_to(m)
plugins.MeasureControl(position="topleft", primary_length_unit="kilometers").add_to(m)
plugins.MousePosition(
    position="bottomright", separator=" | ", prefix="",
    lat_formatter="function(n){return n.toFixed(4)+'°N';}",
    lng_formatter="function(n){return n.toFixed(4)+'°E';}",
).add_to(m)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend = """
<div style="position:fixed;top:10px;right:10px;z-index:1000;
  background:rgba(10,10,10,0.88);padding:16px 18px;border-radius:10px;
  color:white;font-family:sans-serif;font-size:12px;min-width:280px;
  box-shadow:0 2px 12px rgba(0,0,0,0.5);">

  <div style="font-size:14px;font-weight:700;color:#FFD700;margin-bottom:4px;">
    CH4Net · Bełchatów · Aug 24 2024
  </div>
  <div style="font-size:11px;color:#bbb;margin-bottom:12px;">
    T34UCB · Sentinel-2B · S/C = 27.3× · τ = 3.5796
  </div>

  <table style="border-collapse:collapse;width:100%;">
    <tr style="border-bottom:1px solid #333;">
      <td style="padding:6px 0;vertical-align:top;width:28px;">
        <div style="width:20px;height:20px;border:3px solid #185FA5;
          background:rgba(24,95,165,0.30);"></div>
      </td>
      <td style="padding:6px 0 6px 6px;">
        <b>S/C site crop — analysis window</b><br>
        <span style="color:#aaa;font-size:11px;">
          1km × 1km · 100×100 px at 10m<br>
          Centred on KWB mine · 51.242°N, 19.275°E (corrected)
        </span>
      </td>
    </tr>
    <tr style="border-bottom:1px solid #333;">
      <td style="padding:6px 0;vertical-align:top;">
        <div style="width:20px;height:20px;border:2px dashed #185FA5;
          background:rgba(24,95,165,0.15);"></div>
      </td>
      <td style="padding:6px 0 6px 6px;">
        <b>Control crops × 4</b><br>
        <span style="color:#aaa;font-size:11px;">
          1km × 1km each · N/S: 0.20° (~22km) · E: 0.30° (~33km) · W: 0.39° (~43km)<br>
          Background reference for S/C denominator
        </span>
      </td>
    </tr>
    <tr style="border-bottom:1px solid #333;">
      <td style="padding:6px 0;vertical-align:top;">
        <div style="width:20px;height:20px;border:2px dashed #E8851A;
          background:rgba(232,133,26,0.08);"></div>
      </td>
      <td style="padding:6px 0 6px 6px;">
        <b>Mine quantification polygon</b><br>
        <span style="color:#aaa;font-size:11px;">
          KWB Bełchatów exact boundary · ~21km × 4.2km<br>
          NW 51.257°N 19.097°E → NE 51.2566°N 19.390°E<br>
          SE 51.219°N 19.3996°E → SW 51.2185°N 19.099°E<br>
          CEMF+IME mass retrieval bounded here → Q (kg/h)
        </span>
      </td>
    </tr>
    <tr>
      <td style="padding:6px 0;vertical-align:top;">
        <div style="display:flex;flex-direction:column;gap:4px;margin-top:4px;">
          <div style="width:20px;height:20px;border-radius:50%;
            background:#D85A30;text-align:center;line-height:20px;
            font-size:11px;">★</div>
          <div style="width:20px;height:20px;border-radius:50%;
            background:#185FA5;text-align:center;line-height:20px;
            font-size:10px;font-weight:700;">S</div>
          <div style="width:20px;height:20px;border-radius:50%;
            background:#3B6D11;text-align:center;line-height:20px;
            font-size:9px;font-weight:700;">CT</div>
        </div>
      </td>
      <td style="padding:6px 0 6px 6px;vertical-align:top;">
        <span style="color:#D85A30;font-weight:600;">★</span>
        Detection centroid · 51.2495°N, 19.2227°E<br>
        <span style="color:#6699cc;font-weight:600;">S</span>
        S/C crop centre · 51.242°N, 19.275°E <i style="color:#aaa;">(KWB mine — corrected)</i><br>
        <span style="color:#7fc96d;font-weight:600;">CT</span>
        Climate TRACE · 51.2420°N, 19.2754°E<br>
        <span style="color:#E8851A;">→</span>
        Wind FROM 245° WSW · 2.19 m/s ERA5
      </td>
    </tr>
  </table>

  <div style="border-top:1px solid #333;margin-top:8px;padding-top:6px;
    font-size:10px;color:#777;">
    Cursor coords bottom-right · Ruler top-left · Satellite toggle bottom-left
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend))

# Fit bounds to all meaningful content: S crop (south) → N crop (north),
# W crop (west) → E crop (east), with a small pad so no label is clipped.
north_crop_lat = SRC[0] + CTRL_OFFSET   # 51.442
south_crop_lat = SRC[0] - CTRL_OFFSET   # 51.042
west_crop_lon  = SRC[1] - CTRL_OFFSET_W # 18.885
east_crop_lon  = SRC[1] + CTRL_OFFSET_E # 19.575
m.fit_bounds(
    [[south_crop_lat - 0.02, west_crop_lon - 0.02],
     [north_crop_lat + 0.02, east_crop_lon + 0.02]]
)

out = "results_analysis/belchatow_map.html"
m.save(out)
print(f"Saved → {out}")
print("Run:  open results_analysis/belchatow_map.html")
