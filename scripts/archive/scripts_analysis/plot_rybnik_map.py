"""
scripts/analysis/plot_rybnik_map.py
=====================================
Interactive analysis boundary map for KWK ROW Ruch Chwałowice, Rybnik.
(Polska Grupa Górnicza S.A. — hard coal, Upper Silesia, Poland)

All constants imported directly from rybnik_chwalowice_annual_timeseries.py
so the map and the timeseries code are guaranteed to be in sync.

Shows:
  ● S/C detection crop         — 1km × 1km blue square at CM pin
  ● Control crops (N/S/E/W)    — same size, ±0.20° symmetric offsets
  ● KWK Chwałowice polygon     — approximate concession boundary
  ● Carbon Mapper detections   — all 6 events with date + flow rate labels
  ● CH4Net detection centroid  — 2025-03-22 probability centroid
  ● Wind arrows                — one per CM detection with ERA5/ECMWF direction

Run:
    conda activate methane
    python scripts/analysis/plot_rybnik_map.py
    open results_analysis/rybnik_map.html
"""

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts" / "detection"))
sys.path.insert(0, str(_root / "scripts" / "timeseries"))

# Import all constants directly from the timeseries script
from apply_bitemporal_diff import SC_CROP_PX
from rybnik_chwalowice_annual_timeseries import (
    LAT, LON,
    SC_OFFSET_N, SC_OFFSET_S, SC_OFFSET_E, SC_OFFSET_W,
    MINE_POLYGON_LATLON,
    SITE_NAME,
    TILE_ID,
    DETECTION_THRESHOLD,
    CONFORMAL_TAU,
    CM_CSV,
)

import folium
from folium import plugins

# ── CM detection centroid (rybnik_centroid_vs_wind.json) ──────────────────────
# Probability-weighted centroid of CH4Net prob ≥ 0.50 pixels, S2B 2025-03-22
CENTROID_LAT, CENTROID_LON = 50.06147, 18.520388

# ── Derived geometry ──────────────────────────────────────────────────────────
SC_CROP_DEG_LAT = (SC_CROP_PX * 10) / 2 / 111_000
SC_CROP_DEG_LON = (SC_CROP_PX * 10) / 2 / (111_000 * math.cos(math.radians(LAT)))

def km_to_deg_lat(km):      return km / 111.0
def km_to_deg_lon(km, lat): return km / (111.0 * math.cos(math.radians(lat)))

def bbox(center, half_lat, half_lon):
    return [[center[0]-half_lat, center[1]-half_lon],
            [center[0]+half_lat, center[1]+half_lon]]

# ── Load CM detections from CSV ───────────────────────────────────────────────
cm_events = []
if CM_CSV.exists():
    with open(CM_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dt_str = row.get("datetime", "").strip()
            if not dt_str:
                continue
            dt = datetime.fromisoformat(dt_str.replace("+00", "+00:00"))
            em_str = row.get("emission_auto", "").strip()
            ws_str = row.get("wind_speed_avg_auto", "").strip()
            wd_str = row.get("wind_direction_avg_auto", "").strip()
            cm_events.append({
                "date":        dt.strftime("%Y-%m-%d"),
                "time_utc":    dt.strftime("%H:%M"),
                "instrument":  row.get("instrument", "").upper(),
                "platform":    row.get("platform", ""),
                "emission_kgh": float(em_str) if em_str else None,
                "wind_speed_ms": float(ws_str) if ws_str else None,
                "wind_dir_deg":  float(wd_str) if wd_str else None,
                "plume_lat":    float(row.get("plume_latitude", LAT)),
                "plume_lon":    float(row.get("plume_longitude", LON)),
            })

# Sort chronologically
cm_events.sort(key=lambda x: x["date"])

# ── Map ───────────────────────────────────────────────────────────────────────
# Centre between the S/C crop and the CH4Net centroid
map_center = [(LAT + CENTROID_LAT) / 2, (LON + CENTROID_LON) / 2]
m = folium.Map(location=map_center, zoom_start=12, tiles=None)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap", show=True).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri", name="Esri Satellite", show=False,
).add_to(m)

# ── KWK Chwałowice mine polygon ───────────────────────────────────────────────
closed = MINE_POLYGON_LATLON + [MINE_POLYGON_LATLON[0]]
folium.Polygon(
    locations=closed,
    color="#E8851A", weight=2.5, dash_array="10 5",
    fill=True, fill_opacity=0.09,
    popup=folium.Popup(
        "<b>KWK ROW Ruch Chwałowice — quantification boundary</b><br>"
        "<i>Polska Grupa Górnicza S.A., Rybnik</i><br>"
        "<i>d. 'Donnersmarck Grube' — approx. concession area</i><br><br>"
        f"NW: {MINE_POLYGON_LATLON[0][0]}°N {MINE_POLYGON_LATLON[0][1]}°E<br>"
        f"NE: {MINE_POLYGON_LATLON[1][0]}°N {MINE_POLYGON_LATLON[1][1]}°E<br>"
        f"SE: {MINE_POLYGON_LATLON[2][0]}°N {MINE_POLYGON_LATLON[2][1]}°E<br>"
        f"SW: {MINE_POLYGON_LATLON[3][0]}°N {MINE_POLYGON_LATLON[3][1]}°E<br><br>"
        "~7.5 km E-W × ~3.6 km N-S<br>"
        "Underground hard coal · 243.7 Mt estimated reserves",
        max_width=320),
    tooltip="KWK Chwałowice — quantification boundary (imported from timeseries script)",
).add_to(m)

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
        "<b>S/C site crop — KWK Chwałowice</b><br>"
        f"<i>Imported from rybnik_chwalowice_annual_timeseries.py</i><br><br>"
        f"Centre: {LAT}°N, {LON}°E  (Carbon Mapper pin)<br>"
        f"Size: {SC_CROP_PX*10}m × {SC_CROP_PX*10}m ({SC_CROP_PX}×{SC_CROP_PX} px at 10m)<br>"
        f"NW: {LAT+SC_CROP_DEG_LAT:.5f}°N, {LON-SC_CROP_DEG_LON:.5f}°E<br>"
        f"SE: {LAT-SC_CROP_DEG_LAT:.5f}°N, {LON+SC_CROP_DEG_LON:.5f}°E<br><br>"
        f"S/C threshold: {DETECTION_THRESHOLD}  |  τ={CONFORMAL_TAU}  (α=0.10, n=35)<br>"
        f"skip_bitemporal=True",
        max_width=320),
    tooltip=f"S/C crop — {SC_CROP_PX}px = {SC_CROP_PX*10}m — {LAT}°N, {LON}°E",
).add_to(m)

# S/C crop corner coordinate labels
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
        icon_size=(155, 14), icon_anchor=(75, 7),
    )).add_to(m)

# ── Control crops ─────────────────────────────────────────────────────────────
ctrl_dirs = [
    ("N",  SC_OFFSET_N,    0.0,         "#185FA5"),
    ("S", -SC_OFFSET_S,    0.0,         "#185FA5"),
    ("E",  0.0,            SC_OFFSET_E, "#185FA5"),
    ("W",  0.0,           -SC_OFFSET_W, "#185FA5"),
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
            f"Offset: {off}° (~{off*111:.0f} km) from site<br>"
            f"Centre: {cc[0]:.4f}°N, {cc[1]:.4f}°E<br>"
            f"Size: {SC_CROP_PX*10}m × {SC_CROP_PX*10}m",
            max_width=240),
        tooltip=f"Control {label} — {off}° (~{off*111:.0f}km)",
    ).add_to(m)
    folium.Marker(cc, icon=folium.DivIcon(
        html=f'<div style="font-size:9px;color:#185FA5;font-family:monospace;'
             f'background:rgba(255,255,255,0.88);padding:1px 3px;border-radius:2px;'
             f'white-space:nowrap;">ctrl {label} ({off}°)</div>',
        icon_size=(100, 14), icon_anchor=(50, 7),
    )).add_to(m)
    folium.PolyLine([(LAT, LON), cc], color=col, weight=1,
                    opacity=0.25, dash_array="3 6").add_to(m)

# ── S/C crop centre (= CM pin) ────────────────────────────────────────────────
folium.CircleMarker(
    (LAT, LON), radius=10, color="white", weight=2.5,
    fill=True, fill_color="#185FA5", fill_opacity=1.0,
    popup=folium.Popup(
        f"<b>S/C crop centre = Carbon Mapper source pin</b><br>"
        f"<code>{LAT}°N, {LON}°E</code><br><br>"
        f"KWK ROW Ruch Chwałowice (PGG S.A.)<br>"
        f"Imported: LAT, LON from rybnik_chwalowice_annual_timeseries.py",
        max_width=280),
    tooltip=f"S/C centre / CM pin · {LAT}°N, {LON}°E",
).add_to(m)

# ── CH4Net probability centroid (2025-03-22) ──────────────────────────────────
folium.CircleMarker(
    (CENTROID_LAT, CENTROID_LON), radius=9,
    color="white", weight=2.5,
    fill=True, fill_color="#FFD700", fill_opacity=1.0,
    popup=folium.Popup(
        "<b>CH4Net detection centroid</b><br>"
        f"<code>{CENTROID_LAT}°N, {CENTROID_LON}°E</code><br><br>"
        "Probability-weighted centroid of CH4Net prob ≥ 0.50<br>"
        "within 3 km of CM pin · S2B 2025-03-22 09:50 UTC<br>"
        "597 active pixels · max prob = 0.696<br><br>"
        "<i>Source: rybnik_centroid_vs_wind.json</i>",
        max_width=300),
    tooltip=f"CH4Net centroid · {CENTROID_LAT}°N, {CENTROID_LON}°E · 2025-03-22",
).add_to(m)
folium.Marker((CENTROID_LAT, CENTROID_LON), icon=folium.DivIcon(
    html=f'<div style="font-size:9px;color:#FFD700;font-weight:bold;font-family:monospace;'
         f'background:rgba(0,0,0,0.85);padding:2px 5px;border-radius:3px;'
         f'white-space:nowrap;border:1px solid #FFD700;margin-top:14px;">'
         f'CH4Net centroid {CENTROID_LAT:.5f}°N {CENTROID_LON:.5f}°E</div>',
    icon_size=(280, 16), icon_anchor=(10, -8),
)).add_to(m)

# Plume displacement line: CM pin → centroid
folium.PolyLine(
    [(LAT, LON), (CENTROID_LAT, CENTROID_LON)],
    color="#FFD700", weight=2, opacity=0.7, dash_array="6 4",
    tooltip="Plume displacement 2025-03-22: 2,558 m, bearing 224° SW from CM pin to CH4Net centroid",
).add_to(m)

# ── Carbon Mapper detection events ────────────────────────────────────────────
# Instrument colours
INST_COLOR = {"TAN": "#FF6B35", "EMI": "#9B59B6"}

for ev in cm_events:
    col = INST_COLOR.get(ev["instrument"], "#FF6B35")
    em  = f"{ev['emission_kgh']:.0f} kg/h" if ev["emission_kgh"] else "(no auto-emission)"
    ws  = f"{ev['wind_speed_ms']:.2f} m/s" if ev["wind_speed_ms"] else "—"
    wd  = f"{ev['wind_dir_deg']:.0f}°" if ev["wind_dir_deg"] else "—"

    # Plume marker at plume centroid lat/lon (from CSV)
    folium.CircleMarker(
        (ev["plume_lat"], ev["plume_lon"]),
        radius=7, color=col, weight=2,
        fill=True, fill_color=col, fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{ev['instrument']} detection — {ev['date']}</b><br>"
            f"Time (UTC): {ev['time_utc']}<br>"
            f"Platform: {ev['platform']}<br>"
            f"Emission: <b>{em}</b><br>"
            f"Wind: FROM {wd} at {ws} (ECMWF IFS)",
            max_width=260),
        tooltip=f"{ev['instrument']} {ev['date']} — {em}",
    ).add_to(m)

    # Date + emission label
    folium.Marker((ev["plume_lat"], ev["plume_lon"]), icon=folium.DivIcon(
        html=f'<div style="font-size:8px;color:{col};font-weight:bold;font-family:monospace;'
             f'background:rgba(0,0,0,0.82);padding:2px 4px;border-radius:3px;'
             f'white-space:nowrap;border:1px solid {col};margin-top:14px;">'
             f'{ev["date"]}  {ev["instrument"]}  {em}</div>',
        icon_size=(230, 16), icon_anchor=(10, -8),
    )).add_to(m)

    # Wind arrow (3 km, pointing in plume direction = FROM wind_dir + 180°)
    if ev["wind_dir_deg"] is not None and ev["wind_speed_ms"] is not None:
        plume_dir = (ev["wind_dir_deg"] + 180.0) % 360.0
        bearing_rad = math.radians(plume_dir)
        arrow_end = (
            ev["plume_lat"] + km_to_deg_lat(2.5) * math.cos(bearing_rad),
            ev["plume_lon"] + km_to_deg_lon(2.5, ev["plume_lat"]) * math.sin(bearing_rad),
        )
        folium.PolyLine(
            [(ev["plume_lat"], ev["plume_lon"]), arrow_end],
            color=col, weight=2.5, opacity=0.75, dash_array="4 3",
            tooltip=f"Plume direction {ev['date']}: FROM {ev['wind_dir_deg']:.0f}° → TO {plume_dir:.0f}°",
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
cm_lines = ""
for ev in cm_events:
    em = f"{ev['emission_kgh']:.0f} kg/h" if ev["emission_kgh"] else "no auto-emission"
    col = INST_COLOR.get(ev["instrument"], "#FF6B35")
    cm_lines += (
        f'&nbsp; <span style="color:{col};">●</span>'
        f' {ev["date"]}  {ev["instrument"]}  {em}<br>'
    )

legend = f"""
<div style="position:fixed;top:10px;right:10px;z-index:1000;
  background:rgba(10,10,10,0.91);padding:14px 16px;border-radius:10px;
  color:white;font-family:monospace;font-size:11px;min-width:330px;
  box-shadow:0 2px 12px rgba(0,0,0,0.5);">
  <div style="font-size:13px;font-weight:700;color:#FFD700;margin-bottom:6px;">
    KWK ROW Ruch Chwałowice · Rybnik ({TILE_ID})
  </div>
  <div style="color:#aaa;margin-bottom:10px;font-size:10px;">
    All values imported from rybnik_chwalowice_annual_timeseries.py<br>
    PGG S.A. · underground hard coal · 243.7 Mt reserves
  </div>

  <span style="color:#185FA5;">■</span> <b>S/C site crop</b><br>
  &nbsp; Centre (= CM pin): {LAT}°N, {LON}°E<br>
  &nbsp; {SC_CROP_PX}px = {SC_CROP_PX*10}m · skip_bitemporal=True<br>
  &nbsp; τ={CONFORMAL_TAU} · S/C thresh={DETECTION_THRESHOLD}<br><br>

  <span style="color:#185FA5;">□</span> <b>Control crops</b> (N/S/E/W)<br>
  &nbsp; {SC_OFFSET_N}° (~{SC_OFFSET_N*111:.0f}km) all directions<br><br>

  <span style="color:#E8851A;">■</span> <b>KWK Chwałowice polygon</b><br>
  &nbsp; Approx. concession ~7.5km × 3.6km<br><br>

  <span style="color:#FFD700;">●</span> <b>CH4Net centroid</b> (2025-03-22)<br>
  &nbsp; {CENTROID_LAT}°N, {CENTROID_LON}°E<br>
  &nbsp; 2,558 m SW of CM pin · 597 px ≥ 0.50<br><br>

  <b>Carbon Mapper detections ({len(cm_events)} total)</b><br>
  {cm_lines}
  <div style="border-top:1px solid #333;margin-top:8px;padding-top:6px;
    font-size:9px;color:#777;">
    Cursor coords bottom-right · Ruler top-left · Satellite toggle bottom-left
  </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend))

# ── Save ──────────────────────────────────────────────────────────────────────
out = Path("results_analysis/rybnik_map.html")
out.parent.mkdir(parents=True, exist_ok=True)
m.save(str(out))
print(f"Saved → {out}")
print(f"  S/C crop: {LAT}°N, {LON}°E  ({SC_CROP_PX}px = {SC_CROP_PX*10}m)")
print(f"  Control offsets: N={SC_OFFSET_N}° S={SC_OFFSET_S}° E={SC_OFFSET_E}° W={SC_OFFSET_W}°")
print(f"  Mine polygon: {len(MINE_POLYGON_LATLON)} corners (KWK Chwałowice)")
print(f"  CM detections loaded: {len(cm_events)}")
for ev in cm_events:
    em = f"{ev['emission_kgh']:.0f} kg/h" if ev["emission_kgh"] else "no auto-emission"
    print(f"    {ev['date']}  {ev['instrument']:5s}  {em}")
print("Run:  open results_analysis/rybnik_map.html")
