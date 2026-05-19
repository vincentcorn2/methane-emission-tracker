"""
demo_pipeline.py — End-to-end demonstration of the emission tracking pipeline.

Run this after downloading the WRI database CSV to data/ folder.

This script shows the full chain:
  1. (Simulated) satellite detection → plume at known coordinates
  2. Entity resolution → nearest power plant from WRI database
  3. Quantification → estimated flow rate
  4. Financial impact → IRA Waste Emissions Charge calculation
  5. Output → structured JSON matching the API schema
"""
import json
import uuid
import numpy as np
from datetime import datetime

# Import our modules
from src.entity_resolution.resolver import (
    AssetDatabase,
    CorporateResolver,
    ResolvedEmission,
)
from src.quantification.ime import IntegratedMassEnhancement


def main():
    print("=" * 70)
    print("METHANE EMISSION TRACKER — Pipeline Demo")
    print("=" * 70)

    # ──────────────────────────────────────────
    # Step 1: Simulate a CH4Net detection
    # ──────────────────────────────────────────
    print("\n[1] Simulating CH4Net detection at known super-emitter site T7...")

    # These coordinates are from the CH4Net paper Table 2
    # Site T7: Turkmenistan, 39% of images contain emissions
    detected_lat = 39.45965
    detected_lon = 53.77921
    confidence = 0.87

    # Simulate a plume mask (in practice, this comes from CH4Net inference)
    fake_mask = np.zeros((100, 100), dtype=np.uint8)
    # Create a plume-shaped region (~280 pixels)
    fake_mask[30:55, 40:55] = 1
    fake_mask[35:50, 55:65] = 1
    fake_mask[40:48, 65:80] = 1

    plume_pixels = int(fake_mask.sum())
    print(f"    Plume detected: {plume_pixels} pixels")
    print(f"    Confidence: {confidence:.2f}")
    print(f"    Location: ({detected_lat:.5f}, {detected_lon:.5f})")

    # ──────────────────────────────────────────
    # Step 2: Entity Resolution
    # ──────────────────────────────────────────
    print("\n[2] Resolving to nearest industrial asset...")

    asset_db = AssetDatabase()

    # Try to load WRI database
    import os
    wri_path = "data/global_power_plant_database.csv"
    if os.path.exists(wri_path):
        asset_db.load_wri_database(wri_path)
        asset = asset_db.find_nearest_asset(detected_lat, detected_lon)
        if asset:
            print(f"    MATCHED: {asset.name}")
            print(f"    Owner: {asset.owner_name}")
            print(f"    Fuel: {asset.fuel_type}")
            print(f"    Distance: {asset.distance_km} km from plume centroid")
        else:
            print("    No asset within 5km tolerance radius")
            asset = None
    else:
        print(f"    [SKIP] WRI database not found at {wri_path}")
        print(f"    Download from: https://datasets.wri.org/dataset/globalpowerplantdatabase")
        asset = None

    # Corporate resolution (Phase 2 stub)
    resolver = CorporateResolver()
    corporate = resolver.resolve(asset.owner_name if asset else None)
    print(f"    Corporate resolution: {corporate.subsidiary_name}")
    if corporate.ticker:
        print(f"    Ticker: {corporate.ticker} ({corporate.exchange})")
    else:
        print("    Ticker: [Phase 2 — OpenCorporates + GLEIF integration]")

    # ──────────────────────────────────────────
    # Step 3: Quantification
    # ──────────────────────────────────────────
    print("\n[3] Quantifying emission flow rate (IME methodology)...")

    ime = IntegratedMassEnhancement(pixel_size_m=20.0)
    quant = ime.estimate(
        plume_mask=fake_mask,
        wind_speed_ms=3.5,  # Would come from ERA5 in production
    )

    print(f"    Flow rate: {quant.flow_rate_kgh:.1f} kg CH4/hour")
    print(f"    Range: [{quant.flow_rate_lower_kgh:.1f}, {quant.flow_rate_upper_kgh:.1f}] kg/h")
    print(f"    Wind speed used: {quant.wind_speed_ms} m/s")

    # ──────────────────────────────────────────
    # Step 4: Financial Impact
    # ──────────────────────────────────────────
    print("\n[4] Calculating financial impact...")

    if quant.annual_tonnes_estimate:
        print(f"    Annualized emissions: {quant.annual_tonnes_estimate:,.0f} tonnes CH4")
        print(f"    IRA Waste Emissions Charge (2026): ${quant.ira_waste_charge_usd:,.0f}")
        print(f"    (At statutory max rate of $1,500/metric ton)")

    # ──────────────────────────────────────────
    # Step 5: Structured JSON Output
    # ──────────────────────────────────────────
    print("\n[5] Generating API-ready JSON response...")

    event = {
        "event_uuid": f"evt-{uuid.uuid4().hex[:12]}",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "latitude": detected_lat,
        "longitude": detected_lon,
        "model_confidence": confidence,
        "plume_area_pixels": plume_pixels,
        "flow_rate_kgh": quant.flow_rate_kgh,
        "flow_rate_lower_kgh": quant.flow_rate_lower_kgh,
        "flow_rate_upper_kgh": quant.flow_rate_upper_kgh,
        "annual_tonnes_estimate": quant.annual_tonnes_estimate,
        "ira_waste_charge_usd": quant.ira_waste_charge_usd,
        "asset_id": asset.asset_id if asset else None,
        "asset_name": asset.name if asset else None,
        "asset_distance_km": asset.distance_km if asset else None,
        "asset_fuel_type": asset.fuel_type if asset else None,
        "owner_name": asset.owner_name if asset else None,
        "corporate_lei": corporate.lei,
        "financial_ticker": corporate.ticker,
        "exchange": corporate.exchange,
    }

    print(json.dumps(event, indent=2))

    print("\n" + "=" * 70)
    print("Pipeline demo complete.")
    print("Next steps:")
    print("  1. Download WRI database CSV to data/ folder")
    print("  2. Copy trained CH4Net weights to weights/best_model.pth")
    print("  3. Run: uvicorn src.api.main:app --reload")
    print("  4. Open http://localhost:8000/docs")
    print("=" * 70)


if __name__ == "__main__":
    main()
