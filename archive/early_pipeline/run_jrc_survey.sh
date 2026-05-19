#!/bin/bash
# JRC European Power Plant Survey
# Runs live_pipeline on all sites sequentially, logs each to results_survey/<site>/run.log
# Usage: bash run_jrc_survey.sh
# Runs overnight — safe to leave unattended.

set -e
cd "$(dirname "$0")"

WEIGHTS="weights/ch4net_div8_retrained.pth"
START="2024-06-01"
END="2024-08-31"
LOG="results_survey/survey_run.log"

mkdir -p results_survey
echo "JRC Survey started at $(date)" | tee "$LOG"

run_site() {
    local NAME=$1
    local POLYGON=$2
    local OUT="results_survey/$NAME"
    mkdir -p "$OUT"

    # Skip if already completed (GeoTIFF exists)
    if ls "$OUT"/*.tif 2>/dev/null | grep -q .; then
        echo "[SKIP] $NAME — GeoTIFF already exists" | tee -a "$LOG"
        return
    fi

    echo "" | tee -a "$LOG"
    echo "[$NAME] Starting at $(date)" | tee -a "$LOG"

    python -m scripts.live_pipeline \
        --region "$POLYGON" \
        --start "$START" \
        --end "$END" \
        --weights "$WEIGHTS" \
        --max-products 1 \
        --max-cloud 20 \
        --threshold 0.18 \
        --output "$OUT" \
        2>&1 | tee -a "$OUT/run.log" | tail -5

    echo "[$NAME] Done at $(date)" | tee -a "$LOG"
}

# ── Positive control (your known 2.95× result) ────────────────────────────────
run_site "groningen"     "POLYGON((6.48 53.05, 6.88 53.05, 6.88 53.45, 6.48 53.45, 6.48 53.05))"

# ── Explicitly flagged as not yet tested in experimental doc ──────────────────
run_site "irsching"      "POLYGON((11.38 48.60, 11.78 48.60, 11.78 48.95, 11.38 48.95, 11.38 48.60))"
run_site "belchatow"     "POLYGON((19.13 51.06, 19.53 51.06, 19.53 51.46, 19.13 51.46, 19.13 51.06))"

# ── New diverse terrain sites ─────────────────────────────────────────────────
run_site "maasvlakte"    "POLYGON((3.80 51.75, 4.20 51.75, 4.20 52.15, 3.80 52.15, 3.80 51.75))"
run_site "philippsburg"  "POLYGON((8.26 49.05, 8.66 49.05, 8.66 49.45, 8.26 49.45, 8.26 49.05))"
run_site "cordemais"     "POLYGON((-2.07 47.08, -1.67 47.08, -1.67 47.48, -2.07 47.48, -2.07 47.08))"
run_site "brindisi"      "POLYGON((17.65 40.42, 18.05 40.42, 18.05 40.82, 17.65 40.82, 17.65 40.42))"
run_site "avedore"       "POLYGON((12.25 55.40, 12.65 55.40, 12.65 55.80, 12.25 55.80, 12.25 55.40))"
run_site "pocerady"      "POLYGON((13.34 50.19, 13.74 50.19, 13.74 50.59, 13.34 50.59, 13.34 50.19))"
run_site "pembroke"      "POLYGON((-5.19 51.48, -4.79 51.48, -4.79 51.88, -5.19 51.88, -5.19 51.48))"
run_site "dunkerque"     "POLYGON((2.16 50.85, 2.56 50.85, 2.56 51.25, 2.16 51.25, 2.16 50.85))"
run_site "larobla"       "POLYGON((-5.81 42.59, -5.41 42.59, -5.41 42.99, -5.81 42.99, -5.81 42.59))"

echo "" | tee -a "$LOG"
echo "All sites complete at $(date)" | tee -a "$LOG"
echo "Results in results_survey/  |  Full log: $LOG"
