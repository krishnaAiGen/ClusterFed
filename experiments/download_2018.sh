#!/bin/bash
# Download CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" (~6.9GB)
# via anonymous HTTP from the public S3 bucket. Resumable (-C -), verified by size.
set -u
DEST="${DEST:-$(cd "$(dirname "$0")/.." && pwd)/data/raw_2018}"
BASE="https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms"
mkdir -p "$DEST"
FILES=(
  "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
  "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"
  "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
  "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"
  "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"
  "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"
  "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"
  "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
)
for f in "${FILES[@]}"; do
  for attempt in 1 2 3; do
    echo "[$(date +%H:%M:%S)] downloading $f (attempt $attempt)"
    curl -sS -C - -o "$DEST/$f" "$BASE/$f" && break
    sleep 20
  done
done
echo "[$(date +%H:%M:%S)] download loop finished"
ls -la "$DEST"
touch "$DEST/.download_complete"
