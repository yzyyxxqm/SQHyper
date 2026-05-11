#!/bin/bash
# Sequentially run all 4 SQHH datasets at --itr 1 for fast iteration.
# Logs each into /tmp/sqhh_v3_logs/SQHH_<dataset>_1itr.log.
set -e
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOGDIR=/tmp/sqhh_v3_logs
mkdir -p "$LOGDIR"

for ds in USHCN P12 MIMIC_III HumanActivity; do
    echo "[$(date '+%H:%M:%S')] >>> $ds start"
    bash "$SCRIPT_DIR/$ds.sh" > "$LOGDIR/SQHH_${ds}_1itr.log" 2>&1
    rc=$?
    echo "[$(date '+%H:%M:%S')] <<< $ds done (rc=$rc)"
done
echo "[$(date '+%H:%M:%S')] ALL DONE"
