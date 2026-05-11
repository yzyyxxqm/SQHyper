#!/bin/bash
# Remote ablation queue runner.
# Assumes:
#   * cwd = /opt/codes/QSH
#   * USHCN triplet (no_sri / no_sqc / no_both) already launched & running.
#   * Logs go to /tmp/sqhh_v3_logs/
#
# Schedule (parallel waves):
#   Wave 0 (already in flight): USHCN_{no_sri,no_sqc,no_both}
#   Wave 1 (launch now):        MIMIC_III_{no_sri,no_sqc,no_both}
#   Wave 2 (after Wave 0+1):    P12_{no_sri,no_sqc,no_both}
#   Wave 3 (after Wave 2):      HumanActivity_{no_sri,no_sqc,no_both}   (1 at a time)
#
# Wave-completion check = "no python main.py with that dataset_name in ps".

set -u
mkdir -p /tmp/sqhh_v3_logs
export PATH=/home/user/micromamba/envs/pyomnits/bin:$PATH

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a /tmp/sqhh_v3_logs/_queue.log; }

count_procs() {  # $1 = dataset_name; counts python main.py procs for that dataset
    local ds="$1"
    ps -ef | grep "python main.py" | grep -v grep \
        | grep -- "--dataset_name $ds" | wc -l
}

wait_done() {  # $1 = dataset_name; $2 = label
    local ds="$1"; local label="$2"
    while : ; do
        n=$(count_procs "$ds")
        if [ "$n" -eq 0 ]; then
            log "$label: all $ds procs finished"
            break
        fi
        log "$label: $n $ds procs still running, sleep 60s"
        sleep 60
    done
}

launch_triplet_parallel() {  # $1 = dataset_name; $2 = wave label
    local ds="$1"; local wave="$2"
    for v in no_sri no_sqc no_both; do
        log "$wave: launching ${ds}_${v}"
        nohup bash scripts/SQHH/${ds}_${v}.sh \
            > /tmp/sqhh_v3_logs/SQHH_${ds}_${v}.log 2>&1 &
        sleep 6
    done
}

launch_triplet_serial() {  # $1 = dataset_name; $2 = wave label
    local ds="$1"; local wave="$2"
    for v in no_sri no_sqc no_both; do
        log "$wave: launching ${ds}_${v} (serial)"
        bash scripts/SQHH/${ds}_${v}.sh \
            > /tmp/sqhh_v3_logs/SQHH_${ds}_${v}.log 2>&1
        log "$wave: ${ds}_${v} done"
    done
}

# --- Wave 1: MIMIC triplet (start now in parallel with USHCN wave 0) ---
log "Wave 1: starting MIMIC_III triplet"
launch_triplet_parallel "MIMIC_III" "Wave1"

# --- Wait for USHCN (wave 0) + MIMIC (wave 1) to drain ---
wait_done "USHCN"     "Wave0"
wait_done "MIMIC_III" "Wave1"

# --- Wave 2: P12 triplet ---
log "Wave 2: starting P12 triplet"
launch_triplet_parallel "P12" "Wave2"
wait_done "P12" "Wave2"

# --- Wave 3: HumanActivity triplet (serial; HA is big & slow) ---
log "Wave 3: starting HumanActivity triplet (serial)"
launch_triplet_serial "HumanActivity" "Wave3"

log "ALL WAVES COMPLETE."
