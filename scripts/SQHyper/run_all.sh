#!/bin/bash
# Run SQHyper on all datasets sequentially
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

echo "=== SQHyper: Starting all datasets ==="

for dataset in USHCN P12 HumanActivity MIMIC_III; do
    echo ""
    echo ">>> Starting $dataset at $(date)"
    bash "$SCRIPT_DIR/$dataset.sh" 2>&1 | tee "$SCRIPT_DIR/../../storage/logs/SQHyper_${dataset}_$(date +%m%d_%H%M).log"
    echo ">>> Finished $dataset at $(date)"
done

echo ""
echo "=== SQHyper: All datasets complete ==="
