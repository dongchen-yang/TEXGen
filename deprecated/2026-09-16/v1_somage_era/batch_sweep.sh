#!/usr/bin/env bash
# Find the largest per-GPU batch size that fits on a single H100 (80 GB).
# Runs launch.py with limit_train_batches=2 at each batch size; records peak
# GPU memory via nvidia-smi polling.
#
# Usage:  bash batch_sweep.sh
# Pre:    salloc'd onto an H100 node, env activated, $SLURM_TMPDIR/baked_uv populated.

set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT=${DATA_ROOT:-${SLURM_TMPDIR}/baked_uv}
SWEEP_DIR=${SWEEP_DIR:-${SLURM_TMPDIR}/sweep}
mkdir -p "$SWEEP_DIR"
test -d "$DATA_ROOT" || { echo "FATAL: $DATA_ROOT not found"; exit 1; }

echo "data_root: $DATA_ROOT  ($(find "$DATA_ROOT" -name somage.npz | wc -l) NPZs)"
echo "sweep_dir: $SWEEP_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_one() {
    local BS="$1"
    local LOG="$SWEEP_DIR/bs${BS}.log"
    local PEAK="$SWEEP_DIR/bs${BS}.peak"
    : > "$PEAK"

    # background poller: every 1s, record current GPU mem (MB).
    (
        max=0
        while sleep 1; do
            m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
            [ -z "$m" ] && continue
            if [ "$m" -gt "$max" ]; then max="$m"; echo "$max" > "$PEAK"; fi
        done
    ) &
    POLL=$!

    echo "=== bs=$BS launching === ($(date +%H:%M:%S))"
    # tee: live to terminal AND $LOG. PIPESTATUS captures launch.py's exit code.
    timeout 360 stdbuf -oL -eL python -u launch.py \
        --config configs/lightgen_pointuv_256_batch32_emissive_74k.yaml \
        --gpu 0 --train \
        data.data_root="$DATA_ROOT" \
        data.batch_size="$BS" \
        data.num_workers=2 \
        trainer.limit_train_batches=3 \
        trainer.limit_val_batches=0 \
        trainer.limit_test_batches=0 \
        trainer.max_epochs=1 \
        trainer.check_val_every_n_epoch=999 \
        trainer.num_sanity_val_steps=0 \
        exp_root_dir="$SWEEP_DIR/out_bs${BS}" \
        checkpoint.dirpath="$SWEEP_DIR/ckpts_bs${BS}" \
        checkpoint.save_top_k=0 \
        checkpoint.save_last=false \
        2>&1 | tee "$LOG"
    local RC=${PIPESTATUS[0]}
    kill "$POLL" 2>/dev/null
    wait "$POLL" 2>/dev/null

    local PV=$(cat "$PEAK" 2>/dev/null || echo 0)
    if grep -qE 'out of memory|CUDA out of memory|OutOfMemoryError' "$LOG"; then
        echo "  bs=$BS  OOM        peak=${PV} MB"
        return 1
    elif [ "$RC" -ne 0 ]; then
        echo "  bs=$BS  EXIT=$RC   peak=${PV} MB  (see $LOG)"
        return 1
    else
        echo "  bs=$BS  OK         peak=${PV} MB"
    fi
    return 0
}

for BS in 32 48 64 96 128 160; do
    if ! run_one "$BS"; then
        echo "stopping at bs=$BS"
        break
    fi
done

echo "==================="
echo "summary:"
for f in "$SWEEP_DIR"/bs*.peak; do
    bs=$(basename "$f" .peak | sed 's/bs//')
    pv=$(cat "$f" 2>/dev/null || echo "?")
    log="$SWEEP_DIR/bs${bs}.log"
    if grep -qE 'out of memory|CUDA out of memory|OutOfMemoryError' "$log" 2>/dev/null; then
        status=OOM
    elif [ -f "$log" ] && tail -1 "$log" | grep -qiE 'finished|exit code: 0|complete'; then
        status=OK
    else
        # default: if peak > 0 and no OOM, mark OK
        status="?"
    fi
    printf '  bs=%-4s peak=%s MB  %s\n' "$bs" "$pv" "$status"
done
