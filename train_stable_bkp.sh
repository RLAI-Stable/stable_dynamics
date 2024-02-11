#!/bin/bash
set -e

EXP_ID="$1"
N="$2"
ALPHA="$3"
INNER="$4"
INNER_EPSILON="$5"
SMOOTH_V="$6"
REHU="$7"

# Latent space dim:
LSD=$((2 * N))

OUTDIR="experiments/pendulum-${EXP_ID}/${N}_${ALPHA}_${INNER}_${INNER_EPSILON}_${SMOOTH_V}_${REHU}"
MODEL="stabledynamics[latent_space_dim=$LSD,a=$ALPHA,projfn=$INNER,projfn_eps=$INNER_EPSILON,smooth_v=$SMOOTH_V,hp=60,h=100,rehu=$REHU]"

mkdir -p "$OUTDIR"
echo $MODEL >"$OUTDIR/model"

date >>"$OUTDIR/progress.txt"
./.colorize ./train.py \
    --log-to "runs/$OUTDIR" \
    --batch-size 2000 \
    --learning-rate "0.001" \
    --epochs 1000 \
    --test-with "pendulum[n=$N,test]" \
    "pendulum[n=$N]" \
    "$MODEL" \
    "$OUTDIR/checkpoint_{epoch:0>5}.pth" | tee -a "$OUTDIR/progress.txt"

./.colorize ./pendulum_error.py \
    pendulum[n=$N,test] \
    "$MODEL" \
    "$OUTDIR/checkpoint_*.pth" \
    1000 | tee -a "$OUTDIR/eval.txt"
