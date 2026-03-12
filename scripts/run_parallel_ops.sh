#!/bin/bash
SEED=2333

DOMAIN="KnownBits"
NUM_ITERS=5
NUM_STEPS=1000
NUM_MCMC=100
MBW="8,5000 16,5000"
HBW="32,5000,10000 64,5000,10000"
VBW="4,8,16,32,64"
MUTATION_FLAGS="replace_entire_op,replace_operand"

COMMON_ARGS="--domain $DOMAIN --num-iters $NUM_ITERS --num-steps $NUM_STEPS --num-mcmc $NUM_MCMC --mbw $MBW --hbw $HBW --vbw $VBW --random-seed $SEED  --mutation-flags $MUTATION_FLAGS"

OUTDIR="outputs/parallel_runs_baseline"

OPERATIONS=(
    "mlir/Operations/And.mlir"
    "mlir/Operations/Add.mlir"
    "mlir/Operations/Mul.mlir"
    "mlir/Operations/Udiv.mlir"
    "mlir/Operations/Modu.mlir"
    "mlir/Operations/Umax.mlir"
    "mlir/Operations/Shl.mlir"
    "mlir/Operations/Lshr.mlir"
    "mlir/Operations/UdivExact.mlir"
    "mlir/Operations/AddNuw.mlir"
)

CORES=(2 5 9 25 36 44 51 57 70 71)

mkdir -p "$OUTDIR"
touch "$OUTDIR/eval.runconf"
cat <<EOL > "$OUTDIR/eval.runconf"
seed: $SEED
num_iters: 2
num_steps: 100
num_mcmc: 50
mutation_flags: $MUTATION_FLAGS
EOL

pids=()

for i in "${!OPERATIONS[@]}"; do
    OP="${OPERATIONS[$i]}"
    OP_NAME=$(basename "$OP" .mlir)
    RUN_OUT="$OUTDIR/$OP_NAME"
	CORE="${CORES[$i]}"
    mkdir -p "$RUN_OUT"

    taskset -c "$CORE" \
        sxf "$OP" $COMMON_ARGS --output "$RUN_OUT" \
        > "$RUN_OUT/stdout.log" 2> "$RUN_OUT/stderr.log" &

    pids+=($!)
    echo "Started $OP_NAME on core $CORE (pid ${pids[-1]})"
done

echo ""
echo "Waiting for all runs to finish..."

all_ok=true
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Run $i finished OK"
    else
        echo "Run $i FAILED (exit code $?)"
        all_ok=false
    fi
done

echo ""
if $all_ok; then
    echo "All runs completed successfully."
else
    echo "Some runs failed — check $OUTDIR/run_*/stderr.log"
fi
