#!/bin/bash

run_id=1
team_count=64
PORT=7200
device="cuda:0"
LOGFILE="debug${PORT}.log"

source_reg="A"
target_reg="B"
num_envs=16
total_steps=$((2 * 51 * 98304))

echo "Starting Pokémon Showdown on port $PORT..."

# Start Showdown in background
cd pokemon-showdown
node pokemon-showdown start $PORT --no-security > /dev/null 2>&1 &
SHOWDOWN_PID=$!
cd ..

# Ensure Showdown is killed when script exits (success or failure)
cleanup() {
    echo "Stopping Pokémon Showdown (PID $SHOWDOWN_PID)..."
    kill $SHOWDOWN_PID 2>/dev/null
}
trap cleanup EXIT

# Give server time to start
sleep 10

echo "Running fine-tune..."

# Run python module and save output
python3.13 -u -m vgc_bench.finetune \
    --run_id $run_id \
    --num_teams $team_count \
    --num_envs $num_envs \
    --num_eval_workers $num_envs \
    --port $PORT \
    --device $device \
    --self_play \
    --reg_source $source_reg \
    --reg_target $target_reg \
    --total_steps "$total_steps" \
    > "$LOGFILE" 2>&1

EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Fine-tune failed with exit status $EXIT_STATUS"
else
    echo "Fine-tune finished successfully"
fi
