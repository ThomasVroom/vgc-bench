#!/bin/bash

PORT=7200
DEVICE="cuda:0"
LOGFILE="debug${PORT}.log"

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

echo "Running evaluation..."

# Run python module and save output
python3.13 -u -m vgc_bench.eval \
    --port $PORT \
    --device $DEVICE \
    > "$LOGFILE" 2>&1

EXIT_STATUS=$?

if [ $EXIT_STATUS -ne 0 ]; then
    echo "Eval failed with exit status $EXIT_STATUS"
else
    echo "Eval finished successfully"
fi
