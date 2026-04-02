#!/bin/bash

run_ids=(1)
ports=(7200)
devices=("cuda:0")

start_showdown() {
    local port=$1
    (
        cd pokemon-showdown
        node pokemon-showdown start $port --no-security > /dev/null 2>&1 &
        echo $!
    )
}

eval() {
    local i=$1
    local run_id="${run_ids[$i]}"
    local port="${ports[$i]}"
    local device="${devices[$i]}"

    echo "Starting Showdown server for eval process $i..."
    showdown_pid=$(start_showdown $port)
    sleep 5
    echo "Starting eval process $i..."
    python3.13 -m vgc_bench.eval \
        --port $port \
        --device $device \
        > "debug$port.log" 2>&1
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Eval process $i died with exit status $exit_status"
    else
        echo "Eval process $i finished!"
    fi
    kill $showdown_pid
}

trap "echo 'Stopping...'; kill 0" SIGINT
for i in "${!run_ids[@]}"; do
    eval $i &
    sleep 30
done
wait
