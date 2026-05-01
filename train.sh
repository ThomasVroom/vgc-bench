#!/bin/bash

run_ids=(2 2 2 2 2 2 2 2 2 2)
team_counts=(64 64 64 64 64 64 64 64 64 64)
ports=(7200 7200 7200 7200 7200 7200 7200 7200 7200 7200)
devices=("cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0")
regs=("A" "B" "C" "D" "E" "F" "G" "H" "I" "J")

num_envs=16
total_steps=$((51 * 98304))  # 98_304 is the number of steps per save during training

start_showdown() {
    local port=$1
    (
        cd pokemon-showdown
        node pokemon-showdown start $port --no-security > /dev/null 2>&1 &
        echo $!
    )
}

train() {
    local i=$1
    local run_id="${run_ids[$i]}"
    local num_teams="${team_counts[$i]}"
    local port="${ports[$i]}"
    local device="${devices[$i]}"
    local reg="${regs[$i]}"

    echo "Starting Showdown server for training process $i..."
    showdown_pid=$(start_showdown $port)
    sleep 5
    echo "Starting training process $i..."
    python3.13 -m vgc_bench.train \
        --run_id $run_id \
        --num_teams $num_teams \
        --num_envs $num_envs \
        --num_eval_workers $num_envs \
        --port $port \
        --device $device \
        --self_play \
        --reg $reg \
        --total_steps "$total_steps" \
        > "debug$port.log" 2>&1
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Training process $i died with exit status $exit_status"
    else
        echo "Training process $i finished!"
    fi
    kill $showdown_pid
}

trap "echo 'Stopping...'; kill 0" SIGINT
for i in "${!run_ids[@]}"; do
    train $i
    sleep 30
done
wait
