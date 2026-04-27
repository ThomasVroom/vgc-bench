#!/bin/bash

run_ids=(1 1 1 1 1)
team_counts=(64 64 64 64 64)
ports=(7200 7201 7202 7203 7204)
devices=("cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0")
regs_source=("A" "A" "A" "C" "F")
regs_target=("C" "F" "I" "F" "I")

num_envs=16
total_steps=$((2 * 51 * 98304))  # 98_304 is the number of steps per save during training

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
    local reg_source="${regs_source[$i]}"
    local reg_target="${regs_target[$i]}"

    echo "Starting Showdown server for fine-tune process $i..."
    showdown_pid=$(start_showdown $port)
    sleep 5
    echo "Starting fine-tune process $i..."
    python3.13 -m vgc_bench.finetune \
        --run_id $run_id \
        --num_teams $num_teams \
        --num_envs $num_envs \
        --num_eval_workers $num_envs \
        --port $port \
        --device $device \
        --self_play \
        --reg_source $reg_source \
        --reg_target $reg_target \
        --total_steps "$total_steps" \
        > "debug$port.log" 2>&1
    exit_status=$?
    if [ $exit_status -ne 0 ]; then
        echo "Fine-tune process $i died with exit status $exit_status"
    else
        echo "Fine-tune process $i finished!"
    fi
    kill $showdown_pid
}

trap "echo 'Stopping...'; kill 0" SIGINT
for i in "${!run_ids[@]}"; do
    train $i
    sleep 30
done
wait
