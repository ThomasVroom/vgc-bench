#!/bin/bash

run_ids=(1 1 1 1 1 1 1 1)
team_counts=(64 64 64 64 64 64 64 64)
ports=(7200 7200 7200 7200 7200 7200 7200 7200)
devices=("cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0" "cuda:0")
regs_source=("a_to_b" "a_to_b_to_c" "a_to_b_to_c_to_d" "a_to_b_to_c_to_d_to_e" "a_to_b_to_c_to_d_to_e_to_f" "a_to_b_to_c_to_d_to_e_to_f_to_g" "a_to_b_to_c_to_d_to_e_to_f_to_g_to_h" "a_to_b_to_c_to_d_to_e_to_f_to_g_to_h_to_i")
regs_target=("c" "d" "e" "f" "g" "h" "i" "j")

num_envs=16

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
    local total_steps=$(((3 + $i) * 51 * 98304))

    echo "Starting Showdown server for fine-tune process $i..."
    showdown_pid=$(start_showdown $port)
    sleep 5
    echo "Starting fine-tune process $i..."
    python3.13 -u -m vgc_bench.finetune \
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
