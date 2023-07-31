#!/usr/bin/env sh


model_path=$1

time_stamp=$(date +"%Y-%m-%d-%H:%M:%S")
for file in $(ls ${model_path}); do
    if [[ $file == *.csv ]]; then
        # strip .csv
        file_name="${file%.csv}"
        log_path=$HETERO_SCHEDULE_HOME/log/
        cmd=$"python $HETERO_SCHEDULE_HOME/scripts/solve_and_run_network.py --model="${model_path}/${file}" --chip bst --dump=$HETERO_SCHEDULE_HOME/results/bst/$file_name-${time_stamp} | tee ${log_path}/${file_name}-${time_stamp}.log 2>&1"
        echo $cmd >> "${log_path}/run-bst-${time_stamp}.log"
        eval $cmd
    fi
done
