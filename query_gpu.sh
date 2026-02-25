#!/bin/bash

job_id=$1

output_file="${job_id}_gpu_logs.txt"
touch $output_file

running=true

trap 'running=false' SIGTERM

while $running; do
	nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu --format=csv >> "$output_file" 
	sleep 10
done

