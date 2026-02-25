#!/bin/bash
condition=

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,temperature.gpu --format=csv >> gpu_logs.txt
