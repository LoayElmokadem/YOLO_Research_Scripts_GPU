#!/bin/bash

# List of YOLO models

# models=("8n" "8s" "8m" "8l" "8x")
# models=("9t" "9s" "9m" "9c" "9e")
# models=("10n" "10s" "10m" "10l" "10x")
# models=("11n" "11s" "11m" "11l" "11x")
models=("12n" "12s" "12m" "12l" "12x")



# Path to logs directory
log_dir="logs"
mkdir -p "$log_dir"

# Scripts to execute
scripts=("load_time.py" "disk5.py" "ram_cpu.py" "validation.py" "iou.py")

for model_size in "${models[@]}"; do
    output_file="$log_dir/${model_size}.log"
    echo "Processing YOLO model: $model_size" | tee -a "$output_file"

    # Record start temperature
    start_temp=$(tegrastats | head -n 1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^GPU@[0-9.]+C$/) {gsub("GPU@","",$i); gsub("C","",$i); print $i}}')
    echo "Start Temperature: ${start_temp}°C" | tee -a "$output_file"

    for script in "${scripts[@]}"; do
        echo "Running $script for YOLO model $model_size..." | tee -a "$output_file"
        YOLO_MODEL_SIZE="$model_size" python3 "$script" >> "$output_file" 2>&1
        if [ $? -eq 0 ]; then
            echo "$script executed successfully." | tee -a "$output_file"
        else
            echo "Error occurred while running $script." | tee -a "$output_file"
        fi
        echo "----------------------------------------" >> "$output_file"
    done

    # Record end temperature
    end_temp=$(tegrastats | head -n 1 | awk '{for(i=1;i<=NF;i++) if($i ~ /^GPU@[0-9.]+C$/) {gsub("GPU@","",$i); gsub("C","",$i); print $i}}')
    echo "End Temperature: ${end_temp}°C" | tee -a "$output_file"

    # Calculate and log temperature difference
    temp_diff=$(echo "$end_temp - $start_temp" | bc)
    echo "Temperature Difference: ${temp_diff}°C" | tee -a "$output_file"

    # Run clear_cache.sh and save its output in the log
    echo "Clearing cache..." | tee -a "$output_file"
    bash clear_cache.sh >> "$output_file" 2>&1
    if [ $? -eq 0 ]; then
        echo "Cache cleared successfully." | tee -a "$output_file"
    else
        echo "Error occurred while clearing cache." | tee -a "$output_file"
    fi

    echo "All scripts for YOLO model $model_size executed. Results are saved in $output_file."
    echo "=========================================================================" >> "$output_file"
done

echo "All YOLO models processed. Check the logs in $log_dir."
