#!/bin/bash

# Define base name, index range, and command
base_name="carlisle_run"
start_index=1
end_index=1
command_to_run="/home/91/23016891/software/LISFLOOD-FP/build/lisflood"
log_file="simulation.log"
# Path to parameter files (add this line to specify where parameter files are located)
param_dir="/home/91/23016891/projects/carlisle/data/carlisle"

# Check if LISFLOOD-FP exists
if [ ! -f "$command_to_run" ]; then
    echo "Error: LISFLOOD-FP not found at $command_to_run"
    exit 1
fi

echo "Starting simulations at $(date)" | tee -a "$log_file"

# Loop through the index range
for ((i=start_index; i<=end_index; i++)); do
    file_name="${param_dir}/${base_name}${i}.par"
    
    # Check if parameter file exists
    if [ ! -f "$file_name" ]; then
        echo "Error: Parameter file $file_name not found" | tee -a "$log_file"
        continue
    fi  # Fixed syntax error: changed } to fi

    # Start timing
    start_time=$(date +%s)
    
    # Run the command with the file name
    echo "Running simulation $i of $end_index: $file_name" | tee -a "$log_file"
    
    if $command_to_run $file_name >> "$log_file" 2>&1; then
        # Calculate elapsed time
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        hours=$((elapsed / 3600))
        minutes=$(( (elapsed % 3600) / 60 ))
        seconds=$((elapsed % 60))
        
        echo "Successfully completed simulation $i in ${hours}h ${minutes}m ${seconds}s" | tee -a "$log_file"
    else
        # Calculate elapsed time even for failed runs
        end_time=$(date +%s)
        elapsed=$((end_time - start_time))
        hours=$((elapsed / 3600))
        minutes=$(( (elapsed % 3600) / 60 ))
        seconds=$((elapsed % 60))
        
        echo "Error in simulation $i after ${hours}h ${minutes}m ${seconds}s" | tee -a "$log_file"
    fi
done

echo "Completed all simulations at $(date)" | tee -a "$log_file"
