#!/bin/bash

# Define base name, index range, and command
base_name="carlisle_run"
start_index=1
end_index=9
command_to_run="/home/91/23016891/software/LISFLOOD-FP/build/lisflood"
log_file="simulation.log"

# Check if LISFLOOD-FP exists
if [ ! -f "$command_to_run" ]; then
    echo "Error: LISFLOOD-FP not found at $command_to_run"
    exit 1
fi

echo "Starting simulations at $(date)" | tee -a "$log_file"

# Loop through the index range
for ((i=start_index; i<=end_index; i++)); do
    file_name="${base_name}${i}.par"
    
    # Check if parameter file exists
    if [ ! -f "$file_name" ]; then
        echo "Error: Parameter file $file_name not found" | tee -a "$log_file"
        continue
    }

    # Run the command with the file name
    echo "Running simulation $i of $end_index: $file_name" | tee -a "$log_file"
    if $command_to_run $file_name >> "$log_file" 2>&1; then
        echo "Successfully completed simulation $i" | tee -a "$log_file"
    else
        echo "Error in simulation $i" | tee -a "$log_file"
    fi
done

echo "Completed all simulations at $(date)" | tee -a "$log_file"
