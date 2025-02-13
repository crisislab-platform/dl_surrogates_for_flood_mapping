#!/bin/bash

# Define base name, index range, and command
base_name="carlisle_run"
start_index=3
end_index=9
command_to_run="/home/91/23016891/software/LISFLOOD-FP/build/lisflood"

# Loop through the index range
for ((i=start_index; i<=end_index; i++)); do
  # Create file name with index suffix
  file_name="${base_name}${i}.par"

  # Run the command with the file name
  echo "Running: $command_to_run $file_name"
  $command_to_run $file_name
done
