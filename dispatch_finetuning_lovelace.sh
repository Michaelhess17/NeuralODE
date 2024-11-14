#!/bin/bash

# Number of GPUs available
NUM_GPUS=4

# Number of parallel jobs per GPU
JOBS_PER_GPU=3

# Calculate total concurrent jobs
N=$((NUM_GPUS * JOBS_PER_GPU))

# Create a list of subject IDs (0 to 19)
subjects=($(seq 5 14))

# Initialize arrays to track processes per GPU
declare -A processes_per_gpu
for ((i=0; i<NUM_GPUS; i++)); do
    processes_per_gpu[$i]=0
done

# Function to find GPU with least processes
find_available_gpu() {
    local min_processes=${processes_per_gpu[0]}
    local selected_gpu=0
    
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        if ((${processes_per_gpu[$gpu]} < min_processes)); then
            min_processes=${processes_per_gpu[$gpu]}
            selected_gpu=$gpu
        fi
    done
    echo $selected_gpu
}

# Function to run jobs in parallel
run_parallel() {
    # Track running processes
    running=0
    
    # Loop through all subjects
    for subject in "${subjects[@]}"; do
        # If we've reached max processes, wait for one to finish
        if (( running >= N )); then
            wait -n
            # Decrease process count for the GPU that finished
            for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
                if ((${processes_per_gpu[$gpu]} > 0)); then
                    processes_per_gpu[$gpu]=$((processes_per_gpu[$gpu] - 1))
                    break
                fi
            done
            running=$((running - 1))
        fi
        
        # Find GPU with least processes
        gpu_id=$(find_available_gpu)
        
        # Start the process in background with assigned GPU
        CUDA_VISIBLE_DEVICES=$gpu_id python finetune_models.00.py --subject $subject &
        
        # Update counters
        processes_per_gpu[$gpu_id]=$((processes_per_gpu[$gpu_id] + 1))
        running=$((running + 1))
        
        echo "Started subject $subject on GPU $gpu_id (GPU $gpu_id has ${processes_per_gpu[$gpu_id]} processes)"
    done
    
    # Wait for remaining processes to finish
    wait
}

# Run the parallel processing function
run_parallel