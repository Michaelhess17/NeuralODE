#!/bin/bash
# testing
# Number of GPUs available
NUM_GPUS=1

# Number of parallel jobs per GPU
JOBS_PER_GPU=4

# Calculate total concurrent jobs
N=$((NUM_GPUS * JOBS_PER_GPU))

# Create a list of subject IDs (0 to 19)
subjects=($(seq 0 29))

# Function to run jobs in parallel
run_parallel() {
    # Track running processes
    running=0
    
    # Loop through all subjects
    for subject in "${subjects[@]}"; do
        # If we've reached max processes, wait for one to finish
        if (( running >= N )); then
            wait -n
            running=$((running - 1))
        fi
        
        # Calculate which GPU to use (cycles through available GPUs)
        gpu_id=$((running % NUM_GPUS))
        
        # Start the process in background with assigned GPU
        CUDA_VISIBLE_DEVICES=$gpu_id python finetune_models.00.py --subject $subject &
        
        running=$((running + 1))
        echo "Started subject $subject on GPU $gpu_id (Running processes: $running)"
    done
    
    # Wait for remaining processes to finish
    wait
}

# Run the parallel processing function
run_parallel