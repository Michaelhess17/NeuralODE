SOURCE_FILE="/home/lovelace/Synology/Python/NeuralODE/figures/trained_model_generation.png"
DESTINATION_PREFIX="/home/lovelace/Synology/Python/NeuralODE/figures/train_model_generation_"
COUNTER=1

while true; do
    # Check if file exists
    if [ -f "$SOURCE_FILE" ]; then
        # Move file with unique name
        mv "$SOURCE_FILE" "${DESTINATION_PREFIX}${COUNTER}.png"
        echo "Moved file to ${DESTINATION_PREFIX}${COUNTER}.png"
        COUNTER=$((COUNTER + 1))
    fi
    
    # Wait a bit before checking again
    sleep 1
done