#!/bin/bash

# Define the script and base arguments
SCRIPT="Meilenstein4/meilenstein4/keras_contrastive.py"
BASE_ARGS="train -d /home/pmayer1/Data_Deep-Learning-2024/data/vgg_faces2/train -b 32"

# Define the parameter values for -m
M_VALUES=(0.2 0.4 0.6 0.8 1)

# Loop over each value of -m and execute the script
for M in "${M_VALUES[@]}"; do
    # Format the name based on the current value of -m
    NAME="margin_${M//./_}_contrastive.keras"
    echo "Running with -m $M and -n $NAME"
    CUDA_VISIBLE_DEVICES=0 python3 $SCRIPT $BASE_ARGS -m $M -n $NAME
    if [ $? -ne 0 ]; then
        echo "Execution failed for -m $M"
        exit 1
    fi
done

echo "All executions initiated."
