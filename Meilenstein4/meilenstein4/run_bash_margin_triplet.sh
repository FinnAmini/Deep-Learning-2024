#!/bin/bash

# Define the script and base arguments
SCRIPT="Meilenstein4/meilenstein4/keras_triplet.py"
BASE_ARGS="train -d /home/pmayer1/Data_Deep-Learning-2024/data/vgg_faces2/train -b 16"

# Define the parameter values for -m
# M_VALUES=(0.1 0.05 0.025)
LR_VALUES=(0.001 0.0001 0.00001)

# Loop over each value of -m and execute the script
for M in "${LR_VALUES[@]}"; do
    # Format the name based on the current value of -m
    NAME="learning_rate_${M//./_}triplet.keras"
    echo "Running with -lr $M and -n $NAME"
    CUDA_VISIBLE_DEVICES=2 python3 $SCRIPT $BASE_ARGS -lr $M -n $NAME
    if [ $? -ne 0 ]; then
        echo "Execution failed for -lr $M"
        exit 1
    fi
done

echo "All executions completed."