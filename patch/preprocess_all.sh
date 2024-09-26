#!/bin/bash

for CONFIG in "${CONFIGURATIONS[@]}"; do
    echo "Processing configuration ./preprocess/train.json..."
    python preprocess/preprocess.py \
        -c ./preprocess/train.json \
        -p training 
done