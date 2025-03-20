#!/bin/bash

# Activate conda environment
conda activate dsc

# Loop over x from 1 to 10
for x in {1..10}; do
    # Loop over d values 1 to 4
    python disruptsc/main.py ECA --duration 1
    for d in 2 3 4; do
        # Run Python script with arguments
        python disruptsc/main.py ECA --cache same_logistic_routes --duration $d
    done
done