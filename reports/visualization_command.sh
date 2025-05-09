#!/bin/bash
# Run this command when both experiments are complete
python visualize_results.py --experiments results/final_dataset_full results/filtered_dataset_full --labels "Original Dataset" "Filtered Dataset" --output-dir reports
