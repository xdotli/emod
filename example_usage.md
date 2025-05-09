# EMOD Experiment Runner Usage Examples

The `run_all_experiments.py` script now supports both running experiments and downloading results. Here are some example use cases:

## Running Experiments

To just run all experiments without downloading results:

```bash
# Run all experiments (default behavior)
python run_all_experiments.py

# Explicitly specify run only
python run_all_experiments.py --run
```

## Downloading Results

To only download results from previously run experiments:

```bash
# Download all results to default directory (./emod_results)
python run_all_experiments.py --download

# Download to a specific directory
python run_all_experiments.py --download --output-dir ./my_results_folder
```

## Run and Download in One Command

To run experiments and then download results:

```bash
# Run experiments and immediately download results
python run_all_experiments.py --run --download

# Run experiments, wait 60 minutes, then download results
python run_all_experiments.py --run --download --wait 60
```

## Using the Results

After downloading, the results will be organized as follows:

```
emod_results/
├── experiment_summary.json           # Summary of all experiments
├── IEMOCAP_Final_text_roberta_base_TIMESTAMP/
│   ├── logs/
│   │   ├── training_log.json         # Detailed training logs
│   │   └── final_results.json        # Final metrics and results
│   └── checkpoints/
│       └── best_model.pt             # Best model checkpoint
├── IEMOCAP_Final_text_distilbert_base_uncased_TIMESTAMP/
│   ├── ...
...
```

The `experiment_summary.json` file contains a summary of all experiments, including model type, dataset, and performance metrics.

## Tips

- For large experiments, use the `--wait` parameter to allow experiments to complete before downloading
- You can run the download command multiple times to get updated results as experiments complete
- Use the downloaded results for analysis or to continue training from checkpoints 