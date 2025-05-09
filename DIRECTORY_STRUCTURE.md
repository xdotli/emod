# EMOD Project Directory Structure

The EMOD codebase has been refactored with a cleaner, more organized directory structure:

## Core Directories

- `src/`: Core implementation files
  - `src/core/`: Core model implementation (VAD predictor, emotion classifier)
  - `src/processing/`: Data and results processing utilities
  - `src/utils/`: General utilities and logging
  - `src/modal/`: Modal integration framework

- `tests/`: All unit tests
  - Contains comprehensive tests for all components

- `experiments/`: Modal experiment files
  - Contains different Modal experiment configurations
  - Managed through the centralized `emod_cli.py` interface

- `scripts/`: Utility scripts
  - Helper scripts for downloading results, generating reports, etc.

- `notebooks/`: Jupyter notebooks
  - Analysis and exploration notebooks

## Data & Results Directories

- `results/`: Experiment results
  - Downloaded from Modal volumes
  - Processed for comparison

- `reports/`: Generated reports
  - HTML and Markdown summaries
  - Performance visualizations

## Key Files

- `emod_cli.py`: Main command-line interface
  - Central entry point for experiments, results processing, and reporting

- `run_tests.py`: Test runner script
  - Discovers and runs all tests

- `requirements.txt`: Python dependencies

## Running the Project

```
# Running experiments
python emod_cli.py experiment --text-models roberta-base --epochs 10

# Processing results
python emod_cli.py results

# Generating reports
python emod_cli.py report --format html

# Running tests
./run_tests.py
```

This refactored structure provides better separation of concerns, easier maintenance, and improved test coverage. 