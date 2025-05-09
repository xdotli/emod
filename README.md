# EMOD: Emotion Detection System

EMOD is a two-stage emotion recognition system that predicts emotions from text and audio data.

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/emod.git
cd emod

# Install dependencies
pip install -r requirements.txt

# For distributed training (optional)
pip install modal
modal setup
```

### Quick Usage

```bash
# Run a text-only experiment
python emod_cli.py experiment --text-models roberta-base --epochs 10

# Run a multimodal experiment
python emod_cli.py experiment --multimodal --text-models roberta-base --audio-features mfcc --fusion-types early

# Process results
python emod_cli.py results

# Generate a report
python emod_cli.py report --format html
```

## System Overview

EMOD implements a two-stage approach to emotion recognition:

1. **Stage 1**: VAD (Valence-Arousal-Dominance) prediction
   - Transforms input features into continuous emotional dimensions
   - Uses transfer learning with pretrained language models

2. **Stage 2**: Emotion Classification
   - Maps VAD predictions to discrete emotion categories
   - Uses ensemble of traditional ML classifiers

### Variants

- **Text-only**: Uses only text transcripts for emotion recognition
- **Multimodal**: Combines text and audio features with various fusion strategies

## Command Line Interface

All operations are accessible through the unified CLI:

```bash
python emod_cli.py [command] [options]
```

### Running Experiments

```bash
python emod_cli.py experiment [options]
```

Main options:
- `--text-models`: Comma-separated list of models (e.g., "roberta-base,bert-base")
- `--multimodal`: Flag to run multimodal experiments
- `--audio-features`: Audio feature types (for multimodal)
- `--fusion-types`: Fusion strategies (for multimodal)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--dry-run`: Print commands without executing

### Processing Results

```bash
python emod_cli.py results [options]
```

Main options:
- `--target-dir`: Directory containing results
- `--skip-download`: Skip downloading results from Modal
- `--skip-report`: Skip report generation

### Generating Reports

```bash
python emod_cli.py report [options]
```

Main options:
- `--format`: Output format ("html" or "markdown")
- `--target-dir`: Directory containing processed results

## Experiments

### Text-only Model

```bash
python emod_cli.py experiment --text-models roberta-base --epochs 10
```

### Multimodal Model

```bash
python emod_cli.py experiment --multimodal --text-models roberta-base --audio-features mfcc --fusion-types early
```

### Multiple Experiments (Grid Search)

```bash
python emod_cli.py experiment --text-models "roberta-base,bert-base" --epochs "10,20"
```

## Extending the System

### Adding a New Text Model

Simply use any model available in Hugging Face Transformers:

```bash
python emod_cli.py experiment --text-models your-new-model
```

### Adding New Audio Features

1. Add extraction function in `src/core/emod_multimodal.py`
2. Run with the new feature:

```bash
python emod_cli.py experiment --multimodal --audio-features your-new-feature
```

### Creating a New Fusion Strategy

1. Add fusion function in `src/core/emod_multimodal.py`
2. Run with the new strategy:

```bash
python emod_cli.py experiment --multimodal --fusion-types your-new-strategy
```

## Results and Reports

After running experiments:

```bash
# Process all results
python emod_cli.py results

# Generate a comprehensive report
python emod_cli.py report --format html
```

## Directory Structure

```
emod/
├── src/
│   ├── core/            # Core model implementations
│   │   ├── common.py    # Shared utility functions
│   │   ├── emod.py      # Text-only pipeline
│   │   └── emod_multimodal.py  # Multimodal pipeline
│   ├── modal/           # Modal integration
│   ├── processing/      # Results processing
│   └── utils/           # Utility functions
├── experiments/         # Experiment configurations
├── scripts/             # Utility scripts
├── results/             # Experiment results
├── reports/             # Generated reports
├── emod_cli.py          # Command-line interface
└── requirements.txt     # Project dependencies
```

## Key Concepts

- **VAD Dimensions**: Continuous representation of emotion (Valence, Arousal, Dominance)
- **Fusion Strategies**: Methods for combining text and audio features
  - Early fusion: Combines features before model processing
  - Late fusion: Combines predictions after separate processing
  - Hybrid fusion: Combines at multiple levels

## Troubleshooting

For common issues:

1. **Out of Memory Errors**: Reduce batch size (`--batch-size 8`)
2. **Modal Issues**: Run `modal token new` to re-authenticate
3. **Performance Issues**: Increase epochs or try different models
