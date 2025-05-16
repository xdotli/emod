# EMOD: Emotion Detection System

EMOD is a two-stage emotion recognition system that predicts emotions from text and audio data using continuous VAD (Valence-Arousal-Dominance) dimensions.

## System Overview

EMOD implements a two-stage approach to emotion recognition:

1. **Stage 1**: VAD (Valence-Arousal-Dominance) prediction
   - Transforms input features into continuous emotional dimensions
   - Uses transfer learning with pretrained language models for text
   - Leverages audio features for multimodal analysis

2. **Stage 2**: Emotion Classification
   - Maps VAD predictions to discrete emotion categories
   - Uses ensemble of traditional ML classifiers

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/emod.git
cd emod

# Install dependencies
pip install -r requirements.txt

# For distributed training
pip install modal
modal setup
```

## Datasets

The system uses the IEMOCAP dataset in various formats:

- **IEMOCAP_Final.csv**: The complete processed dataset
- **IEMOCAP_Filtered.csv**: A filtered version removing short utterances and outliers

Dataset filtering removed 1,650 samples (16.44%) from the original dataset, mostly short utterances like "Yeah." and "No." to improve model performance.

## Running Experiments

Experiments can be run using several scripts:

```bash
# Run a single experiment
python run_single_experiment.py --dataset IEMOCAP_Final --model roberta-base --epochs 40

# Run full experiments on both datasets
python run_full_experiments.py --models "roberta-base,distilbert-base-uncased" --epochs 40
```

### Modal Integration

Experiments are optimized to run on Modal's cloud infrastructure using H100 GPUs:

```bash
# Upload datasets to Modal volume
python upload_datasets.py --datasets "IEMOCAP_Final,IEMOCAP_Filtered"

# Run comprehensive experiments
python run_all_experiments.py --run --download
```

### Experiment Parameters

- **Text Models**: Multiple pretrained transformer models from Hugging Face
- **Audio Features**: Various feature extraction methods for multimodal experiments
- **Fusion Types**: Different strategies for combining text and audio
- **ML Classifiers**: Various classifiers for the second stage

## Directory Structure

```
emod/
├── src/                  # Core implementation files
│   ├── core/             # Core model implementation
│   ├── processing/       # Data and results processing
│   ├── utils/            # General utilities and logging
│   └── modal/            # Modal integration
├── experiments/          # Experiment configurations
├── scripts/              # Utility scripts
├── results/              # Experiment results
├── reports/              # Generated reports
├── Datasets/             # IEMOCAP and processed data
└── *.py                  # Command-line scripts
```

## Downloading and Analyzing Results

After experiments complete, results can be downloaded and analyzed:

```bash
# Download experiment results
python download_successful.py --output-dir ./emod_results

# Analyze experiment results
python analyze_results.py --dir ./emod_results --plots
```

Results include:
- Training logs with metrics per epoch
- Final performance metrics (MSE, RMSE, MAE, R²)
- Model checkpoints for best-performing models
- VAD predictions for further analysis

## Performance Metrics

The system evaluates performance using:
- MSE, RMSE, MAE for VAD regression performance
- Accuracy, F1-score for emotion classification
- R² for correlation between predicted and ground truth values

## Extending the System

### Adding New Text Models

Use any model available in Hugging Face Transformers:

```bash
python run_all_experiments.py --models "your-new-model" --datasets "IEMOCAP_Final"
```

### Adding New Audio Features

1. Add extraction function in `src/core/emod_multimodal.py`
2. Run with the new feature

### Creating New Fusion Strategies

1. Add fusion function in `src/core/emod_multimodal.py`
2. Run with the new strategy

## Troubleshooting

For common issues:

1. **Out of Memory Errors**: Reduce batch size (`--batch-size 8`)
2. **Modal Issues**: Run `modal token new` to re-authenticate
3. **Performance Issues**: Increase epochs or try different models

## Citation

If you use this system in your research, please cite:

```
@misc{emod2023,
  author = {Your Name},
  title = {EMOD: A Two-Stage Emotion Recognition System},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/emod}
}
```

## Report Improvements

Based on reviewer feedback, the following improvements have been made to the CS297-298-Xiangyi-Report:

1. **Shortened Report**: The report has been reduced from ~60 pages to ~30 pages (48.5% reduction), focusing more on the experiment and results sections. The shortened version is available at `CS297-298-Xiangyi-Report/main_shortened.tex`.

2. **Improved Figure Readability**: 
   - Figures with arrows have been enhanced for better visibility
   - Key diagrams (system architecture, fusion strategies) have been regenerated with clearer arrows
   - The improved figures are stored in `CS297-298-Xiangyi-Report/Figures_Improved/`

3. **Standardized Captions**:
   - Image and table captions have been updated to standard format
   - Paragraphs have been removed from captions, keeping only concise descriptions

4. **Updated Metrics**:
   - Tables 3 and 4 now include Macro F1, Micro F1, Precision, and Recall metrics
   - Both train and test performance metrics are reported to assess overfitting/underfitting
   - R2 metric has been removed from AVD prediction (first stage) as it's not meaningful for non-linear models
   - The updated tables are available in `updated_tables/`

5. **Restored Missing References**:
   - Fixed issue where 12 out of 34 references were missing after shortening the report
   - All 34 original citations are now included in the bibliography
   - A complete version with all references is available at `CS297-298-Xiangyi-Report/main_complete.tex`

The scripts used for these improvements are:
- `shorten_report.py`: Reduces report length while preserving key sections
- `update_metrics.py`: Updates metrics in Tables 3 and 4
- `improve_figures.py`: Enhances figure readability, focusing on arrows
- `restore_references.py`: Restores missing citations

To recompile the final report, use the following command:
```
cd CS297-298-Xiangyi-Report
pdflatex main_complete.tex
bibtex main_complete
pdflatex main_complete.tex
pdflatex main_complete.tex
```

### Next Steps

To finalize the report:
1. Review the shortened report `main_shortened.tex` and make any necessary adjustments
2. Replace Tables 3 and 4 in the report with the updated versions from `updated_tables/`
3. Update figure references to use the improved figures from `Figures_Improved/`
4. Run LaTeX compilation to generate the final PDF
