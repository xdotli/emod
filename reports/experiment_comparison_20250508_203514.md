# EMOD Experiment Comparison Report

Generated: 2025-05-08 20:35:14

## Experiment Configurations

| Parameter | final_dataset_full | filtered_dataset_full |
|----------|------------|------------|
| batch_size | 16 | 16 | 
| dataset | IEMOCAP_Final.csv | IEMOCAP_Filtered.csv | 
| epochs | 40 | 40 | 
| model_name | roberta-base | roberta-base | 
| test_samples | 951 | 876 | 
| total_samples | 6336 | 5836 | 
| train_samples | 4435 | 4085 | 
| val_samples | 950 | 875 | 

## Performance Metrics

### VAD Prediction Metrics

| Metric | final_dataset_full | filtered_dataset_full |
|----------|------------|------------|
| mae | 0.5198, 0.5484, 0.0000 | 0.4949, 0.5560, 0.0000 | 
| mse | 0.4664, 0.4849, 0.0000 | 0.4208, 0.4994, 0.0000 | 
| r2 | 0.5004, 0.0411, 0.0000 | 0.5485, 0.1380, 0.0000 | 
| rmse | 0.6829, 0.6964, 0.0000 | 0.6487, 0.7067, 0.0000 | 

### Emotion Classification Metrics

| Metric | final_dataset_full | filtered_dataset_full |
|----------|------------|------------|
| accuracy | 0.6130 | 0.6084 | 
| f1_macro | 0.5650 | 0.5349 | 
| f1_weighted | 0.6098 | 0.5932 | 

## Training History

See the generated plots for visual comparison of training histories.

## Conclusion

Analysis of the results shows:

- **VAD Prediction Performance**: filtered_dataset_full performed better with mse of 0.3067 compared to 0.3171 for final_dataset_full.
- **Emotion Classification Performance**: final_dataset_full performed better with accuracy of 0.6130 compared to 0.6084 for filtered_dataset_full.

Overall, this comparison demonstrates the impact of dataset filtering on emotion detection performance.
