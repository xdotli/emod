# EMOD Experiment Comparison Report

Generated: 2025-05-08 20:34:23

## Experiment Configurations

| Parameter | final_dataset_full | filtered_dataset_full |
|----------|------------|------------|

## Performance Metrics

### VAD Prediction Metrics

| Metric | final_dataset_full | filtered_dataset_full |
|----------|------------|------------|
| mae | [0.5198, 0.5484, 0.0] | [0.4949, 0.556, 0.0] | 
| mse | [0.4664, 0.4849, 0.0] | [0.4208, 0.4994, 0.0] | 
| r2 | [0.5004, 0.0411, 0.0] | [0.5485, 0.138, 0.0] | 
| rmse | [0.6829, 0.6964, 0.0] | [0.6487, 0.7067, 0.0] | 

### Emotion Classification Metrics

| Metric | final_dataset_full | filtered_dataset_full |
|----------|------------|------------|
| accuracy | 0.6130 | 0.6084 | 
| classification_report | {'angry': {'precision': 0.65, 'recall': 0.79, 'f1-score': 0.72, 'support': 373}, 'happy': {'precision': 0.81, 'recall': 0.6, 'f1-score': 0.69, 'support': 248}, 'neutral': {'precision': 0.4, 'recall': 0.5, 'f1-score': 0.44, 'support': 207}, 'sad': {'precision': 0.63, 'recall': 0.31, 'f1-score': 0.42, 'support': 123}} | {'angry': {'precision': 0.64, 'recall': 0.78, 'f1-score': 0.7, 'support': 352}, 'happy': {'precision': 0.81, 'recall': 0.69, 'f1-score': 0.75, 'support': 221}, 'neutral': {'precision': 0.38, 'recall': 0.49, 'f1-score': 0.43, 'support': 171}, 'sad': {'precision': 0.65, 'recall': 0.17, 'f1-score': 0.27, 'support': 132}} | 
| f1_macro | 0.5650 | 0.5349 | 
| f1_weighted | 0.6098 | 0.5932 | 

## Training History

See the generated plots for visual comparison of training histories.

## Conclusion

Analysis of the results shows:

- **VAD Prediction Performance**: 