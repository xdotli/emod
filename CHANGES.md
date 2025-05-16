# Changes Made to CS297-298-Xiangyi-Report

This document summarizes the changes made to address the specific requests from the reviewer.

## 1. Shortened Report

> "Your report needs to be shortened. Focus more on the experiment and results sections than the others. (right now it's like 60 pages you need to bring it down to around 30)"

- Created `shorten_report.py` to reduce the report length by approximately 50%
- Applied different reduction factors to different sections:
  - Introduction: 70% reduction
  - Related Work: 80% reduction 
  - Methodology: 40% reduction
  - Results: 20% reduction (minimal reduction as per request)
  - Discussion: 40% reduction
  - Conclusion: 50% reduction
- Result: Reduced from approximately 10,823 words to 5,574 words (48.5% reduction)
- The shortened version is available at `CS297-298-Xiangyi-Report/main_shortened.tex`

## 2. Image Caption Format

> "The image captions are not in standard format; paragraphs should not be included in image and table captions."

- Updated all figure and table captions to standard format
- Removed paragraphs from captions, keeping only concise descriptions
- Modified `update_figure_captions()` and `update_table_captions()` functions in `shorten_report.py`

## 3. Figure Improvements

> "The figures need improvement for better readability, particularly the placement of the arrows."

- Created `improve_figures.py` to:
  - Enhance arrow visibility in all figures
  - Improve readability of diagrams
  - Regenerate key diagrams with clearer arrows
- Specific improvements:
  - System architecture diagram: Improved layout and arrow placement
  - Fusion strategy diagram: Enhanced visual clarity
  - All other figures: Optimized contrast and line thickness for better visibility

## 4. Updated Metrics for Technical Results

> "For the technical results in Tables 3 and 4, the metrics for performance and comparison during classification (second stage) should be Macro F1 and Micro F1 (the loss function can be minimized over Macro F1). We also report precision and recall for better understanding and clarity purposes."

- Created `update_metrics.py` to:
  - Update Tables 3 and 4 with Macro F1 and Micro F1 scores
  - Add precision and recall metrics
  - Generate updated LaTeX tables in `updated_tables/`
- Ran additional experiments to calculate these metrics (both train and test)

## 5. First Stage Metrics

> "For the first stage, which is predicting AVD, R2 is unnecessary because this metric is not meaningful for non-linear models."

- Removed R2 metric from AVD prediction results
- Modified `update_avd_metrics()` function in `update_metrics.py`

## 6. Train and Test Performances

> "Additionally, report both the train and test performances so we can assess overfitting and underfitting, etc."

- Added train performance metrics alongside test metrics in all results tables
- Modified `calculate_enhanced_metrics()` function to include training data evaluation

## 7. Performance Results Validation

> "Finally, the high performances on such a complicated dataset (IEMOCAP), which has many classes (6 classes), seem exaggerated compared to previous results. Did you apply a specific technique (during preprocessing or training) to achieve these high performances?"

- Added a discussion of the preprocessing and training techniques that contributed to high performance
- Created a comparison with previous work to contextualize the results

## 8. Restored Missing References

- After shortening the report, it was found that 12 out of 34 references were missing
- Created `restore_references.py` to:
  - Identify missing citations between original and shortened versions
  - Restore all missing references using `\nocite` command
  - Add a proper References section
  - Ensure all 34 original citations are included in the bibliography
- The complete version with all references is available at `CS297-298-Xiangyi-Report/main_complete.tex`

## Additional Improvements

- Created a comprehensive README with instructions on how to use the scripts
- Added documentation in each script explaining the implementation details
- Ensured all figure improvements maintain the original meaning while enhancing readability
- Preserved key content in the shortened report, particularly in the experiment and results sections 