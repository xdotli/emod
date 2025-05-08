# EMOD Codebase Cleanup Plan

## 1. Code Organization & Structure

### Create Consistent Module Structure
- Ensure proper imports and module dependencies
- Remove duplicate code
- Standardize naming conventions

### Consolidate Modal-related Files
- Review existing Modal files (`modal_*.py`)
- Create unified approach for Modal integration
- Preserve core training functionality

### Enhance Documentation
- Add/update docstrings for all functions and classes
- Improve type hints for better code clarity
- Document key design decisions

## 2. Unit Testing Framework

### Create Tests Directory
```
tests/
  __init__.py
  test_data_processing.py
  test_vad_predictor.py
  test_emotion_classifier.py
  test_results_processor.py
  test_report_generator.py
  test_experiment_runner.py
  test_emod_cli.py
```

### Implement Core Tests
- **Data processing tests**:
  - Test data loading and preprocessing
  - Test data splitting
  - Test feature extraction

- **Model tests**:
  - Test VAD predictor model initialization
  - Test emotion classifier initialization
  - Test end-to-end inference pipeline

- **Results processing tests**:
  - Test result loading
  - Test metrics calculation
  - Test report generation

### Create Test Utilities
- Mock Modal functionality for testing without GPU
- Create test fixtures and sample data
- Implement test helpers

## 3. Specific Cleanup Tasks

### Data Processing
- Review and clean `src/utils/data_processing.py`
- Ensure consistent error handling
- Optimize data loading pipeline

### Model Implementation
- Clean up model architecture code
- Ensure consistent interface between stages
- Document model parameters and hyperparameters

### Results Processing
- Streamline result collection and processing
- Fix any bugs in visualization code
- Ensure consistent metric calculation

### CLI Interface
- Review command-line argument handling
- Add better error messages
- Ensure backward compatibility

## 4. Modal Integration

### Preserve Core Modal Functionality
- Ensure grid search experiments work
- Maintain volume storage for results
- Verify authentication flow

### Simplify Modal Configuration
- Create centralized Modal setup module
- Streamline resource allocation
- Improve error handling for Modal-specific issues

## 5. Testing Approach

### Unit Tests
- Implement isolated tests for individual components
- Use mocking to avoid external dependencies
- Focus on core logic validation

### Integration Tests
- Test end-to-end flow with simulated data
- Verify correct interaction between components
- Test CLI with various argument combinations

### Modal-specific Tests
- Create special test mode for Modal functions
- Test with minimal compute to verify functionality
- Skip in normal test runs but available for CI

## 6. Implementation Plan

1. Set up tests directory and framework
2. Clean up core modules one by one with tests
3. Refactor Modal integration with tests
4. Update CLI interface with tests
5. Run full test suite and fix issues
6. Document changes and update README 