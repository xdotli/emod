# Fine-tuning Pre-trained Language Models for Emotion Recognition

This document describes the fine-tuning process for pre-trained language models to improve emotion recognition performance.

## Approach

The fine-tuning approach involves the following steps:

1. **Data Preparation**: Prepare the IEMOCAP dataset for fine-tuning, including text preprocessing and VAD value scaling.
2. **Model Initialization**: Initialize a pre-trained language model (e.g., RoBERTa, BERT) with a regression head for VAD prediction.
3. **Fine-tuning**: Train the model to predict VAD values from text using mean squared error loss.
4. **Evaluation**: Evaluate the fine-tuned model on validation and test sets.
5. **Integration**: Use the fine-tuned model in the emotion recognition pipeline.

## Implementation

### VAD Regressor Model

The VAD regressor model consists of a pre-trained language model encoder followed by a regression head:

```python
class VADRegressor(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super(VADRegressor, self).__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load pre-trained model
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Add regression head
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Get encoder outputs
        if token_type_ids is not None:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and regression head
        pooled_output = self.dropout(pooled_output)
        vad_values = self.regressor(pooled_output)
        
        return vad_values
```

### Fine-tuning Process

The fine-tuning process involves the following steps:

1. **Data Loading**: Load the IEMOCAP dataset and split it into train, validation, and test sets.
2. **Tokenization**: Tokenize the text data using the pre-trained model's tokenizer.
3. **Model Training**: Train the model using mean squared error loss and AdamW optimizer.
4. **Evaluation**: Evaluate the model on validation and test sets using MSE, RMSE, and MAE metrics.
5. **Model Saving**: Save the fine-tuned model for later use.

```python
# Training loop
for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch in train_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        vad = batch['vad'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = criterion(outputs, vad)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        train_loss += loss.item() * input_ids.size(0)
    
    # Calculate average training loss
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad = batch['vad'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs, vad)
            
            # Update statistics
            val_loss += loss.item() * input_ids.size(0)
            
            # Collect predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(vad.cpu().numpy())
    
    # Calculate average validation loss
    val_loss /= len(val_loader.dataset)
    
    # Calculate validation metrics
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    val_mse = mean_squared_error(all_targets, all_preds)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(all_targets, all_preds)
```

## Hyperparameters

The fine-tuning process uses the following hyperparameters:

- **Model**: RoBERTa-base
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 5
- **Max Sequence Length**: 128
- **Optimizer**: AdamW
- **Loss Function**: Mean Squared Error

## Training Log

The training log records the following metrics for each epoch:

- **Train Loss**: Mean squared error loss on the training set
- **Validation Loss**: Mean squared error loss on the validation set
- **Validation MSE**: Mean squared error on the validation set
- **Validation RMSE**: Root mean squared error on the validation set
- **Validation MAE**: Mean absolute error on the validation set

Example log format:
```
epoch,train_loss,val_loss,val_mse,val_rmse,val_mae
1,0.123456,0.123456,0.123456,0.351364,0.281234
2,0.098765,0.098765,0.098765,0.314270,0.245678
...
```

## Results

The fine-tuned model is evaluated on the test set using the following metrics:

- **MSE**: Mean squared error
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error

Additionally, the model is evaluated on each VAD dimension separately to understand its performance on valence, arousal, and dominance prediction.

## Integration with Emotion Recognition Pipeline

The fine-tuned model is integrated into the emotion recognition pipeline as follows:

1. **VAD Prediction**: The fine-tuned model predicts VAD values from text.
2. **Emotion Classification**: A Random Forest classifier predicts emotion labels from the predicted VAD values.
3. **Evaluation**: The complete pipeline is evaluated on the test set.

## Comparison with Zero-shot Approach

The fine-tuned approach is compared with the zero-shot approach using the following metrics:

- **Emotion Classification**: Accuracy, F1-score (macro and weighted)
- **VAD Prediction**: MSE, RMSE, MAE

The comparison helps to understand the benefits of fine-tuning pre-trained language models for emotion recognition.

## Visualizations

The fine-tuning process generates the following visualizations:

- **Loss Curves**: Training and validation loss over epochs
- **Validation Metrics**: MSE, RMSE, and MAE over epochs
- **Confusion Matrix**: Confusion matrix for emotion classification
- **VAD Distribution**: Distribution of predicted VAD values
- **t-SNE Visualization**: t-SNE visualization of VAD values colored by emotion

## Conclusion

Fine-tuning pre-trained language models for VAD prediction can improve emotion recognition performance compared to zero-shot approaches. The fine-tuned model learns to better capture the relationship between text and VAD values, leading to more accurate emotion classification.
