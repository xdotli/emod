# Mermaid Diagram Definitions for Emotion Recognition Architecture

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Input["Input Data"]
        Audio("Audio Signal")
        Text("Text Transcript")
    end
    
    subgraph Stage1["Stage 1: Modality-Specific Processing"]
        subgraph AudioProcessing["Audio Processing"]
            FeatureExtraction("Feature Extraction")
            AudioModel("Audio Model")
        end
        
        subgraph TextProcessing["Text Processing"]
            Tokenization("Tokenization")
            TransformerModel("Transformer Model")
        end
        
        Audio --> FeatureExtraction
        FeatureExtraction --> AudioModel
        Text --> Tokenization
        Tokenization --> TransformerModel
    end
    
    subgraph Stage2["Stage 2: Multimodal Fusion"]
        EarlyFusion("Early Fusion")
        LateFusion("Late Fusion")
        HybridFusion("Hybrid Fusion")
        AttentionFusion("Attention-Based Fusion")
        
        FusionStrategy{"Selected\nFusion Strategy"}
        
        FusionStrategy --> EarlyFusion & LateFusion & HybridFusion & AttentionFusion
    end
    
    AudioModel --> Stage2
    TransformerModel --> Stage2
    
    Stage2 --> EmotionPrediction("Emotion Prediction")
    
    style Stage1 fill:#d1e7dd,stroke:#198754
    style Stage2 fill:#cfe2ff,stroke:#0d6efd
    style EmotionPrediction fill:#f8d7da,stroke:#dc3545,color:#000
    style FusionStrategy fill:#fff3cd,stroke:#ffc107
```

## 2. Text Model Architecture Detail

```mermaid
flowchart TD
    Text("Text Input") --> Tokenizer("Tokenizer\n(Model-specific)")
    Tokenizer --> TokenIDs("Token IDs + Attention Mask")
    
    subgraph TransformerEncoder["Transformer Encoder"]
        Embedding("Token + Position Embedding")
        Attention("Self-Attention Layer")
        FFN("Feed-Forward Network")
        Norm("Layer Normalization")
        
        Embedding --> Attention
        Attention --> FFN
        FFN --> Norm
        Norm --> |Next Layer| TransformerEncoder
    end
    
    TokenIDs --> Embedding
    
    TransformerEncoder --> CLSToken("CLS Token Representation")
    CLSToken --> ClassificationHead("Classification Head")
    ClassificationHead --> Output("Emotion Prediction")
    
    style TransformerEncoder fill:#d1e7dd,stroke:#198754
    style ClassificationHead fill:#cfe2ff,stroke:#0d6efd
    style Output fill:#f8d7da,stroke:#dc3545,color:#000
```

## 3. Fusion Strategies Comparison

```mermaid
flowchart TD
    subgraph Inputs["Input Modalities"]
        TextFeatures("Text Features")
        AudioFeatures("Audio Features")
    end
    
    subgraph EarlyFusion["Early Fusion"]
        EF_Concat("Feature Concatenation")
        EF_Joint("Joint Processing")
        EF_Output("Prediction")
        
        TextFeatures --> EF_Concat
        AudioFeatures --> EF_Concat
        EF_Concat --> EF_Joint --> EF_Output
    end
    
    subgraph LateFusion["Late Fusion"]
        LF_TextModel("Text Model")
        LF_AudioModel("Audio Model")
        LF_Combine("Decision Combination")
        LF_Output("Prediction")
        
        TextFeatures --> LF_TextModel
        AudioFeatures --> LF_AudioModel
        LF_TextModel --> LF_Combine
        LF_AudioModel --> LF_Combine
        LF_Combine --> LF_Output
    end
    
    subgraph HybridFusion["Hybrid Fusion"]
        HF_TextInter("Text Intermediate\nRepresentation")
        HF_AudioInter("Audio Intermediate\nRepresentation")
        HF_Concat("Intermediate Feature\nConcatenation")
        HF_Joint("Joint Processing")
        HF_Output("Prediction")
        
        TextFeatures --> HF_TextInter
        AudioFeatures --> HF_AudioInter
        HF_TextInter --> HF_Concat
        HF_AudioInter --> HF_Concat
        HF_Concat --> HF_Joint --> HF_Output
    end
    
    subgraph AttentionFusion["Attention-Based Fusion"]
        AF_TextRep("Text Representation\nSequence")
        AF_AudioRep("Audio Representation\nSequence")
        AF_CrossAttn("Cross-Modal\nAttention")
        AF_SelfAttn("Multi-Head\nSelf-Attention")
        AF_Output("Prediction")
        
        TextFeatures --> AF_TextRep
        AudioFeatures --> AF_AudioRep
        AF_TextRep --> AF_CrossAttn
        AF_AudioRep --> AF_CrossAttn
        AF_CrossAttn --> AF_SelfAttn --> AF_Output
    end
    
    style EarlyFusion fill:#d1e7dd,stroke:#198754
    style LateFusion fill:#cfe2ff,stroke:#0d6efd
    style HybridFusion fill:#fff3cd,stroke:#ffc107
    style AttentionFusion fill:#f8d7da,stroke:#dc3545
```

## 4. MFCC Feature Extraction Pipeline

```mermaid
flowchart LR
    Audio("Raw Audio Signal") --> Preemphasis("Pre-emphasis")
    Preemphasis --> Framing("Framing\n(25ms windows)")
    Framing --> Windowing("Hamming Window")
    Windowing --> FFT("Fast Fourier Transform")
    FFT --> PowerSpectrum("Power Spectrum")
    PowerSpectrum --> MelFilter("Mel Filter Bank\n(40 filters)")
    MelFilter --> LogCompression("Log Compression")
    LogCompression --> DCT("Discrete Cosine Transform")
    DCT --> MFCC("MFCC Features\n(40 coefficients)")
    
    MFCC --> DeltaFeatures("Delta & Delta-Delta\nCoefficients")
    DeltaFeatures --> Normalization("Z-score Normalization")
    Normalization --> FinalFeatures("Final Feature Vector")
    
    style Audio fill:#d1e7dd,stroke:#198754
    style MelFilter fill:#cfe2ff,stroke:#0d6efd
    style MFCC fill:#fff3cd,stroke:#ffc107
    style FinalFeatures fill:#f8d7da,stroke:#dc3545
```

## 5. Experiment Execution Framework

```mermaid
flowchart TD
    subgraph Configuration["Experiment Configuration"]
        ExpConfig[("Experiment\nParameters")]
        Models[("Model\nArchitectures")]
        Features[("Audio Feature\nTypes")]
        Fusion[("Fusion\nStrategies")]
        Datasets[("Datasets")]
    end
    
    subgraph Cloud["Modal Cloud Infrastructure"]
        Container("Container Instance")
        GPU("GPU Resources")
        Storage("Data Storage")
        Scheduler("Job Scheduler")
    end
    
    subgraph ExperimentFlow["Experiment Pipeline"]
        DataPrep("Data Preparation")
        ModelTraining("Model Training")
        Evaluation("Performance Evaluation")
        ResultsStorage("Results Storage")
    end
    
    ExpConfig --> Scheduler
    Models --> Container
    Features --> Container
    Fusion --> Container
    Datasets --> Storage
    
    Scheduler --> Container
    Container --> GPU
    Storage --> DataPrep
    
    DataPrep --> ModelTraining
    ModelTraining --> Evaluation
    Evaluation --> ResultsStorage
    ResultsStorage --> |Metrics| Analysis("Analysis &\nVisualization")
    
    style Cloud fill:#d1e7dd,stroke:#198754
    style ExperimentFlow fill:#cfe2ff,stroke:#0d6efd
    style Analysis fill:#f8d7da,stroke:#dc3545,color:#000
```

To render these diagrams:
1. Visit [Mermaid Live Editor](https://mermaid.live/) 
2. Paste the code for each diagram
3. Export as PNG or SVG
4. Include in your LaTeX document

Alternatively, you can use the Mermaid CLI to render these diagrams directly:
```bash
npx @mermaid-js/mermaid-cli -i input.mmd -o output.svg
``` 