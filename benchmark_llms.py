#!/usr/bin/env python3
"""
Benchmark different LLMs for emotion recognition tasks.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

from utils.config import get_config
from utils.openrouter_client import OpenRouterClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define standard emotion categories for evaluation
STANDARD_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# Map similar emotions to standard categories
EMOTION_MAPPING = {
    # Anger
    "anger": "anger", "angry": "anger", "annoyance": "anger", "rage": "anger", "frustration": "anger",
    # Disgust
    "disgust": "disgust", "disgusted": "disgust", "repulsion": "disgust", "revulsion": "disgust",
    # Fear
    "fear": "fear", "afraid": "fear", "scared": "fear", "anxious": "fear", "nervous": "fear", "worry": "fear",
    # Joy
    "joy": "joy", "happy": "joy", "happiness": "joy", "excited": "joy", "elated": "joy", "pleased": "joy", 
    "content": "joy", "satisfied": "joy", "cheerful": "joy",
    # Sadness
    "sad": "sadness", "sadness": "sadness", "depressed": "sadness", "unhappy": "sadness", "grief": "sadness",
    "melancholy": "sadness", "disappointed": "sadness", "regret": "sadness",
    # Surprise
    "surprise": "surprise", "surprised": "surprise", "astonished": "surprise", "amazed": "surprise", 
    "shocked": "surprise",
    # Neutral
    "neutral": "neutral", "calm": "neutral", "indifferent": "neutral", "balanced": "neutral"
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark LLMs for emotion recognition")
    
    # Dataset options
    parser.add_argument("--data-path", type=str, default="data/processed_data.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--sample-size", type=int, default=100,
                        help="Number of samples to use for benchmarking")
    
    # Model options
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["anthropic/claude-3-7-sonnet-20240620", 
                                 "openai/gpt-4o-2024-05-13",
                                 "deepseek-ai/deepseek-coder-v2"],
                        help="List of models to benchmark")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    parser.add_argument("--save-responses", action="store_true",
                        help="Save raw LLM responses")
    
    # Execution options
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Maximum number of parallel workers")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Timeout in seconds for each LLM request")
    
    return parser.parse_args()

def standardize_emotion(emotion: str) -> str:
    """Map an emotion to one of the standard categories."""
    emotion = emotion.lower().strip()
    
    # Direct match
    if emotion in STANDARD_EMOTIONS:
        return emotion
    
    # Check mapping
    if emotion in EMOTION_MAPPING:
        return EMOTION_MAPPING[emotion]
    
    # Find closest match
    for key, value in EMOTION_MAPPING.items():
        if key in emotion or emotion in key:
            return value
    
    # Default to neutral if no match found
    logger.warning(f"Could not map emotion '{emotion}' to standard categories. Defaulting to 'neutral'.")
    return "neutral"

def extract_emotion_from_response(response: Dict[str, Any]) -> str:
    """Extract the primary emotion from an LLM response."""
    try:
        content = response['choices'][0]['message']['content']
        
        # Try to parse as JSON
        try:
            # Check if content contains JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
                data = json.loads(json_str)
            elif "```" in content:
                json_str = content.split("```")[1].strip()
                data = json.loads(json_str)
            else:
                data = json.loads(content)
            
            # Extract primary emotion
            if "primary_emotion" in data:
                if isinstance(data["primary_emotion"], dict) and "name" in data["primary_emotion"]:
                    return data["primary_emotion"]["name"]
                else:
                    return str(data["primary_emotion"])
            elif "emotion" in data:
                return str(data["emotion"])
        except (json.JSONDecodeError, KeyError, IndexError):
            # If JSON parsing fails, try to extract emotion from text
            lower_content = content.lower()
            
            # Look for phrases like "primary emotion: happiness"
            for emotion in EMOTION_MAPPING.keys():
                if f"primary emotion: {emotion}" in lower_content or f"primary emotion is {emotion}" in lower_content:
                    return emotion
            
            # Look for any emotion mentioned
            for emotion in EMOTION_MAPPING.keys():
                if emotion in lower_content:
                    return emotion
        
        # Default to neutral if extraction fails
        logger.warning(f"Could not extract emotion from response. Defaulting to 'neutral'.")
        return "neutral"
    except Exception as e:
        logger.error(f"Error extracting emotion from response: {e}")
        return "neutral"

def analyze_text_with_llm(
    client: OpenRouterClient,
    model: str,
    text: str,
    timeout: int = 30
) -> Dict[str, Any]:
    """Analyze text with an LLM to detect emotions."""
    system_prompt = """
    You are an expert emotion detection system. Analyze the provided text and identify the primary emotion expressed.
    
    Focus on these primary emotions: anger, disgust, fear, joy, sadness, surprise, neutral.
    
    Return your analysis in JSON format with the following structure:
    {
        "primary_emotion": "emotion_name",
        "confidence": confidence_score,
        "explanation": "brief explanation"
    }
    
    Only return the JSON, no additional text.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Text to analyze: {text}"}
    ]
    
    try:
        start_time = time.time()
        response = client.generate_text(model, messages, temperature=0.3, max_tokens=500)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        return {
            "response": response,
            "response_time": response_time
        }
    except Exception as e:
        logger.error(f"Error analyzing text with {model}: {e}")
        return {
            "error": str(e),
            "response_time": timeout
        }

def benchmark_models(args):
    """Benchmark different LLMs on emotion recognition tasks."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    try:
        data = pd.read_csv(args.data_path)
        logger.info(f"Loaded dataset with {len(data)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Sample data for benchmarking
    if args.sample_size < len(data):
        benchmark_data = data.sample(args.sample_size, random_state=42)
    else:
        benchmark_data = data
    
    logger.info(f"Using {len(benchmark_data)} samples for benchmarking")
    
    # Initialize OpenRouter client
    client = OpenRouterClient()
    
    # Prepare results storage
    results = []
    raw_responses = {}
    
    # Benchmark each model
    for model in args.models:
        logger.info(f"Benchmarking model: {model}")
        model_results = []
        model_raw_responses = []
        
        # Process samples with ThreadPoolExecutor for parallelism
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            
            for idx, row in benchmark_data.iterrows():
                text = row['text']
                ground_truth = row['emotion']
                
                # Submit task to executor
                future = executor.submit(
                    analyze_text_with_llm,
                    client,
                    model,
                    text,
                    args.timeout
                )
                futures.append((future, idx, text, ground_truth))
            
            # Process results as they complete
            for future, idx, text, ground_truth in tqdm(futures, desc=f"Processing with {model}"):
                try:
                    result = future.result(timeout=args.timeout)
                    
                    if "error" in result:
                        predicted_emotion = "neutral"  # Default on error
                        confidence = 0.0
                        explanation = f"Error: {result['error']}"
                    else:
                        response = result["response"]
                        predicted_emotion = extract_emotion_from_response(response)
                        
                        # Try to extract confidence and explanation
                        try:
                            content = response['choices'][0]['message']['content']
                            data = json.loads(content)
                            confidence = data.get("confidence", 0.0)
                            explanation = data.get("explanation", "")
                        except:
                            confidence = 0.0
                            explanation = ""
                    
                    # Standardize emotions
                    std_ground_truth = standardize_emotion(ground_truth)
                    std_predicted_emotion = standardize_emotion(predicted_emotion)
                    
                    # Store result
                    model_results.append({
                        'id': idx,
                        'text': text,
                        'ground_truth': ground_truth,
                        'std_ground_truth': std_ground_truth,
                        'predicted_emotion': predicted_emotion,
                        'std_predicted_emotion': std_predicted_emotion,
                        'confidence': confidence,
                        'explanation': explanation,
                        'response_time': result["response_time"],
                        'correct': std_ground_truth == std_predicted_emotion
                    })
                    
                    # Store raw response
                    if args.save_responses:
                        model_raw_responses.append({
                            'id': idx,
                            'text': text,
                            'response': result.get("response", {"error": "No response"})
                        })
                
                except Exception as e:
                    logger.error(f"Error processing result for {model}: {e}")
                    model_results.append({
                        'id': idx,
                        'text': text,
                        'ground_truth': ground_truth,
                        'std_ground_truth': standardize_emotion(ground_truth),
                        'predicted_emotion': "neutral",
                        'std_predicted_emotion': "neutral",
                        'confidence': 0.0,
                        'explanation': f"Error: {str(e)}",
                        'response_time': args.timeout,
                        'correct': False
                    })
        
        # Calculate metrics
        model_df = pd.DataFrame(model_results)
        
        y_true = model_df['std_ground_truth']
        y_pred = model_df['std_predicted_emotion']
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Calculate average response time
        avg_response_time = model_df['response_time'].mean()
        
        # Store results
        results.append({
            'model': model,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'avg_response_time': avg_response_time,
            'report': report,
            'results': model_results
        })
        
        # Store raw responses
        if args.save_responses:
            raw_responses[model] = model_raw_responses
        
        # Save model results
        model_df.to_csv(os.path.join(args.output_dir, f"{model.replace('/', '_')}_results.csv"), index=False)
        
        # Print summary
        logger.info(f"Model: {model}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 (macro): {f1_macro:.4f}")
        logger.info(f"F1 (weighted): {f1_weighted:.4f}")
        logger.info(f"Average response time: {avg_response_time:.2f} seconds")
        logger.info("Classification Report:")
        print(classification_report(y_true, y_pred))
        print("\n" + "="*80 + "\n")
    
    # Save overall results
    overall_results = {
        'models': [r['model'] for r in results],
        'accuracy': [r['accuracy'] for r in results],
        'f1_macro': [r['f1_macro'] for r in results],
        'f1_weighted': [r['f1_weighted'] for r in results],
        'avg_response_time': [r['avg_response_time'] for r in results]
    }
    
    overall_df = pd.DataFrame(overall_results)
    overall_df.to_csv(os.path.join(args.output_dir, "overall_results.csv"), index=False)
    
    # Save raw responses
    if args.save_responses:
        for model, responses in raw_responses.items():
            with open(os.path.join(args.output_dir, f"{model.replace('/', '_')}_raw_responses.json"), 'w') as f:
                json.dump(responses, f, indent=2)
    
    # Print overall comparison
    logger.info("Overall Comparison:")
    print(overall_df.to_string(index=False))
    
    # Create comparison visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set(style="whitegrid")
        
        # Accuracy comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='models', y='accuracy', data=overall_df)
        plt.title('Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "accuracy_comparison.png"))
        
        # F1 score comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='models', y='f1_macro', data=overall_df, label='F1 (macro)')
        sns.barplot(x='models', y='f1_weighted', data=overall_df, label='F1 (weighted)', alpha=0.7)
        plt.title('F1 Score Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "f1_comparison.png"))
        
        # Response time comparison
        plt.figure(figsize=(10, 6))
        sns.barplot(x='models', y='avg_response_time', data=overall_df)
        plt.title('Average Response Time Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "response_time_comparison.png"))
        
        logger.info(f"Visualizations saved to {args.output_dir}")
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")

def main():
    """Main function."""
    args = parse_args()
    benchmark_models(args)

if __name__ == "__main__":
    main()
