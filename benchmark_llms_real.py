#!/usr/bin/env python3
"""
Benchmark different LLMs for emotion recognition tasks using real emotional text.
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

# Define standard emotion categories for IEMOCAP evaluation
STANDARD_EMOTIONS = ["anger", "joy", "sadness", "neutral"]

# Map similar emotions to standard IEMOCAP categories
EMOTION_MAPPING = {
    # Anger
    "anger": "anger", "angry": "anger", "annoyance": "anger", "rage": "anger", "frustration": "anger",
    "irritated": "anger", "mad": "anger", "furious": "anger", "annoyed": "anger", "ang": "anger",

    # Joy
    "joy": "joy", "happy": "joy", "happiness": "joy", "excited": "joy", "elated": "joy", "pleased": "joy",
    "content": "joy", "satisfied": "joy", "cheerful": "joy", "hap": "joy", "exc": "joy",

    # Sadness
    "sad": "sadness", "sadness": "sadness", "depressed": "sadness", "unhappy": "sadness", "grief": "sadness",
    "melancholy": "sadness", "disappointed": "sadness", "regret": "sadness", "upset": "sadness",

    # Neutral
    "neutral": "neutral", "calm": "neutral", "indifferent": "neutral", "balanced": "neutral", "neu": "neutral",

    # Map other emotions to closest IEMOCAP category
    "disgust": "anger", "disgusted": "anger", "repulsion": "anger", "revulsion": "anger",
    "fear": "sadness", "afraid": "sadness", "scared": "sadness", "anxious": "sadness", "nervous": "sadness", "worry": "sadness",
    "surprise": "neutral", "surprised": "neutral", "astonished": "neutral", "amazed": "neutral", "shocked": "neutral"
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
                                 "anthropic/claude-2.0",
                                 "openai/gpt-4o-2024-05-13",
                                 "openai/gpt-4",
                                 "google/gemini-2.5-pro"],
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
    utterance_id: str = "",
    timeout: int = 30
) -> Dict[str, Any]:
    """Analyze text with an LLM to detect emotions."""
    system_prompt = """
    You are an expert emotion detection system specializing in conversational utterances from the IEMOCAP dataset.

    The IEMOCAP dataset contains short utterances from acted conversations, often lacking full context. These utterances can be ambiguous and may contain speech disfluencies, interruptions, or incomplete thoughts.

    Analyze the provided utterance and identify the primary emotion expressed from these categories ONLY:
    - anger: expressions of frustration, irritation, hostility
    - joy: expressions of happiness, excitement, pleasure
    - sadness: expressions of grief, disappointment, depression
    - neutral: expressions lacking clear emotional content

    Consider these guidelines:
    1. Focus on the text content, tone, and word choice
    2. Short utterances may have subtle emotional cues
    3. Questions can express emotions (e.g., angry questions)
    4. Incomplete sentences may still convey emotion

    Return your analysis in JSON format with the following structure:
    {
        "primary_emotion": "emotion_name",
        "confidence": confidence_score,
        "explanation": "brief explanation"
    }

    Only return the JSON, no additional text.
    """

    context = f"Utterance ID: {utterance_id}" if utterance_id else ""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Text to analyze: \"{text}\" {context}"}
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

def load_real_emotional_text(data_path: str, sample_size: int) -> pd.DataFrame:
    """Load real emotional text from the IEMOCAP dataset."""
    try:
        # Load the dataset
        data = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with {len(data)} samples")

        # Filter out rows with missing text
        data = data.dropna(subset=['text'])

        # Filter out rows with very short text (less than 5 characters)
        data = data[data['text'].str.len() > 5]

        # Filter out rows with [BREATHING] or other non-speech markers
        data = data[~data['text'].str.contains('\[BREATHING\]')]
        data = data[~data['text'].str.contains('\[LAUGHTER\]')]
        data = data[~data['text'].str.contains('\[NOISE\]')]

        # Standardize emotion labels
        data['std_emotion'] = data['emotion'].apply(standardize_emotion)

        # Filter to only include the standard IEMOCAP emotions
        data = data[data['std_emotion'].isin(STANDARD_EMOTIONS)]

        # Ensure balanced representation of emotions
        emotions = STANDARD_EMOTIONS
        samples_per_emotion = min(sample_size // len(emotions),
                                 min([len(data[data['std_emotion'] == e]) for e in emotions]))

        # Sample equally from each emotion
        balanced_data = pd.DataFrame()
        for emotion in emotions:
            emotion_data = data[data['std_emotion'] == emotion]
            if len(emotion_data) > samples_per_emotion:
                emotion_sample = emotion_data.sample(samples_per_emotion, random_state=42)
            else:
                emotion_sample = emotion_data
            balanced_data = pd.concat([balanced_data, emotion_sample])

        # Shuffle the data
        benchmark_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Using {len(benchmark_data)} samples for benchmarking")
        logger.info(f"Emotion distribution: {benchmark_data['std_emotion'].value_counts().to_dict()}")

        return benchmark_data

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

def benchmark_models(args):
    """Benchmark different LLMs on emotion recognition tasks."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load real emotional text
    benchmark_data = load_real_emotional_text(args.data_path, args.sample_size)

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

                # Get utterance_id if available
                utterance_id = row.get('utterance_id', '')

                # Submit task to executor
                future = executor.submit(
                    analyze_text_with_llm,
                    client,
                    model,
                    text,
                    utterance_id,
                    args.timeout
                )
                futures.append((future, idx, text, ground_truth, utterance_id))

            # Process results as they complete
            for future, idx, text, ground_truth, utterance_id in tqdm(futures, desc=f"Processing with {model}"):
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
                            'utterance_id': utterance_id,
                            'text': text,
                            'ground_truth': ground_truth,
                            'predicted_emotion': std_predicted_emotion,
                            'response': result.get("response", {"error": "No response"})
                        })

                except Exception as e:
                    logger.error(f"Error processing result for {model}: {e}")
                    model_results.append({
                        'id': idx,
                        'utterance_id': utterance_id,
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
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Generate classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

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
        print(classification_report(y_true, y_pred, zero_division=0))
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
        sns.set_theme(style="whitegrid")

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
