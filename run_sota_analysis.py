#!/usr/bin/env python3
"""
Script to run emotion analysis using state-of-the-art models.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

from utils.config import get_config
from sota_models.integration import SotaEmotionAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run emotion analysis using SOTA models")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio", type=str, help="Path to audio file for analysis")
    input_group.add_argument("--text", type=str, help="Text for analysis")
    
    # Model options
    parser.add_argument("--llm", type=str, default="anthropic/claude-3-7-sonnet-20240620",
                        help="LLM model to use for analysis")
    parser.add_argument("--compare-llms", action="store_true",
                        help="Compare analyses from multiple LLMs")
    
    # Output options
    parser.add_argument("--output", type=str, help="Path to save analysis results (JSON)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    
    # Other options
    parser.add_argument("--language", type=str, help="Language code for transcription")
    parser.add_argument("--timestamps", action="store_true", 
                        help="Include word-level timestamps in transcription")
    parser.add_argument("--no-cuda", action="store_true", 
                        help="Disable CUDA even if available")
    
    return parser.parse_args()

def save_results(results: Dict[str, Any], output_path: Optional[str], pretty: bool = False):
    """Save analysis results to a file or print to stdout."""
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")
    else:
        # Print to stdout
        if pretty:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(results, ensure_ascii=False))

def main():
    """Main function."""
    args = parse_args()
    
    # Check if required environment variables are set
    config = get_config()
    if not config['api_keys']['openrouter']:
        logger.error("OPENROUTER_API_KEY environment variable is not set")
        sys.exit(1)
    
    # Initialize the analyzer
    analyzer = SotaEmotionAnalyzer(use_cuda=not args.no_cuda)
    
    # Run analysis
    if args.audio:
        if not os.path.exists(args.audio):
            logger.error(f"Audio file not found: {args.audio}")
            sys.exit(1)
        
        results = analyzer.analyze_audio(
            audio_path=args.audio,
            language=args.language,
            llm_model=args.llm,
            return_timestamps=args.timestamps
        )
    elif args.text:
        if args.compare_llms:
            results = analyzer.compare_llm_analyses(
                text=args.text
            )
        else:
            results = analyzer.analyze_text(
                text=args.text,
                llm_model=args.llm
            )
    
    # Save or print results
    save_results(results, args.output, args.pretty)

if __name__ == "__main__":
    main()
