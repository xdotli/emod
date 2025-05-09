#!/usr/bin/env python3
"""
Test Runner for EMOD Project

Discovers and runs all tests in the tests directory.
"""

import os
import sys
import unittest
import argparse
from typing import List, Optional

# Make sure src is in the Python path
sys.path.insert(0, os.path.abspath("."))

def run_tests(test_pattern: Optional[str] = None, verbose: bool = False) -> bool:
    """
    Run all tests in the 'tests' directory.
    
    Args:
        test_pattern: Optional pattern to filter test modules
        verbose: Whether to show verbose output
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    # Set up the test loader
    loader = unittest.TestLoader()
    
    # Discover tests in the 'tests' directory
    if test_pattern:
        print(f"Running tests matching pattern: {test_pattern}")
        if os.path.exists(test_pattern):
            # Pattern is a file path
            suite = loader.discover(os.path.dirname(test_pattern), pattern=os.path.basename(test_pattern))
        else:
            # Pattern is a module name or pattern
            suite = loader.discover('tests', pattern=f"*{test_pattern}*.py")
    else:
        print("Running all tests")
        suite = loader.discover('tests')
    
    # Set up the test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    # Run the tests
    print(f"Discovered {suite.countTestCases()} test cases")
    result = runner.run(suite)
    
    # Print a summary
    print(f"\nTest Summary:")
    print(f"  Ran {result.testsRun} tests")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    # Return True if all tests passed
    return len(result.failures) == 0 and len(result.errors) == 0

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run EMOD tests")
    parser.add_argument("-p", "--pattern", type=str, help="Pattern to filter test modules")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show verbose output")
    return parser.parse_args()

def main() -> int:
    """Main entry point for the test runner."""
    # Parse arguments
    args = parse_args()
    
    # Run tests
    success = run_tests(test_pattern=args.pattern, verbose=args.verbose)
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 