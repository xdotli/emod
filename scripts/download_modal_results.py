#!/usr/bin/env python3
"""
Download results from Modal volume for local analysis.
This script uses the Modal CLI to download all experiment results.
"""

import os
import subprocess
import sys
import argparse
from datetime import datetime

def ensure_modal_installed():
    """Ensure Modal CLI is installed"""
    try:
        subprocess.run(["modal", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Modal CLI not found or not properly installed.")
        print("Please install it with: pip install modal")
        return False

def download_volume_data(target_dir="./results", verbose=False):
    """Download all data from the Modal volume to a local directory"""
    print(f"Downloading Modal volume data to {target_dir}...")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Build the command
    command = ["modal", "volume", "get", "emod-results-vol:/", target_dir]
    if verbose:
        print(f"Running command: {' '.join(command)}")
    
    # Run the command
    try:
        result = subprocess.run(command, check=True, capture_output=not verbose)
        print(f"Successfully downloaded Modal volume data to {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading Modal volume data: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        return False

def list_volume_contents(verbose=False):
    """List contents of the Modal volume without downloading"""
    print("Listing Modal volume contents...")
    
    # Build the command
    command = ["modal", "volume", "ls", "emod-results-vol:/"]
    if verbose:
        print(f"Running command: {' '.join(command)}")
    
    # Run the command
    try:
        result = subprocess.run(command, check=True, capture_output=not verbose)
        if not verbose and result.stdout:
            print("Volume contents:")
            print(result.stdout.decode())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error listing Modal volume contents: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr.decode()}")
        return False

def authenticate_modal():
    """Ensure Modal authentication is active"""
    print("Checking Modal authentication...")
    
    try:
        # Try a simple command that requires authentication
        result = subprocess.run(
            ["modal", "volume", "ls"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("Modal authentication is active.")
        return True
    except subprocess.CalledProcessError as e:
        if "Please run 'modal token new'" in e.stderr.decode():
            print("Modal authentication required.")
            print("Running 'modal token new'...")
            try:
                subprocess.run(["modal", "token", "new"], check=True)
                print("Authentication successful.")
                return True
            except subprocess.CalledProcessError:
                print("Failed to authenticate with Modal.")
                return False
        else:
            print(f"Error checking Modal authentication: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Download EMOD experiment results from Modal volume")
    parser.add_argument("--list-only", action="store_true", help="Only list volume contents without downloading")
    parser.add_argument("--target-dir", default="./results", help="Target directory for downloaded data (default: ./results)")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    args = parser.parse_args()
    
    print(f"=== EMOD Modal Results Downloader ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if Modal CLI is installed
    if not ensure_modal_installed():
        return
    
    # Authenticate with Modal if needed
    if not authenticate_modal():
        return
    
    # List or download volume contents
    if args.list_only:
        list_volume_contents(args.verbose)
    else:
        download_volume_data(args.target_dir, args.verbose)
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 