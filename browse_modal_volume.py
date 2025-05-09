#!/usr/bin/env python3
"""
Script to browse the Modal volume structure
"""

import subprocess
import sys

def browse_volume(path="/"):
    """Browse the Modal volume at the specified path"""
    cmd = ["modal", "volume", "ls", "emod-results-vol", path]
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error listing directory: {result.stderr}")
            return False
            
        print("\nDirectory Contents:")
        print(result.stdout)
        
        # Count directories and files
        lines = result.stdout.strip().split('\n')
        dirs = []
        files = []
        
        for line in lines:
            if not line.strip() or line.startswith("ls:"):
                continue
                
            parts = line.split()
            if len(parts) >= 9:  # standard ls -l format
                if line.startswith('d'):
                    dirs.append(parts[-1])
                else:
                    files.append(parts[-1])
        
        print(f"Found {len(dirs)} directories and {len(files)} files")
        
        # If we found directories, allow the user to navigate into them
        if dirs:
            print("\nAvailable Directories:")
            for i, d in enumerate(dirs):
                print(f"{i+1}. {d}")
                
            print("\nEnter directory number to navigate into (or press Enter to exit):", end=" ")
            choice = input()
            
            if choice.strip():
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(dirs):
                        new_path = f"{path}/{dirs[idx]}".replace("//", "/")
                        browse_volume(new_path)
                    else:
                        print("Invalid directory number")
                except ValueError:
                    print("Please enter a valid number")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def search_results():
    """Search for specific patterns in the Modal volume"""
    print("Searching for logs and results files...")
    search_cmd = ["modal", "volume", "find", "emod-results-vol", "/", "-name", "final_results.json"]
    
    try:
        result = subprocess.run(search_cmd, capture_output=True, text=True)
        print("Search Results:")
        print(result.stdout)
        
        # Try another pattern
        print("\nSearching for logs directories...")
        search_cmd = ["modal", "volume", "find", "emod-results-vol", "/", "-type", "d", "-name", "logs"]
        result = subprocess.run(search_cmd, capture_output=True, text=True)
        print("Logs Directories:")
        print(result.stdout)
        
        # Check for specific patterns matching your screenshot
        pattern = "IEMOCAP_Filtered_multimodal"
        print(f"\nSearching for directories matching: {pattern}")
        search_cmd = ["modal", "volume", "find", "emod-results-vol", "/", "-type", "d", "-name", f"*{pattern}*"]
        result = subprocess.run(search_cmd, capture_output=True, text=True)
        print(f"Matching {pattern} directories:")
        print(result.stdout)
        
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    print("Modal Volume Browser")
    print("--------------------")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--search":
        search_results()
    else:
        browse_volume() 