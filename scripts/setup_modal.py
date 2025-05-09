#!/usr/bin/env python3
"""
Set up Modal authentication with the provided token ID and token secret.
"""

import subprocess
import sys

def setup_modal_auth(token_id, token_secret):
    """Set up Modal authentication using the provided token credentials."""
    try:
        print(f"Setting up Modal authentication with token ID: {token_id}")
        
        # Run modal token set command
        cmd = ["modal", "token", "set", 
               "--token-id", token_id, 
               "--token-secret", token_secret]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Modal authentication successful!")
            return True
        else:
            print(f"Error setting Modal token: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python setup_modal.py <token_id> <token_secret>")
        sys.exit(1)
    
    token_id = sys.argv[1]
    token_secret = sys.argv[2]
    
    if setup_modal_auth(token_id, token_secret):
        print("You can now run your Modal scripts!")
    else:
        print("Failed to set up Modal authentication.")
        sys.exit(1) 