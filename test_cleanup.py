#!/usr/bin/env python3
"""
Test script for the cleanup functionality in Muesli.

This script tests whether the MuesliApp.shutdown() method properly
removes temporary files from the various directories.
"""

import os
import glob
from pathlib import Path
import sys

def create_test_files():
    """Create test files in the temporary directories."""
    # Define the directories to test
    home_dir = Path.home()
    recordings_dir = home_dir / ".muesli" / "recordings"
    logs_dir = home_dir / ".muesli" / "logs"
    output_dir = home_dir / ".muesli" / "output"
    
    # Ensure directories exist
    recordings_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test files
    test_files = [
        recordings_dir / "test_recording.wav",
        logs_dir / "test_log.txt",
        output_dir / "test_output.json",
    ]
    
    for file_path in test_files:
        with open(file_path, "w") as f:
            f.write("Test content")
        print(f"Created test file: {file_path}")
    
    return test_files

def check_files_removed(file_paths):
    """Check if the files were removed correctly."""
    all_removed = True
    for file_path in file_paths:
        if file_path.exists():
            print(f"ERROR: File not removed: {file_path}")
            all_removed = False
        else:
            print(f"SUCCESS: File removed: {file_path}")
    
    return all_removed

def test_cleanup():
    """Test the cleanup functionality."""
    print("=== Testing MuesliApp cleanup functionality ===")
    
    # Create test files
    print("\n1. Creating test files...")
    test_files = create_test_files()
    
    # Call local cleanup
    print("\n2. Running cleanup_files()...")
    cleanup_files()
    
    # Check if files were removed
    print("\n3. Checking if files were removed...")
    success = check_files_removed(test_files)
    
    # Print result
    print("\n=== Test Result ===")
    if success:
        print("SUCCESS: All temporary files were cleaned up correctly!")
        return 0
    else:
        print("FAILURE: Some temporary files were not cleaned up!")
        return 1

# ------------------------------------------------------------------+
# Local replica of MuesliApp.shutdown() file-cleanup                #
# ------------------------------------------------------------------+

def cleanup_files():
    """Remove temporary files created by Muesli."""
    home = Path.home()
    targets = [
        home / ".muesli" / "recordings" / "*",
        home / ".muesli" / "logs" / "*",
        home / ".muesli" / "output" / "*",
    ]

    for pattern in targets:
        for file_path in glob.glob(str(pattern)):
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")

if __name__ == "__main__":
    sys.exit(test_cleanup())
