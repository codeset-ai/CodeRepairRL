#!/usr/bin/env python3
"""Example script demonstrating the use of different diff implementations."""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.diff import get_diff


def main():
    """Demonstrate the use of different diff implementations."""
    print("=== Diff Utility Example ===\n")
    
    # Sample code
    before_code = """def calculate(x, y):
    # Add two numbers
    result = x + y
    return result"""
    
    after_code = """def calculate(x, y):
    # Add two numbers and multiply by 2
    result = (x + y) * 2
    return result"""
    
    print("Before code:")
    print("------------")
    print(before_code)
    print("\nAfter code:")
    print("-----------")
    print(after_code)
    
    # Get different diff implementations
    search_replace_diff = get_diff('search_replace')
    unified_diff = get_diff('unified')
    
    # Generate diffs
    sr_diff = search_replace_diff.generate_diff(before_code, after_code)
    unified = unified_diff.generate_diff(before_code, after_code)
    
    # Display diffs
    print("\nSearch/Replace Diff:")
    print("-------------------")
    print(sr_diff)
    
    print("\nUnified Diff:")
    print("-------------")
    print(unified)
    
    # Apply diffs
    sr_result = search_replace_diff.apply_diff(before_code, sr_diff)
    unified_result = unified_diff.apply_diff(before_code, unified)
    
    # Verify results
    print("\nVerification:")
    print("-------------")
    print(f"Search/Replace result matches: {sr_result == after_code}")
    print(f"Unified Diff result matches: {unified_result == after_code}")
    
    # Custom unified diff with more context lines
    custom_unified = get_diff('unified', context_lines=5)
    custom_diff = custom_unified.generate_diff(before_code, after_code)
    
    print("\nUnified Diff with 5 context lines:")
    print("---------------------------------")
    print(custom_diff)


if __name__ == "__main__":
    main()