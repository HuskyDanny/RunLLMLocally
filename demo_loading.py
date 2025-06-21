#!/usr/bin/env python3
"""
Demo script showing the loading indicator functionality
without requiring the full DeepSeek model download.
"""

import time
import sys
from run_deepseek_improved import show_loading


def demo_quick_operation():
    """Simulate a quick operation"""
    time.sleep(1)
    return "Quick task completed!"


def demo_medium_operation():
    """Simulate a medium-length operation"""
    time.sleep(3)
    return "Medium task completed!"


def demo_long_operation():
    """Simulate a longer operation"""
    time.sleep(5)
    return "Long task completed!"


def main():
    print("🎯 Loading Indicator Demo")
    print("=" * 50)
    
    # Demo different loading scenarios
    result1 = show_loading(demo_quick_operation, "Processing data")
    print(f"Result: {result1}\n")
    
    result2 = show_loading(demo_medium_operation, "Training model")
    print(f"Result: {result2}\n")
    
    result3 = show_loading(demo_long_operation, "Downloading files")
    print(f"Result: {result3}\n")
    
    print("✨ Demo completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0) 