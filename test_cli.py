#!/usr/bin/env python3
"""
Test script for CLI functionality of run_multimodal.py
"""

import sys
import os
import subprocess

def test_help_command():
    """Test --help command"""
    print("🧪 Testing --help command...")
    result = subprocess.run([sys.executable, "run_multimodal.py", "--help"], 
                          capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Run multimodal LLM models" in result.stdout
    assert "--model_id" in result.stdout
    assert "--prompt" in result.stdout
    assert "--image_path" in result.stdout
    print("   ✅ Help command test passed")


def test_argument_parsing():
    """Test argument parsing without model execution"""
    print("🧪 Testing argument parsing...")
    
    # Test the exact command from the user
    test_code = '''
import sys
sys.path.append(".")
from run_multimodal import parse_arguments
sys.argv = ["run_multimodal.py", "--model_id", "Qwen/Qwen2.5-Omni-3B", 
           "--prompt", "Describe this image and explain what's happening", 
           "--image_path", "sample_data/sample_image.jpg", "--max_new_tokens", "300"]
args = parse_arguments()
assert args.model_id == "Qwen/Qwen2.5-Omni-3B"
assert args.prompt == "Describe this image and explain what's happening"
assert args.image_path == "sample_data/sample_image.jpg"
assert args.max_new_tokens == 300
assert args.demo == False
print("Arguments parsed correctly!")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise AssertionError("Argument parsing failed")
    
    assert "Arguments parsed correctly!" in result.stdout
    print("   ✅ Argument parsing test passed")


def test_demo_mode_detection():
    """Test demo mode detection when no prompt is provided"""
    print("🧪 Testing demo mode detection...")
    
    test_code = '''
import sys
sys.path.append(".")
from run_multimodal import parse_arguments
sys.argv = ["run_multimodal.py"]
args = parse_arguments()
# Demo mode should be detected in main() when no prompt is provided
assert args.prompt is None
assert args.demo == False  # Initially false, but main() will set it to True
print("Demo mode detection works!")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise AssertionError("Demo mode detection failed")
    
    assert "Demo mode detection works!" in result.stdout
    print("   ✅ Demo mode detection test passed")


def test_syntax_validation():
    """Test that the script has valid syntax"""
    print("🧪 Testing script syntax...")
    
    result = subprocess.run([sys.executable, "-m", "py_compile", "run_multimodal.py"], 
                          capture_output=True, text=True)
    
    assert result.returncode == 0, f"Syntax error: {result.stderr}"
    print("   ✅ Syntax validation test passed")


def run_cli_tests():
    """Run all CLI tests"""
    print("🚀 Running CLI functionality tests...\n")
    
    try:
        test_syntax_validation()
        test_help_command()
        test_argument_parsing()
        test_demo_mode_detection()
        
        print("\n✅ All CLI tests passed!")
        print("\n📋 Ready to use:")
        print("   python run_multimodal.py --model_id 'Qwen/Qwen2.5-Omni-3B' --prompt 'Describe this image and explain what's happening' --image_path sample_data/sample_image.jpg --max_new_tokens 300")
        return True
        
    except Exception as e:
        print(f"\n❌ CLI test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_cli_tests()
    sys.exit(0 if success else 1)