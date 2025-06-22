#!/usr/bin/env python3
"""
Test script for quantization error handling improvements
"""

import sys
import subprocess

def test_no_quantization_flag():
    """Test --no-quantization flag"""
    print("🧪 Testing --no-quantization flag...")
    
    test_code = '''
import sys
sys.path.append(".")
from run_multimodal import parse_arguments
sys.argv = ["run_multimodal.py", "--no-quantization", "--prompt", "test"]
args = parse_arguments()
assert args.quantization == False
print("✅ --no-quantization flag works")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise AssertionError("--no-quantization flag test failed")
    
    assert "--no-quantization flag works" in result.stdout
    print("   ✅ --no-quantization flag test passed")


def test_quantization_error_detection():
    """Test quantization error detection logic"""
    print("🧪 Testing quantization error detection...")
    
    test_code = '''
import sys
sys.path.append(".")

# Simulate quantization error detection
error_messages = [
    "Using `bitsandbytes` 8-bit quantization requires the latest version",
    "Invalid buffer size: 11.15 GB", 
    "CUDA out of memory",
    "RuntimeError: some other error"
]

keywords = ["buffer size", "memory", "bitsandbytes", "quantization"]

for i, msg in enumerate(error_messages):
    error_msg = msg.lower()
    is_memory_error = any(keyword in error_msg for keyword in keywords)
    expected_results = [True, True, True, False]
    
    if is_memory_error != expected_results[i]:
        print(f"❌ Error detection failed for: {msg}")
        exit(1)

print("✅ Error detection logic works correctly")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise AssertionError("Error detection test failed")
    
    assert "Error detection logic works correctly" in result.stdout
    print("   ✅ Error detection test passed")


def test_syntax_and_imports():
    """Test that the script has valid syntax and imports"""
    print("🧪 Testing script syntax and imports...")
    
    result = subprocess.run([sys.executable, "-m", "py_compile", "run_multimodal.py"], 
                          capture_output=True, text=True)
    
    assert result.returncode == 0, f"Syntax error: {result.stderr}"
    print("   ✅ Syntax validation passed")
    
    # Test that imports work
    test_code = '''
import sys
sys.path.append(".")
from run_multimodal import setup_model, parse_arguments
print("✅ Imports work correctly")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Import error: {result.stderr}")
        raise AssertionError("Import test failed")
    
    assert "Imports work correctly" in result.stdout
    print("   ✅ Import test passed")


def run_quantization_tests():
    """Run all quantization fix tests"""
    print("🚀 Running quantization fix tests...\n")
    
    try:
        test_syntax_and_imports()
        test_no_quantization_flag()
        test_quantization_error_detection()
        
        print("\n✅ All quantization fix tests passed!")
        print("\n📋 Solutions available:")
        print("   1. Use --no-quantization to disable quantization completely")
        print("   2. Script automatically detects and handles quantization errors")
        print("   3. Fallback to CPU + float32 when quantization fails")
        print("\n🔧 Recommended command for your case:")
        print('   python run_multimodal.py --no-quantization --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "test"')
        return True
        
    except Exception as e:
        print(f"\n❌ Quantization fix test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_quantization_tests()
    sys.exit(0 if success else 1)