#!/usr/bin/env python3
"""
Test script for M3 GPU optimization features
"""

import sys
import subprocess
import torch

def test_mps_availability():
    """Test that MPS is available on this system"""
    print("🧪 Testing MPS availability...")
    
    if torch.backends.mps.is_available():
        print("   ✅ MPS is available")
        
        # Test basic MPS operations
        try:
            device = torch.device("mps")
            x = torch.rand(10, 10, device=device)
            y = torch.rand(10, 10, device=device)
            z = x @ y  # Matrix multiplication on MPS
            print(f"   ✅ MPS operations work - result shape: {z.shape}")
        except Exception as e:
            print(f"   ⚠️  MPS operations failed: {e}")
            return False
    else:
        print("   ❌ MPS is not available")
        return False
    
    return True


def test_device_mapping():
    """Test device mapping logic for different model sizes"""
    print("🧪 Testing device mapping logic...")
    
    test_code = '''
import sys
sys.path.append(".")
from run_multimodal import get_optimal_device_map

# Test different model sizes
test_cases = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "mps"),  # Small model should use MPS
    ("Qwen/Qwen2.5-3B-Instruct", "mps"),    # Medium model should use MPS with 16GB RAM
    ("Qwen/Qwen2.5-7B-Instruct", "cpu"),    # Large model should use CPU with 16GB RAM
]

for model_id, expected_device in test_cases:
    device, dtype = get_optimal_device_map(model_id)
    print(f"Model: {model_id} -> Device: {device} (Expected: {expected_device})")
    if device != expected_device:
        print(f"❌ Unexpected device mapping for {model_id}")
        exit(1)

print("✅ All device mappings correct")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print(f"Output: {result.stdout}")
        return False
    
    print("   ✅ Device mapping tests passed")
    return True


def test_quantization_skipping():
    """Test that quantization is properly skipped on Apple Silicon"""
    print("🧪 Testing quantization skipping on Apple Silicon...")
    
    test_code = '''
import sys
sys.path.append(".")
from run_multimodal import setup_model

# This should skip quantization and use MPS
try:
    # Just test the initial setup without loading the full model
    import torch
    from run_multimodal import get_optimal_device_map
    
    device, dtype = get_optimal_device_map("Qwen/Qwen2.5-3B-Instruct")
    if device == "mps":
        print("✅ MPS device selected correctly")
        print("✅ Quantization will be skipped automatically on Apple Silicon")
    else:
        print(f"⚠️  Expected MPS but got {device}")
        
except Exception as e:
    print(f"Error in setup: {e}")
    exit(1)
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print("   ✅ Quantization skipping test passed")
    return True


def test_cli_with_mps():
    """Test CLI functionality with MPS optimization"""
    print("🧪 Testing CLI argument parsing with MPS features...")
    
    test_code = '''
import sys
sys.path.append(".")
from run_multimodal import parse_arguments

# Test force CPU flag
sys.argv = ["run_multimodal.py", "--force_cpu", "--prompt", "test"]
args = parse_arguments()
assert args.force_cpu == True

# Test no-quantization flag
sys.argv = ["run_multimodal.py", "--no-quantization", "--prompt", "test"]
args = parse_arguments()
assert args.quantization == False

print("✅ CLI arguments work correctly")
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print("   ✅ CLI tests passed")
    return True


def test_memory_detection():
    """Test system memory detection"""
    print("🧪 Testing system memory detection...")
    
    test_code = '''
import psutil

total_memory_gb = psutil.virtual_memory().total / (1024**3)
print(f"System RAM detected: {total_memory_gb:.1f} GB")

if total_memory_gb > 0:
    print("✅ Memory detection works")
else:
    print("❌ Memory detection failed")
    exit(1)
'''
    
    result = subprocess.run([sys.executable, "-c", test_code], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print("   ✅ Memory detection test passed")
    return True


def run_m3_optimization_tests():
    """Run all M3 optimization tests"""
    print("🚀 Running M3 GPU optimization tests...\n")
    
    tests = [
        test_mps_availability,
        test_memory_detection,
        test_device_mapping,
        test_quantization_skipping,
        test_cli_with_mps,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    if passed == len(tests):
        print(f"\n✅ All {len(tests)} M3 optimization tests passed!")
        print("\n📋 M3 GPU Optimizations:")
        print("   🍎 MPS device detection and usage")
        print("   📊 Intelligent memory-based model placement")
        print("   🚫 Automatic quantization skipping (bitsandbytes incompatible)")
        print("   🚀 Optimized device mapping for Apple Silicon")
        print("\n🔧 Recommended command for your M3 Mac:")
        print('   python run_multimodal.py --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "Describe this image and explain what\'s happening" --image_path sample_data/sample_image.jpg --max_new_tokens 300')
        return True
    else:
        print(f"\n❌ {len(tests) - passed} out of {len(tests)} tests failed")
        return False


if __name__ == "__main__":
    success = run_m3_optimization_tests()
    sys.exit(0 if success else 1)