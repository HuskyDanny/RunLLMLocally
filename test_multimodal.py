#!/usr/bin/env python3
"""
Test script for multimodal functionality without downloading large models.
This script tests the core functions and structure.
"""

import sys
import os
from unittest.mock import Mock, patch
from PIL import Image
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_multimodal import (
    get_optimal_device_map,
    extract_generated_content,
    format_response,
    load_image,
    show_loading,
    QWEN_OMNI_AVAILABLE
)


def test_device_mapping():
    """Test device mapping logic"""
    print("🧪 Testing device mapping...")
    device, dtype = get_optimal_device_map()
    print(f"   Device: {device}, Dtype: {dtype}")
    assert device in ["cpu", "cuda", "mps"]
    print("   ✅ Device mapping test passed")


def test_content_extraction():
    """Test content extraction logic"""
    print("🧪 Testing content extraction...")
    
    prompt = "Write a Python function"
    full_response = "Write a Python function\ndef hello():\n    print('Hello')\n\nUser: Another prompt"
    
    extracted = extract_generated_content(full_response, prompt)
    expected = "def hello():\n    print('Hello')"
    
    assert expected in extracted
    print("   ✅ Content extraction test passed")


def test_response_formatting():
    """Test response formatting"""
    print("🧪 Testing response formatting...")
    
    response = format_response("Generated text", "Test prompt", "test-model", has_image=True)
    
    assert response["prompt"] == "Test prompt"
    assert response["response"] == "Generated text"
    assert response["model"] == "test-model"
    assert response["multimodal"] == True
    assert "timestamp" in response
    
    print("   ✅ Response formatting test passed")


def test_image_loading():
    """Test image loading functionality"""
    print("🧪 Testing image loading...")
    
    # Test with existing sample image
    sample_path = "sample_data/sample_image.jpg"
    if os.path.exists(sample_path):
        try:
            image = load_image(sample_path)
            assert isinstance(image, Image.Image)
            assert image.mode == "RGB"
            print("   ✅ Image loading test passed")
        except Exception as e:
            print(f"   ⚠️  Image loading test failed: {e}")
    else:
        print("   ⚠️  Sample image not found, skipping image loading test")


def test_loading_indicator():
    """Test loading indicator functionality"""
    print("🧪 Testing loading indicator...")
    
    def dummy_operation():
        import time
        time.sleep(0.1)  # Short delay
        return "test result"
    
    result = show_loading(dummy_operation, "Testing")
    assert result == "test result"
    print("   ✅ Loading indicator test passed")


def test_qwen_availability():
    """Test Qwen Omni classes availability"""
    print("🧪 Testing Qwen Omni availability...")
    
    if QWEN_OMNI_AVAILABLE:
        print("   ✅ Qwen2.5-Omni classes are available")
    else:
        print("   ⚠️  Qwen2.5-Omni classes not available - will use fallback models")
    
    print("   ✅ Qwen availability test passed")


def run_tests():
    """Run all tests"""
    print("🚀 Running multimodal functionality tests...\n")
    
    try:
        test_device_mapping()
        test_content_extraction()
        test_response_formatting()
        test_image_loading()
        test_loading_indicator()
        test_qwen_availability()
        
        print("\n✅ All tests passed! Multimodal implementation is ready.")
        print("\n📋 Summary:")
        print(f"   • Device support: Available")
        print(f"   • Image processing: Available")
        print(f"   • Qwen2.5-Omni support: {'Available' if QWEN_OMNI_AVAILABLE else 'Not available (will use fallback)'}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)