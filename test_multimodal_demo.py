#!/usr/bin/env python3
"""Demo script to test multi-modal functionality with a smaller model."""

from run_multimodal import setup_model, generate_response, print_response, load_image
import sys
import os

def test_with_deepseek():
    """Test with DeepSeek model (text-only)"""
    print("🧪 Testing with DeepSeek model (text-only)")
    
    try:
        processor, model, is_multimodal = setup_model("deepseek-ai/deepseek-coder-1.3b-base")
        
        prompt = "Write a Python function to calculate the factorial of a number"
        response = generate_response(processor, model, prompt, model_id="deepseek-ai/deepseek-coder-1.3b-base")
        print_response(response)
        
        print("\n✅ DeepSeek test completed successfully!")
        
    except Exception as e:
        print(f"❌ DeepSeek test failed: {e}")
        return False
    
    return True

def test_with_sample_image():
    """Test with sample image if available"""
    print("\n🧪 Testing image loading functionality")
    
    sample_image_path = "sample_data/sample_image.jpg"
    
    if not os.path.exists(sample_image_path):
        print(f"⚠️  Sample image not found: {sample_image_path}")
        return False
    
    try:
        image = load_image(sample_image_path)
        print(f"✅ Successfully loaded image: {image.size}")
        return True
        
    except Exception as e:
        print(f"❌ Image loading test failed: {e}")
        return False

def main():
    """Run demo tests"""
    print("🚀 Multi-Modal Demo Test Suite")
    print("=" * 50)
    
    # Test 1: Text-only model
    success1 = test_with_deepseek()
    
    # Test 2: Image loading
    success2 = test_with_sample_image()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"✅ Text-only model: {'PASS' if success1 else 'FAIL'}")
    print(f"✅ Image loading: {'PASS' if success2 else 'FAIL'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Multi-modal framework is ready.")
        print("\n💡 To test with a multi-modal model, run:")
        print("python run_multimodal.py --model_id Qwen/Qwen2.5-Omni-7B --image_path sample_data/sample_image.jpg --prompt 'Describe this image'")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()