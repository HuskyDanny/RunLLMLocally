# Apple M3 GPU Optimization Guide

This document explains how the multimodal script has been optimized to leverage your Apple M3 chip's powerful integrated GPU using Metal Performance Shaders (MPS).

## M3 Chip Capabilities

Your Apple M3 setup provides:
- **Unified Memory Architecture**: GPU and CPU share the same memory pool
- **Metal Performance Shaders (MPS)**: Hardware-accelerated deep learning operations  
- **16GB Unified Memory**: Sufficient for medium-sized models (3B parameters)
- **High Memory Bandwidth**: Efficient data transfer between compute units

## Optimizations Implemented

### 1. Smart Device Detection
The script automatically detects your M3 capabilities:
```
🍎 Apple Silicon detected with MPS support
📊 System RAM: 16.0 GB
🚀 Medium model with sufficient RAM - using MPS
```

### 2. Model Size-Based Placement
- **Small models (≤2B)**: Always use MPS for optimal performance
- **Medium models (3B-6B)**: Use MPS if RAM ≥16GB (✅ Your system)
- **Large models (≥7B)**: Use MPS if RAM ≥32GB, otherwise CPU

### 3. Automatic Quantization Handling
```
🍎 Skipping quantization on Apple Silicon (bitsandbytes not compatible)
💡 MPS will provide GPU acceleration instead
```

The script automatically:
- Detects Apple Silicon platform
- Skips bitsandbytes quantization (NVIDIA CUDA only)
- Uses native MPS acceleration instead

### 4. Optimized Memory Management
- Uses `torch.float16` precision on MPS for memory efficiency
- Implements intelligent device mapping with `device_map="auto"`
- Fallback strategies for memory errors

## Performance Benefits

| Configuration | Device | Memory Usage | Speed | Best For |
|---------------|--------|-------------|-------|----------|
| **M3 + MPS** | mps | Medium | **Fast** | 3B models |
| CPU Only | cpu | Low | Slow | Large models |
| Force CPU | cpu | Low | Slow | Compatibility |

## Usage Examples

### Optimal M3 Usage (Recommended)
```bash
# Let the script automatically detect and use MPS
python run_multimodal.py --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "Describe this image and explain what's happening" --image_path sample_data/sample_image.jpg --max_new_tokens 300
```

### Force CPU (if needed for troubleshooting)
```bash
python run_multimodal.py --force_cpu --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "test"
```

### Text-only with MPS acceleration
```bash
python run_multimodal.py --prompt "Explain quantum computing in simple terms"
```

## Technical Details

### MPS Device Mapping
```python
# Automatic MPS configuration
if device == "mps":
    model_kwargs["device_map"] = "auto"
    print("🍎 Using MPS device mapping for Apple Silicon GPU")
```

### Memory-Aware Model Selection
```python
# 16GB RAM detected - can handle 3B models on MPS
if is_medium_model and total_memory_gb >= 16:
    print("🚀 Medium model with sufficient RAM - using MPS")
    return "mps", torch.float16
```

## Troubleshooting

### If MPS Not Working
1. **Check MPS availability**:
   ```python
   import torch
   print(torch.backends.mps.is_available())
   ```

2. **Force CPU as fallback**:
   ```bash
   python run_multimodal.py --force_cpu --prompt "test"
   ```

3. **Update PyTorch**:
   ```bash
   pip install torch>=2.0.0
   ```

### Memory Issues
If you encounter memory errors:
1. The script automatically falls back to CPU
2. Try smaller models: `Qwen/Qwen2.5-1.5B-Instruct`
3. Reduce `--max_new_tokens` parameter

## Performance Monitoring

You can monitor GPU usage during model execution:
```bash
# In another terminal
sudo powermetrics --samplers gpu_power -n 1 -i 5000
```

## Expected Performance

With your M3 + 16GB setup:
- **3B models**: Excellent performance on MPS
- **7B models**: Will use CPU (slower but stable)
- **Text generation**: ~20-50 tokens/second (MPS)
- **Image processing**: Hardware-accelerated

## Why This Matters

**Before optimization**: 
- Forced CPU usage even with capable GPU
- Attempted incompatible quantization
- Slower inference times

**After optimization**:
- ✅ Native MPS GPU acceleration
- ✅ No incompatible quantization attempts  
- ✅ Faster inference on supported models
- ✅ Automatic fallback strategies

Your M3 chip is now being utilized to its full potential for AI inference!