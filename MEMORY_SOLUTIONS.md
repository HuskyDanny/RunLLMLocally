# Memory Management Solutions

This document provides solutions for common memory-related errors when running large language models.

## Common Error: "Invalid buffer size: X.XX GB"

### What This Error Means
This error occurs when:
- The model requires more memory than available
- PyTorch cannot allocate a buffer larger than system limits
- GPU memory is insufficient for the model size
- MPS (Metal Performance Shaders) on macOS has memory constraints

### Automatic Solutions Implemented

The script now includes automatic memory management that:

1. **Smart Device Detection**: Automatically detects model size and chooses appropriate device
2. **Quantization**: Uses 8-bit quantization to reduce memory usage by ~50%
3. **Fallback Strategies**: Multiple fallback options when memory errors occur
4. **Platform-Specific Optimizations**: Special handling for macOS, CUDA, and CPU

## Memory Management Features

### 1. Automatic Model Size Detection
```python
# The script automatically detects model sizes:
# - Large models (7B+): Forced to CPU on macOS for stability
# - Medium models (3B-6B): CPU on macOS, smart GPU allocation elsewhere
# - Small models (<3B): Can use MPS/CUDA when available
```

### 2. Quantization Support
```bash
# Enable quantization (default)
python run_multimodal.py --quantization --model_id "Qwen/Qwen2.5-Omni-3B"

# Disable quantization if needed
python run_multimodal.py --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "Hello"
```

### 3. Force CPU Usage
```bash
# Force CPU for any model (useful for memory issues)
python run_multimodal.py --force_cpu --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "Hello world"
```

### 4. Memory-Efficient Loading
The script automatically:
- Uses `low_cpu_mem_usage=True`
- Implements progressive fallback strategies
- Loads models with appropriate precision (float32/float16)

## Recommended Solutions by Error Type

### "Invalid buffer size" Error
1. **Try with force CPU**:
   ```bash
   python run_multimodal.py --force_cpu --model_id "YOUR_MODEL" --prompt "YOUR_PROMPT"
   ```

2. **Use a smaller model**:
   ```bash
   python run_multimodal.py --model_id "Qwen/Qwen2.5-1.5B-Instruct" --prompt "YOUR_PROMPT"
   ```

3. **Enable quantization** (default, but explicit):
   ```bash
   python run_multimodal.py --quantization --model_id "YOUR_MODEL" --prompt "YOUR_PROMPT"
   ```

### "bitsandbytes" Quantization Errors
1. **Disable quantization completely**:
   ```bash
   python run_multimodal.py --no-quantization --model_id "YOUR_MODEL" --prompt "YOUR_PROMPT"
   ```

2. **Update bitsandbytes** (if you want to keep quantization):
   ```bash
   pip install -U bitsandbytes
   ```

3. **The script automatically handles quantization errors** and falls back to non-quantized loading

### MPS Memory Errors on macOS
The script automatically handles these by:
- Forcing CPU for models ≥3B parameters
- Using float32 precision for stability
- Implementing memory-efficient loading

### CUDA Out of Memory
1. **Check available GPU memory**:
   ```python
   import torch
   print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
   ```

2. **Use quantization**:
   ```bash
   python run_multimodal.py --quantization --model_id "YOUR_MODEL"
   ```

3. **Force CPU if GPU memory < 12GB for large models**:
   ```bash
   python run_multimodal.py --force_cpu --model_id "YOUR_MODEL"
   ```

## Model Size Recommendations

| System RAM | GPU Memory | Recommended Model Size | Command Example |
|------------|------------|----------------------|-----------------|
| 8GB        | None       | 1.5B-3B              | `--model_id "Qwen/Qwen2.5-1.5B-Instruct" --force_cpu` |
| 16GB       | None       | 3B-7B                | `--model_id "Qwen/Qwen2.5-3B-Instruct" --quantization` |
| 32GB       | None       | 7B-13B               | `--model_id "Qwen/Qwen2.5-7B-Instruct" --quantization` |
| 16GB       | 8GB        | 7B                   | `--model_id "Qwen/Qwen2.5-7B-Instruct"` |
| 32GB       | 16GB+      | 13B+                 | `--model_id "Qwen/Qwen2.5-14B-Instruct"` |

## Advanced Configuration

### Custom Memory Management
```python
# Programmatic usage with custom settings
from run_multimodal import setup_model

# Force CPU with quantization
processor, model = setup_model(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    use_quantization=True,
    force_cpu=True
)
```

### Environment Variables
You can set these environment variables for additional control:
```bash
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Disable MPS on macOS
export CUDA_VISIBLE_DEVICES=""               # Disable CUDA
```

## Troubleshooting Steps

1. **First, try force CPU**:
   ```bash
   python run_multimodal.py --force_cpu --prompt "test"
   ```

2. **If still failing, use smaller model**:
   ```bash
   python run_multimodal.py --force_cpu --model_id "Qwen/Qwen2.5-1.5B-Instruct" --prompt "test"
   ```

3. **Check system resources**:
   ```bash
   # Check available memory
   free -h  # Linux
   vm_stat  # macOS
   ```

4. **Install quantization dependencies**:
   ```bash
   pip install bitsandbytes>=0.41.0
   ```

## Error Recovery

The script implements automatic error recovery:

1. **Memory Error Detected** → Retry with CPU + float32
2. **Model Loading Failed** → Try fallback model
3. **Quantization Failed** → Load without quantization
4. **Device Error** → Force CPU usage

This ensures the script will work even with limited resources, though it may run slower.

## Performance vs Memory Trade-offs

| Configuration | Memory Usage | Speed | Quality |
|---------------|--------------|-------|---------|
| GPU + float16 | High         | Fast  | Best    |
| GPU + 8-bit   | Medium       | Good  | Good    |
| CPU + float32 | Low          | Slow  | Best    |
| CPU + 8-bit   | Lowest       | Slowest | Good  |

Choose the configuration that best fits your system's capabilities.