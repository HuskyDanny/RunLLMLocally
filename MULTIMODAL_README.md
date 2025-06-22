# Multimodal LLM Implementation

This document describes the multimodal implementation that allows running text and vision-language models from Hugging Face Hub.

## Overview

The `run_multimodal.py` script implements a general-purpose model runner that supports:
- Text-only language models
- Multimodal vision-language models (text + image inputs)
- Dynamic model loading from Hugging Face Hub
- Automatic device mapping (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
- Smart memory management with quantization support

## Key Features

### 🔧 Technical Features
- **AutoProcessor Support**: Uses `transformers.AutoProcessor` for handling both text and image inputs
- **Dynamic Device Mapping**: Automatically selects the best available device (MPS/CUDA/CPU)
- **Memory Optimization**: Includes special handling for large models on macOS to avoid MPS memory issues
- **Chat Templates**: Uses model-specific chat templates for proper prompt formatting
- **Response Cleaning**: Extracts and cleans generated content from model outputs
- **History Saving**: Automatically saves chat histories to JSON files

### 🖼️ Multimodal Capabilities
- **Image Processing**: Built-in image loading and preprocessing with PIL
- **Mixed Input**: Supports prompts that combine text instructions with image analysis
- **Format Support**: Handles common image formats (JPEG, PNG, etc.)
- **Automatic Conversion**: Converts images to RGB format as needed

## Usage Examples

### Command Line Interface

The script now supports full command-line usage with the following options:

```bash
# View all available options
python run_multimodal.py --help

# Text-only generation
python run_multimodal.py --prompt "Explain quantum computing"

# Multimodal generation with image
python run_multimodal.py --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "Describe this image and explain what's happening" --image_path sample_data/sample_image.jpg --max_new_tokens 300

# Use different model with custom parameters
python run_multimodal.py --model_id "Qwen/Qwen2.5-7B-Instruct" --prompt "Write a Python function" --max_new_tokens 200
```

### Demo Mode (Default)
```bash
# Run demo mode (shows both text and multimodal examples)
python run_multimodal.py

# Explicit demo mode
python run_multimodal.py --demo
```

### Advanced Options
```bash
# Disable history saving
python run_multimodal.py --prompt "Hello world" --save_history

# Use custom image with specific model
python run_multimodal.py --model_id "Qwen/Qwen2.5-Omni-7B" --prompt "What's in this photo?" --image_path /path/to/image.jpg --max_new_tokens 500
```

### Programmatic Usage
```python
from run_multimodal import setup_model, generate_response, load_image

# Setup model
processor, model = setup_model("Qwen/Qwen2.5-Omni-7B")

# Text-only generation
response = generate_response(processor, model, "Explain quantum computing")

# Multimodal generation
image = load_image("path/to/your/image.jpg")
response = generate_response(
    processor, model, 
    "What do you see in this image?", 
    image=image
)
```

## Architecture

### Core Functions

1. **`setup_model(model_id)`**
   - Automatically detects model type (Qwen2.5-Omni vs. standard models)
   - Uses `Qwen2_5OmniProcessor` and `Qwen2_5OmniForConditionalGeneration` for Qwen Omni models
   - Falls back to `AutoProcessor` and `AutoModelForCausalLM` for other models
   - Handles device mapping and optimization
   - Returns processor and model objects

2. **`generate_response(processor, model, text_prompt, image=None)`**
   - Handles both text-only and multimodal inputs
   - Uses chat templates for proper formatting
   - Returns formatted response with metadata

3. **`load_image(image_path)`**
   - Loads and validates image files
   - Converts to RGB format if needed
   - Returns PIL Image object

### Device Strategy

The implementation uses intelligent device mapping optimized for each platform:

- **Apple M3 (your system)**: 
  - Automatically detects MPS capability and system RAM
  - Uses MPS GPU for models up to 3B parameters with 16GB RAM
  - Uses MPS GPU for models up to 7B parameters with 32GB+ RAM
  - Skips incompatible bitsandbytes quantization
  - Falls back to CPU for very large models
- **NVIDIA GPUs**: Uses CUDA with float16 precision and quantization
- **CPU Fallback**: Uses CPU with float32 for maximum compatibility

### Model Support

Currently tested with:
- **Qwen2.5-Omni-7B**: Primary multimodal model
- Compatible with any Hugging Face model that supports AutoProcessor

## File Structure

```
├── run_multimodal.py          # Main multimodal script
├── test_multimodal.py         # Unit tests
├── sample_data/
│   └── sample_image.jpg       # Test image
├── chat_histories/            # Generated chat logs
│   ├── multimodal_text_*.json
│   └── multimodal_vision_*.json
├── requirements.txt           # Python dependencies
└── pyproject.toml            # Project configuration
```

## Dependencies

Key dependencies added for multimodal support:
- `Pillow>=9.0.0,<11.0.0` - Image processing
- `transformers>=4.30.0` - Model loading and inference
- `torch>=2.0.0` - Neural network framework

## Testing

Run the test suite to verify functionality:
```bash
python test_multimodal.py
```

Tests cover:
- Device mapping logic
- Content extraction and cleaning
- Response formatting
- Image loading functionality
- Loading indicator system

## Performance Considerations

### Memory Management
- Large models (7B+) automatically use CPU on macOS to avoid MPS memory limitations
- Uses `low_cpu_mem_usage=True` for efficient loading
- Implements proper tensor device mapping

### Generation Parameters
- `temperature=0.7`: Balanced creativity vs. consistency  
- `repetition_penalty=1.1`: Reduces repetitive outputs
- `max_new_tokens=512`: Reasonable response length limit

## Troubleshooting

### Common Issues

1. **"Invalid buffer size" or Memory Errors**
   - **Quick Solution**: Use force CPU mode:
     ```bash
     python run_multimodal.py --force_cpu --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "test"
     ```
   - **Alternative**: Use smaller model:
     ```bash
     python run_multimodal.py --model_id "Qwen/Qwen2.5-1.5B-Instruct" --prompt "test"
     ```
   - The script automatically detects memory issues and implements fallback strategies
   - Quantization is enabled by default to reduce memory usage

2. **"Unrecognized configuration class" Error with Qwen2.5-Omni**
   - **Solution**: Update transformers to version 4.52.0 or later:
     ```bash
     pip install transformers>=4.52.0
     # OR for latest features:
     pip install git+https://github.com/huggingface/transformers
     ```
   - The script automatically detects Qwen2.5-Omni models and uses specific classes
   - Falls back to standard models if Qwen Omni classes are unavailable

2. **MPS Memory Errors on macOS**
   - Automatically handled by fallback to CPU for large models
   - Reduce `max_new_tokens` if needed

3. **Image Loading Errors**
   - Ensure image file exists and is readable
   - Supported formats: JPEG, PNG, GIF, BMP
   - Images are automatically converted to RGB

4. **Model Loading Issues**
   - Check internet connection for model download
   - Ensure sufficient disk space (~15GB for 7B models)
   - Verify Hugging Face Hub access

### Performance Tips

- Use smaller models for faster inference (e.g., 1.3B instead of 7B)
- Enable quantization for memory-constrained environments
- Use CPU for very large models on Apple Silicon

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_id` | str | `Qwen/Qwen2.5-Omni-7B` | Hugging Face model ID to use |
| `--prompt` | str | None | Text prompt for the model (required unless using `--demo`) |
| `--image_path` | str | None | Path to image file for multimodal input |
| `--max_new_tokens` | int | 512 | Maximum number of new tokens to generate |
| `--demo` | flag | False | Run demo mode with example prompts |
| `--save_history` | flag | True | Save chat history to JSON file |
| `--quantization` | flag | True | Use quantization for memory efficiency |
| `--force_cpu` | flag | False | Force CPU usage even if GPU is available |

## Future Enhancements

Planned improvements:
- Support for more model architectures
- Batch processing capabilities
- Fine-tuning integration
- Advanced quantization options
- Video input support
- Audio input support for Qwen Omni models