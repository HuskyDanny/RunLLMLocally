# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for running LLM models locally using Hugging Face Transformers, with support for both text-only and multi-modal (text + image) models. The project consists of:

- `run_deepseek.py` - Basic DeepSeek model implementation
- `run_deepseek_improved.py` - Enhanced version with loading indicators, better formatting, and error handling 
- `run_multimodal.py` - **ENHANCED**: Multi-modal model runner with quantization, caching, and platform optimization
- `download_model.py` - **NEW**: Dedicated model download script with batch support and cache management
- `demo_loading.py` - Standalone demo of the loading indicator functionality
- `test_deepseek.py` - Unit tests for the improved implementation
- `test_multimodal_demo.py` - **NEW**: Demo script for testing multi-modal functionality
- `create_sample_image.py` - **NEW**: Script to create sample images for testing
- `sample_data/` - **NEW**: Directory containing sample images for testing

## Development Commands

### Setup & Installation
```bash
# Install dependencies (includes Pillow, psutil, and bitsandbytes for quantization)
pip install -r requirements.txt

# Install development dependencies  
pip install -e ".[dev]"

# Install test dependencies
pip install -e ".[test]"

# For quantization support (optional but recommended)
pip install bitsandbytes>=0.41.0
```

### Running the Application

#### Text-Only Models
```bash
# Run the improved DeepSeek implementation
python run_deepseek_improved.py

# Run the basic implementation
python run_deepseek.py

# Run via console script (after installation)
run-deepseek
```

#### Multi-Modal Models
```bash
# Run with text-only model (backward compatible)
python run_multimodal.py --model_id deepseek-ai/deepseek-coder-1.3b-base --prompt "Write a Python function"

# Run with multi-modal model and image (default int8 quantization)
python run_multimodal.py --model_id Qwen/Qwen2.5-Omni-7B --image_path sample_data/sample_image.jpg --prompt "Describe this image"

# Run with different quantization levels for memory efficiency
python run_multimodal.py --model_id microsoft/kosmos-2-patch14-224 --quantization int4 --image_path sample_data/sample_image.jpg --prompt "What's in this image?"

# Run without quantization (full precision)
python run_multimodal.py --model_id gpt2 --quantization none --prompt "Hello world"

# Force CPU usage (useful on macOS to avoid Metal GPU limits)
python run_multimodal.py --model_id Qwen/Qwen2.5-Omni-3B --image_path sample_data/sample_image.jpg --prompt "Describe this image"

# Clear cache and re-download model
python run_multimodal.py --model_id MODEL_ID --clear_cache --force_download

# Run via console script (after installation)
run-multimodal --model_id MODEL_ID --image_path IMAGE_PATH --prompt "Your prompt"
```

#### Model Download & Management
```bash
# List popular models available for download
python download_model.py --list-popular

# Download specific model with quantization
python download_model.py --model_id microsoft/kosmos-2-patch14-224 --quantization int8

# Batch download multiple models
python download_model.py --batch-download gpt2 microsoft/kosmos-2-patch14-224 deepseek-ai/deepseek-coder-1.3b-base

# Check cached models and disk usage
python download_model.py --list-cached

# Clear cache and re-download
python download_model.py --model_id MODEL_ID --clear_cache --force_download
```

#### Demo & Testing
```bash
# Run the loading demo (no model download required)
python demo_loading.py

# Test multi-modal functionality
python test_multimodal_demo.py

# Create sample images for testing
python create_sample_image.py
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run specific test file
python -m unittest test_deepseek.py
```

### Code Quality
```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8

# Type checking
mypy .
```

## Architecture Notes

### Core Components
- **Model Setup**: `setup_model()` handles both AutoProcessor (multi-modal) and AutoTokenizer (text-only) initialization
- **Response Generation**: `generate_response()` handles both text-only and multi-modal inference
- **Loading System**: `show_loading()` provides threaded loading spinners for long operations
- **Image Processing**: PIL-based image loading and preprocessing for multi-modal models
- **Response Formatting**: Clean extraction and formatting of model outputs

### Multi-Modal Features
- **Dynamic Model Detection**: Automatically detects if a model supports multi-modal input
- **Flexible Input Handling**: Supports text-only or text+image inputs based on model capabilities
- **Chat Template Integration**: Uses model-specific chat templates for proper prompt formatting
- **Image Preprocessing**: Automatic image conversion to RGB format for consistency
- **CLI Interface**: Command-line arguments for model selection, image input, and prompt customization
- **Quantization Support**: Int8, Int4, and NF4 quantization for memory efficiency
- **Platform Optimization**: Automatic CPU fallback on macOS to avoid Metal GPU limits
- **Model Caching**: Smart caching system to avoid re-downloads

### Supported Models
- **Text-Only**: DeepSeek Coder, GPT-2, DialoGPT, any `AutoModelForCausalLM` compatible model
- **Multi-Modal**: Qwen2.5-Omni (3B/7B), Microsoft Kosmos-2, Salesforce BLIP2, Vision-Language models with `AutoProcessor` support

### Key Features
- Uses `AutoProcessor` for multi-modal models, falls back to `AutoTokenizer` for text-only
- **Quantization Support**: Int8 (default), Int4, NF4, or no quantization for memory optimization
- **Smart Device Mapping**: Automatic CPU fallback on macOS, RAM-based device selection
- **Model Caching**: Intelligent cache detection and management to avoid re-downloads
- **Platform Optimization**: Handles Metal GPU limitations on macOS automatically
- **Robust Error Handling**: Multiple fallback methods for model loading and response decoding
- Response history saving to JSON files in `chat_histories/` directory
- Clean response extraction that removes prompt repetition
- Loading indicators for better UX during model initialization and inference
- Sample image generation for testing purposes

### Model Configuration
- **Temperature**: 0.3 (focused output)
- **Max length**: 256 tokens (configurable)
- **Repetition penalty**: 1.1
- **Data types**: `torch.float16` for memory optimization, quantized options available
- **Device mapping**: Automatic with platform-specific optimizations
- **Trust remote code**: Enabled for custom model architectures (Qwen2.5-Omni, etc.)
- **Quantization options**: 
  - `int8` (default): 50% memory reduction
  - `int4`/`nf4`: 75% memory reduction  
  - `none`: Full precision

## Project Setup
- Remember if you want to run application, you need to be in run_llm_locally conda env and add dependencies in the env
- **Required dependencies**: Pillow (image processing), psutil (memory monitoring), bitsandbytes (quantization)
- **macOS users**: Models >3B automatically use CPU to avoid Metal GPU limitations
- **Memory requirements**: Quantization significantly reduces RAM needs (int8: ~50% less, int4: ~75% less)

## Model Management
- **Cache location**: `~/.cache/huggingface/hub/`
- **Download once**: Models are cached automatically for future use
- **Popular models**: Use `python download_model.py --list-popular` to see recommended models
- **Disk usage**: Check with `python download_model.py --list-cached`