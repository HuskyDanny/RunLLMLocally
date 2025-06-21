# Multi-Modal Model Runner

This implementation provides a flexible framework for running both text-only and multi-modal (text+image) models from Hugging Face Hub.

## Features

- **Multi-Modal Support**: Handles both text-only and vision-language models
- **AutoProcessor Integration**: Uses AutoProcessor for seamless model handling
- **Dynamic Input Handling**: Automatically detects and processes text and image inputs
- **Loading Indicators**: Beautiful progress indicators for model loading and generation
- **Error Handling**: Robust error handling with graceful fallbacks
- **Command Line Interface**: Easy-to-use CLI for both modes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For development, install additional dependencies:
```bash
pip install pytest
```

## Usage

### Text-Only Mode
```bash
python3 run_multimodal.py --prompt "Write a Python function to calculate fibonacci"
```

### Multi-Modal Mode
```bash
python3 run_multimodal.py --prompt "Describe what you see in this image" --image sample_data/sample_image.jpg
```

### Custom Model
```bash
python3 run_multimodal.py --prompt "Hello, how are you?" --model "microsoft/DialoGPT-medium"
```

### All Options
```bash
python3 run_multimodal.py \
  --prompt "Analyze this image and explain what you see" \
  --image path/to/your/image.jpg \
  --model "Qwen/Qwen2.5-Omni-7B" \
  --max-length 512
```

## Command Line Arguments

- `--prompt` (required): Text prompt for the model
- `--image` (optional): Path to image file for multi-modal input
- `--model` (optional): Model ID from Hugging Face Hub (default: Qwen/Qwen2.5-Omni-7B)
- `--max-length` (optional): Maximum generation length (default: 512)

## Architecture

### Key Components

1. **setup_model()**: Loads processor and model with loading indicators
2. **load_image()**: Handles image loading and validation
3. **is_multimodal_model()**: Detects multi-modal capabilities
4. **prepare_inputs()**: Prepares inputs for both text-only and multi-modal models
5. **generate_response()**: Main generation function with comprehensive error handling
6. **format_multimodal_response()**: Formats responses with metadata

### Model Compatibility

The implementation automatically detects whether a model supports multi-modal input:
- **Multi-modal models**: Use both text and image inputs
- **Text-only models**: Fall back to text-only mode even if image is provided

### Supported Models

- **Qwen/Qwen2.5-Omni-7B** (default): Multi-modal model
- **Any AutoProcessor-compatible model**: Text-only or multi-modal
- **Custom models**: As long as they support AutoProcessor and AutoModelForCausalLM

## Testing

Run the comprehensive test suite:

```bash
python3 test_multimodal.py
```

The tests cover:
- Model setup and loading
- Image processing
- Multi-modal detection
- Input preparation for both modes
- Response generation
- Error handling
- Command line argument parsing

## File Structure

```
├── run_multimodal.py          # Main multi-modal implementation
├── test_multimodal.py         # Comprehensive test suite
├── sample_data/               # Test images
│   └── sample_image.jpg       # Generated test image
├── create_sample_image.py     # Script to generate test images
├── requirements.txt           # Dependencies
├── pyproject.toml            # Project configuration
└── README_MULTIMODAL.md      # This documentation
```

## Response Format

The system returns structured responses:

```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "prompt": "Your input prompt",
  "response": "Generated response text",
  "model": "Qwen/Qwen2.5-Omni-7B",
  "input_type": "multimodal"  // or "text_only"
}
```

## Error Handling

- **Missing image files**: Graceful error with helpful message
- **Unsupported models**: Clear error message with suggestions
- **Memory issues**: Automatic fallback and cleanup
- **Keyboard interruption**: Clean shutdown

## Performance Notes

- **First run**: Model downloading and loading may take time
- **Subsequent runs**: Models are cached by Hugging Face
- **Memory usage**: Large models require significant GPU/CPU memory
- **Device optimization**: Automatic device selection (GPU if available)

## Development

The implementation follows Test-Driven Development (TDD):
1. Tests are written first
2. Implementation follows to make tests pass
3. All 12 test cases pass successfully

## History Tracking

Generated responses are automatically saved to `chat_histories/` with timestamps and input type indicators for easy reference and analysis.