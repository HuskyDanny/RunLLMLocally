# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for running DeepSeek LLM models locally using Hugging Face Transformers. The project consists of:

- `run_deepseek.py` - Basic DeepSeek model implementation
- `run_deepseek_improved.py` - Enhanced version with loading indicators, better formatting, and error handling 
- `demo_loading.py` - Standalone demo of the loading indicator functionality
- `test_deepseek.py` - Unit tests for the improved implementation

## Development Commands

### Setup & Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies  
pip install -e ".[dev]"

# Install test dependencies
pip install -e ".[test]"
```

### Running the Application
```bash
# Run the improved DeepSeek implementation
python run_deepseek_improved.py

# Run the basic implementation
python run_deepseek.py

# Run the loading demo (no model download required)
python demo_loading.py

# Run via console script (after installation)
run-deepseek
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
- **Model Setup**: `setup_deepseek()` handles tokenizer and model initialization with loading indicators
- **Response Generation**: `generate_response()` handles inference with configurable parameters
- **Loading System**: `show_loading()` provides threaded loading spinners for long operations
- **Response Formatting**: Clean extraction and formatting of model outputs, removing prompt repetition

### Key Features
- Uses DeepSeek Coder 1.3B model (`deepseek-ai/deepseek-coder-1.3b-base`)
- FP16 precision for memory efficiency
- Automatic device mapping
- Response history saving to JSON files in `chat_histories/` directory
- Clean response extraction that removes prompt repetition
- Loading indicators for better UX during model initialization and inference

### Model Configuration
- Temperature: 0.3 (focused output)
- Max length: 256 tokens
- Repetition penalty: 1.1
- Uses `torch.float16` for memory optimization