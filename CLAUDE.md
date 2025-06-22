# CLAUDE.md

This file provides guidance for developing and working with code in this repository.

## 1. Project Overview

This is a Python project for running various open-source Large Language Models (LLMs) locally, with a focus on DeepSeek Coder and multimodal models like Qwen. It leverages the Hugging Face `transformers` library for model loading, quantization, and inference.

The project includes several scripts:
- `run_multimodal.py`: The primary script for running multimodal models.
- `run_deepseek_improved.py`: An enhanced script for running text-based models.
- `download_model.py`: A utility for managing model downloads and caches.
- `test_deepseek.py`: Unit tests for the implementation.

## 2. Local Environment

This project is developed and tested on macOS but is designed to be cross-platform.

### Environment Specification
- **Operating System**: macOS Sequoia 15.1.1 on Apple M3 (arm64)
- **Kernel**: Darwin 24.1.0
- **Python**: Python 3.13 managed with `conda`.

### Setup & Installation
It is highly recommended to use a `conda` environment.

```bash
# Create and activate a conda environment
conda create -n run_llm_locally python=3.13
conda activate run_llm_locally

# Install base dependencies
pip install -r requirements.txt

# Install development dependencies (for code quality tools)
pip install -e ".[dev]"

# Install test dependencies
pip install -e ".[test]"
```

## 3. Selection of Technology

- **Core Framework**: Python 3.13
- **ML & AI**:
    - `torch`: For tensor computation and neural networks.
    - `transformers`: For model loading and inference from the Hugging Face Hub.
    - `accelerate`: For hardware acceleration and automatic device mapping.
- **Image Processing**: `Pillow` is used for handling image inputs in multimodal models.
- **Environment Management**: `conda` is used for managing Python environments.
- **Code Quality**:
    - `black` for code formatting.
    - `isort` for import sorting.
    - `flake8` for linting.
    - `mypy` for static type checking.
- **Testing**: `pytest`.

## 4. Architecture

- **Modular Scripts**: The project is organized into distinct Python scripts, each with a clear purpose (e.g., running models, downloading models).
- **Model Setup (`setup_model`)**: This core function handles the dynamic loading of models and their corresponding processors/tokenizers. It supports quantization, smart device mapping (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback), and displays loading indicators for a better user experience.
- **Response Generation (`generate_response`)**: This function manages the inference process, handling both text-only and multimodal (text + image) inputs. It also maintains conversation history and formats the model's output.
- **Model Caching**: Models are automatically downloaded and cached by Hugging Face Hub in `~/.cache/huggingface/hub/` to avoid re-downloading.

## 5. Coding Practices

- **Asynchronous Programming**: [In Python, `threading` is suitable for I/O-bound tasks that use blocking libraries, while `asyncio` is preferred for high-throughput I/O with non-blocking libraries. Use `asyncio` where possible to maximize efficiency, but use `threading` as a practical choice when integrating with synchronous code.][[memory:4552483989072794866]]
- **Code Formatting**: All code must be formatted with `black` and `isort` before committing.
- **Linting & Type Safety**: Code should be free of `flake8` errors and include type hints that pass `mypy` checks.
- **Modularity**: Functions should be small, single-purpose, and well-documented.
- **Dependency Documentation**: **ALWAYS** utilize `mcp` tools, specifically the `context7` server, to search for and reference the latest official documentation for project dependencies. This must be the **FIRST STEP** when encountering any library-specific errors or implementing new features with external dependencies. Research thoroughly before implementing solutions.

## 6. Key Features

- **Multimodal Support**: Capable of processing combined text and image inputs with models like Qwen2.5-Omni.
- **Advanced Quantization**: Supports `int8` and `int4` quantization to run large models on systems with limited VRAM/RAM.
- **Performance Optimization**:
    - Automatic device placement (MPS, CUDA, CPU).
    - `torch.float16` precision is used for memory efficiency.
    - On macOS, models larger than 3 billion parameters automatically use the CPU to avoid Metal GPU memory limitations.
- **User Experience**:
    - Provides loading spinners during time-consuming operations.
    - Cleans and formats model outputs to remove conversational artifacts.
    - Saves chat history to JSON files within the `chat_histories/` directory.
- **Model Management**: Includes a dedicated script (`download_model.py`) to list popular models and manage the local cache.