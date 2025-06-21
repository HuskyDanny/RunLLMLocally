# Technical Specification: General-Purpose Hugging Face Model Runner

**Document Status:** `Draft`
**Author:** `AI Assistant`
**Date:** `2023-10-27`

## 1. Introduction & Goal

This document outlines the technical requirements for evolving the `run-llm-locally` project into a general-purpose model runner. The goal is to support a wide variety of models from the Hugging Face Hub, including both text-only and multi-modal (text and image) models.

The primary objective is to refactor the codebase to create a flexible architecture that can dynamically load and run different models based on user input, with the initial multi-modal implementation targeting **`Qwen/Qwen2.5-Omni-7B`**.

## 2. Core Architectural Changes

The fundamental shift is from a single-model script to a flexible, multi-modal framework.

1.  **From Tokenizer to Processor:** For multi-modal models, `transformers.AutoTokenizer` is insufficient. The architecture will use `transformers.AutoProcessor`, which combines a text **Tokenizer** and an **Image Processor**. This component is essential for correctly formatting both text and image data for the model. For text-only models, the standard `AutoTokenizer` will still be used.

2.  **Dynamic Input Handling:** The application must be able to handle either text-only input or mixed text-and-image input based on the selected model's capabilities.

3.  **Model-Agnostic Generation:** The core logic will rely on the `AutoModelForCausalLM` class and the model's associated processor to handle model-specific requirements, such as prompt templates, automatically.

## 3. Implementation Plan

### Phase 1: Project Setup & Dependencies

-   **Modify Dependencies:**
    -   Add `Pillow` to `requirements.txt` and `pyproject.toml` for image processing.
-   **Add Sample Data:**
    -   Create a `sample_data/` directory.
    -   Add a sample image (e.g., `sample_image.jpg`) for testing.

### Phase 2: Refactoring Core Logic for Multi-Modality

This phase focuses on adapting the existing functions to handle multi-modal inputs, using **`Qwen/Qwen2.5-Omni-7B`** as the first implementation target.

-   **`setup_model()` function:**
    -   **Model Class:** Use the general `AutoModelForCausalLM` class, which can load various architectures, including Qwen-Omni.
    -   **Processor Loading:** Use `AutoProcessor.from_pretrained(model_id)`. The processor will automatically load the correct tokenizer and image processor for the given model.
    -   **Return Value:** The function will return a `processor` and a `model`.

-   **`generate_response()` function:**
    -   **Signature Change:** The signature will be updated to `generate_response(processor, model, text_prompt, image=None)`. The `image` will be an optional argument.
    -   **Prompt Templating (General Approach):** Instead of hard-coding a prompt string, the logic will leverage the processor's built-in chat template. This is a robust method that works across different models.
        ```python
        # For multi-modal input
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": text_prompt}
            ]}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(model.device)
        ```
    -   The `model.generate(**inputs, ...)` call will work on the tensors prepared by the processor.

### Phase 3: Updating the Main Execution Flow

-   **`main()` function:**
    -   **Image Loading:** Add logic to load an image if its path is provided.
    -   **Orchestration:** The main function will coordinate the flow:
        1.  Call `setup_model()` to get the processor and model.
        2.  Load an image file (if applicable).
        3.  Define the text prompt.
        4.  Call `generate_response()` with the appropriate inputs.
        5.  Print the final response.

## 4. Future-Proofing and Extensibility (Post-MVP)

To create a truly general-purpose framework, the following steps are critical:

1.  **Command-Line Interface (CLI):**
    -   Integrate `argparse` to allow users to specify:
        -   `--model_id`: **Required.** The Hugging Face ID of any model to use.
        -   `--image_path`: **Optional.** The path to an input image.
        -   `--prompt`: The text prompt.

2.  **Object-Oriented Abstraction (The Handler Pattern):**
    -   Define a base `ModelHandler` abstract class with `load()` and `generate()` methods.
    -   Create concrete implementations for different model *capabilities*:
        -   `TextModelHandler(ModelHandler)`: Handles models that only take text.
        -   `VisionLanguageModelHandler(ModelHandler)`: Handles models taking text and images.
    -   The main script will inspect the loaded model's configuration (`model.config`) to determine its capabilities and instantiate the correct handler. This provides a clean, scalable architecture for supporting any model from the Hugging Face Hub without needing to write model-specific code. 