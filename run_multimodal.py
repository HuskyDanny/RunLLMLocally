import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import sys
import time
import threading
from datetime import datetime
import json
import argparse
import os
from PIL import Image
from typing import Optional, Tuple, Dict, Any


def show_loading(operation, message="Loading"):
    """Show a loading spinner while operation runs - reused from deepseek implementation"""
    def spinner():
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not done.is_set():
            sys.stdout.write(f"\r{chars[i % len(chars)]} {message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write(f"\r✓ {message} complete!\n")
        sys.stdout.flush()

    done = threading.Event()
    spinner_thread = threading.Thread(target=spinner)
    spinner_thread.start()
    
    try:
        result = operation()
        return result
    finally:
        done.set()
        spinner_thread.join()


def setup_model(model_id: str) -> Tuple[Any, Any]:
    """Setup multi-modal model and processor with loading indicators"""
    print(f"🚀 Initializing Multi-Modal Model: {model_id}")
    print("=" * 60)
    
    # Load processor with loading indicator
    def load_processor():
        return AutoProcessor.from_pretrained(model_id)
    
    processor = show_loading(load_processor, "Loading processor")
    
    # Load model with loading indicator
    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    
    model = show_loading(load_model, "Loading model")
    
    print("✅ Model and processor loaded successfully!")
    print(f"📱 Device: {model.device}")
    print(f"🔧 Multi-modal capability: {is_multimodal_model(processor)}")
    print("=" * 60)
    
    return processor, model


def load_image(image_path: str) -> Image.Image:
    """Load image from path and return PIL Image object"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path)


def is_multimodal_model(processor) -> bool:
    """Detect if the processor supports multi-modal (vision) capabilities"""
    return hasattr(processor, 'image_processor')


def prepare_inputs(processor, text: str, image: Optional[Image.Image] = None) -> Dict[str, torch.Tensor]:
    """Prepare inputs for both text-only and multi-modal models"""
    if image is not None:
        # Multi-modal input
        inputs = processor(text=text, images=image, return_tensors="pt")
    else:
        # Text-only input
        inputs = processor(text=text, return_tensors="pt")
    
    return inputs


def format_multimodal_response(raw_response: str, prompt: str, model_id: str, input_type: str) -> Dict[str, Any]:
    """Format multi-modal response in a consistent structure"""
    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": raw_response.strip(),
        "model": model_id,
        "input_type": input_type
    }


def generate_response(model_id: str, text: str, image_path: Optional[str] = None, max_length: int = 512) -> Dict[str, Any]:
    """Generate response with support for both text-only and multi-modal input"""
    print(f"\n💭 Prompt: {text}")
    if image_path:
        print(f"🖼️  Image: {image_path}")
    print("-" * 40)
    
    # Load image if provided
    image = None
    if image_path:
        def load_img():
            return load_image(image_path)
        image = show_loading(load_img, "Loading image")
    
    # Setup model and processor
    def setup():
        return setup_model(model_id)
    processor, model = show_loading(setup, "Setting up model")
    
    # Determine input type
    input_type = "multimodal" if image is not None else "text_only"
    
    # Check if model supports multi-modal and we have an image
    if image and not is_multimodal_model(processor):
        print("⚠️  Warning: Image provided but model doesn't support multi-modal input. Using text-only mode.")
        image = None
        input_type = "text_only"
    
    # Prepare inputs
    inputs = prepare_inputs(processor, text, image)
    
    # Move inputs to model device (with error handling for testing)
    try:
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
    except (TypeError, AttributeError):
        # In testing or if device is not available, keep inputs as is
        inputs = inputs
    
    # Generate response
    def generate():
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the response if it's repeated
        if response.startswith(text):
            response = response[len(text):].strip()
        
        return response
    
    raw_response = show_loading(generate, "Generating response")
    formatted_response = format_multimodal_response(raw_response, text, model_id, input_type)
    
    return formatted_response


def print_response(formatted_response: Dict[str, Any]):
    """Print response in a clean, readable format"""
    print("\n📝 Generated Response:")
    print("=" * 60)
    print(formatted_response["response"])
    print("=" * 60)
    print(f"⏰ Generated at: {formatted_response['timestamp']}")
    print(f"🤖 Model: {formatted_response['model']}")
    print(f"🔧 Input type: {formatted_response['input_type']}")


def parse_arguments(args_list=None):
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multi-modal model runner")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--image", help="Path to image file (optional)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Omni-7B", help="Model ID")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum generation length")
    
    if args_list is not None:
        return parser.parse_args(args_list)
    return parser.parse_args()


def main():
    """Main function with support for both text-only and multi-modal input"""
    try:
        args = parse_arguments()
        
        # Generate response
        response = generate_response(
            model_id=args.model,
            text=args.prompt,
            image_path=args.image,
            max_length=args.max_length
        )
        
        # Print response
        print_response(response)
        
        # Save to file for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_type = response["input_type"]
        filename = f"chat_histories/multimodal_response_{input_type}_{timestamp}.json"
        
        try:
            os.makedirs("chat_histories", exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(response, f, indent=2)
            print(f"💾 Response saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Could not save to file: {e}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()