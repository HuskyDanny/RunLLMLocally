import torch
from transformers import AutoProcessor, AutoModelForCausalLM
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    QWEN_OMNI_AVAILABLE = True
except ImportError:
    QWEN_OMNI_AVAILABLE = False
import sys
import time
import threading
from datetime import datetime
import json
from PIL import Image
import os
import platform
import argparse


def show_loading(operation, message="Loading"):
    """Show a loading spinner while operation runs"""
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


def get_mps_memory_info():
    """Get Apple Silicon GPU memory information"""
    try:
        # Try to get MPS memory info (available in newer PyTorch versions)
        if hasattr(torch.backends.mps, 'driver_allocated_memory'):
            allocated = torch.backends.mps.driver_allocated_memory() / (1024**3)
            return allocated
        return None
    except Exception:
        return None


def get_optimal_device_map(model_id="", force_cpu=False, force_mps=False):
    """Determine the best device mapping strategy based on the platform and model size"""
    if force_cpu:
        print("🖥️  Forcing CPU usage as requested")
        return "cpu", torch.float32
    
    if force_mps:
        if torch.backends.mps.is_available():
            print("🍎 Forcing MPS usage as requested (experimental)")
            return "mps", torch.float16
        else:
            print("❌ MPS not available - falling back to CPU")
            return "cpu", torch.float32
        
    system = platform.system()
    
    # Estimate model size from ID
    is_large_model = any(size in model_id for size in ["7B", "8B", "9B", "10B", "13B", "70B"])
    is_medium_model = any(size in model_id for size in ["3B", "4B", "5B", "6B"])
    is_small_model = any(size in model_id for size in ["1B", "1.5B", "2B"]) or (not is_large_model and not is_medium_model)
    
    # Multimodal models require more memory than text-only models
    is_multimodal = "Omni" in model_id or "VL" in model_id or "Vision" in model_id
    
    if system == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            print("🍎 Apple Silicon detected with MPS support")
            
            # Get system memory info
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"📊 System RAM: {total_memory_gb:.1f} GB")
            
            if is_multimodal:
                print("🔍 Multimodal model detected - requires more memory")
                # Be more conservative with multimodal models
                if is_small_model and total_memory_gb >= 16:
                    print("🚀 Small multimodal model - using MPS with careful memory management")
                    return "mps", torch.float16
                elif is_medium_model and total_memory_gb >= 24:
                    print("🚀 Medium multimodal model with ample RAM - using MPS")
                    return "mps", torch.float16
                else:
                    print("⚠️  Multimodal model - using CPU for memory stability")
                    return "cpu", torch.float32
            else:
                # Regular text-only models
                if is_small_model:
                    print("🚀 Small model - using MPS for optimal performance")
                    return "mps", torch.float16
                elif is_medium_model:
                    if total_memory_gb >= 16:
                        print("🚀 Medium model with sufficient RAM - using MPS")
                        return "mps", torch.float16
                    else:
                        print("⚠️  Medium model with limited RAM - using CPU for stability")
                        return "cpu", torch.float32
                elif is_large_model:
                    if total_memory_gb >= 32:
                        print("🚀 Large model with sufficient RAM - using MPS")
                        return "mps", torch.float16
                    else:
                        print("⚠️  Large model with limited RAM - using CPU")
                        return "cpu", torch.float32
                else:
                    # Unknown model size, be conservative
                    print("🔍 Unknown model size - using CPU for safety")
                    return "cpu", torch.float32
        else:
            print("❌ MPS not available - using CPU")
            return "cpu", torch.float32
    elif torch.cuda.is_available():
        print("🟢 NVIDIA CUDA detected")
        # Check CUDA memory
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
            if is_large_model and gpu_memory < 12:
                print("⚠️  Large model needs more GPU memory than available - using CPU")
                return "cpu", torch.float32
            return "cuda", torch.float16
        except (RuntimeError, AttributeError):
            return "cuda", torch.float16
    else:
        print("🖥️  No GPU acceleration available - using CPU")
        return "cpu", torch.float32


def setup_model(model_id="Qwen/Qwen2.5-Omni-7B", use_quantization=True, force_cpu=False, force_mps=False):
    """Setup multimodal model and processor with loading indicators"""
    print(f"🚀 Initializing Multimodal Model: {model_id}")
    print("=" * 60)
    
    device, dtype = get_optimal_device_map(model_id, force_cpu, force_mps)
    print(f"🖥️  Target device: {device}")
    print(f"📊 Data type: {dtype}")
    
    # Determine quantization strategy
    quantization_config = None
    if use_quantization:
        # Skip quantization on Apple Silicon since bitsandbytes doesn't work there
        if device == "mps":
            print("🍎 Skipping quantization on Apple Silicon (bitsandbytes not compatible)")
            print("   💡 MPS will provide GPU acceleration instead")
            quantization_config = None
        else:
            print("🔧 Checking quantization options...")
            try:
                from transformers import BitsAndBytesConfig
                import importlib.util
                
                # Check if bitsandbytes is available
                if importlib.util.find_spec("bitsandbytes") is None:
                    print("   ⚠️  bitsandbytes not installed")
                    print("   💡 Install with: pip install bitsandbytes>=0.41.0")
                    quantization_config = None
                else:
                    # Check if bitsandbytes version supports the features we need
                    try:
                        # Test for 8-bit quantization support
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                        )
                        print("   ✅ 8-bit quantization enabled")
                    except Exception as e:
                        if "latest version" in str(e).lower() or "bitsandbytes" in str(e).lower():
                            print(f"   ⚠️  bitsandbytes version issue: {e}")
                            print("   🔄 Trying 4-bit quantization as fallback...")
                            try:
                                quantization_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.float16,
                                    bnb_4bit_use_double_quant=True,
                                )
                                print("   ✅ 4-bit quantization enabled")
                            except Exception as e2:
                                print(f"   ⚠️  4-bit quantization also failed: {e2}")
                                print("   🚫 Loading without quantization")
                                quantization_config = None
                        else:
                            print(f"   ⚠️  Quantization config error: {e}")
                            quantization_config = None
                        
            except ImportError as e:
                print(f"   ⚠️  Quantization libraries not available: {e}")
                print("   💡 Install transformers with: pip install transformers>=4.52.0")
                quantization_config = None
    
    # Determine if this is a Qwen Omni model
    is_qwen_omni = "Qwen2.5-Omni" in model_id
    
    if is_qwen_omni and QWEN_OMNI_AVAILABLE:
        print("🔧 Using Qwen2.5-Omni specific classes")
        # Load processor with loading indicator
        def load_processor():
            return Qwen2_5OmniProcessor.from_pretrained(model_id)
        
        processor = show_loading(load_processor, "Loading Qwen Omni processor")
        
        # Load model with loading indicator
        def load_model():
            model_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            
            # Set device mapping based on target device
            if device == "cpu":
                model_kwargs["device_map"] = None
                model_kwargs["torch_dtype"] = torch.float32
            elif device == "mps":
                # For MPS, let accelerate handle device mapping automatically
                model_kwargs["device_map"] = "auto"
                print("🍎 Using MPS device mapping for Apple Silicon GPU")
            else:  # CUDA
                model_kwargs["device_map"] = "auto"
            
            # Add quantization if available
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            try:
                return Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_id, **model_kwargs
                )
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["buffer size", "memory", "bitsandbytes", "quantization"]):
                    if "bitsandbytes" in error_msg:
                        print("⚠️  Quantization error detected - trying without quantization...")
                    else:
                        print("⚠️  Memory error detected - trying with reduced precision...")
                    
                    # Fallback to CPU with float32 and no quantization
                    model_kwargs.update({
                        "device_map": None,
                        "torch_dtype": torch.float32,
                        "quantization_config": None
                    })
                    return Qwen2_5OmniForConditionalGeneration.from_pretrained(
                        model_id, **model_kwargs
                    )
                else:
                    raise
        
        model = show_loading(load_model, "Loading Qwen Omni model")
        
    elif is_qwen_omni and not QWEN_OMNI_AVAILABLE:
        print("❌ Qwen2.5-Omni model detected but Qwen Omni classes not available!")
        print("   Please install the latest transformers version:")
        print("   pip install transformers>=4.52.3")
        print("   or pip install git+https://github.com/huggingface/transformers")
        raise ImportError("Qwen2.5-Omni classes not available in current transformers version")
        
    else:
        print("🔧 Using standard AutoModel classes")
        # Load processor with loading indicator
        def load_processor():
            return AutoProcessor.from_pretrained(model_id)
        
        processor = show_loading(load_processor, "Loading processor")
        
        # Load model with loading indicator
        def load_model():
            model_kwargs = {
                "torch_dtype": dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            
            # Set device mapping based on target device
            if device == "cpu":
                model_kwargs["device_map"] = None
                model_kwargs["torch_dtype"] = torch.float32
            elif device == "mps":
                # For MPS, let accelerate handle device mapping automatically
                model_kwargs["device_map"] = "auto"
                print("🍎 Using MPS device mapping for Apple Silicon GPU")
            else:  # CUDA
                model_kwargs["device_map"] = "auto"
            
            # Add quantization if available
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            try:
                return AutoModelForCausalLM.from_pretrained(
                    model_id, **model_kwargs
                )
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["buffer size", "memory", "bitsandbytes", "quantization"]):
                    if "bitsandbytes" in error_msg:
                        print("⚠️  Quantization error detected - trying without quantization...")
                    else:
                        print("⚠️  Memory error detected - trying with reduced precision...")
                    
                    # Fallback to CPU with float32 and no quantization
                    model_kwargs.update({
                        "device_map": None,
                        "torch_dtype": torch.float32,
                        "quantization_config": None
                    })
                    return AutoModelForCausalLM.from_pretrained(
                        model_id, **model_kwargs
                    )
                else:
                    raise
        
        model = show_loading(load_model, "Loading model")
    
    print("✅ Model and processor loaded successfully!")
    print(f"📱 Device: {model.device}")
    print(f"💾 Model: {model_id}")
    print("=" * 60)
    
    return processor, model


def extract_generated_content(full_response, original_prompt):
    """Extract just the generated content, removing prompt repetition"""
    if full_response.startswith(original_prompt):
        content = full_response[len(original_prompt):].strip()
    else:
        content = full_response.strip()
    
    # Clean up common artifacts
    lines = content.split('\n')
    clean_lines = []
    
    for line in lines:
        # Skip obvious repetitions or artifacts
        if line.strip() and not line.strip().startswith("User:") and not line.strip().startswith("Assistant:"):
            clean_lines.append(line)
    
    return '\n'.join(clean_lines).strip()


def format_response(raw_response, prompt, model_id, has_image=False):
    """Format response in a consistent structure"""
    clean_content = extract_generated_content(raw_response, prompt)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": clean_content,
        "model": model_id,
        "multimodal": has_image
    }


def generate_response(processor, model, text_prompt, image=None, max_length=512):
    """Generate response with text and optional image input"""
    print(f"\n💭 Prompt: {text_prompt}")
    if image:
        print("🖼️  Processing with image input")
    print("-" * 40)
    
    # Check if this is a Qwen Omni model
    is_qwen_omni = hasattr(model, 'config') and hasattr(model.config, 'model_type') and 'qwen2_5_omni' in str(model.config.model_type)
    
    def generate():
        if is_qwen_omni:
            # Use Qwen Omni specific format
            if image:
                conversations = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text_prompt}
                    ]}
                ]
            else:
                conversations = [
                    {"role": "user", "content": [
                        {"type": "text", "text": text_prompt}
                    ]}
                ]
            
            # Apply chat template for Qwen Omni
            inputs = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            # Use standard format for other models
            if image:
                # Multi-modal input (text + image)
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_prompt}
                    ]}
                ]
                
                # Apply chat template for multi-modal input
                prompt = processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Process both text and image
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(model.device)
            else:
                # Text-only input
                messages = [
                    {"role": "user", "content": text_prompt}
                ]
                
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = processor(
                    text=prompt,
                    return_tensors="pt"
                ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            if is_qwen_omni:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
        
        # Decode response
        if is_qwen_omni:
            response = processor.decode(outputs[0], skip_special_tokens=True)
        else:
            response = processor.decode(outputs[0], skip_special_tokens=True)
        return response
    
    raw_response = show_loading(generate, "Generating response")
    formatted_response = format_response(
        raw_response, 
        text_prompt, 
        model.config.name_or_path if hasattr(model.config, 'name_or_path') else "multimodal-model",
        has_image=(image is not None)
    )
    
    return formatted_response


def print_response(formatted_response):
    """Print response in a clean, readable format"""
    print("\n📝 Generated Response:")
    print("=" * 60)
    print(formatted_response["response"])
    print("=" * 60)
    print(f"⏰ Generated at: {formatted_response['timestamp']}")
    print(f"🤖 Model: {formatted_response['model']}")
    if formatted_response["multimodal"]:
        print("🖼️  Multimodal: Yes")


def load_image(image_path):
    """Load and validate image file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Could not load image: {e}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run multimodal LLM models with text and image inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Text-only generation
  python run_multimodal.py --prompt "Explain quantum computing"
  
  # Multimodal generation with image
  python run_multimodal.py --model_id "Qwen/Qwen2.5-Omni-3B" --prompt "Describe this image" --image_path sample_data/sample_image.jpg
  
  # Custom parameters
  python run_multimodal.py --model_id "Qwen/Qwen2.5-Omni-7B" --prompt "What's in this image?" --image_path my_image.jpg --max_new_tokens 300
        """
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="Hugging Face model ID to use (default: Qwen/Qwen2.5-Omni-7B)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for the model (required if not using demo mode)"
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to image file for multimodal input (optional)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate (default: 512)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with example prompts (default if no prompt provided)"
    )
    
    parser.add_argument(
        "--save_history",
        action="store_true",
        default=True,
        help="Save chat history to JSON file (default: True)"
    )
    
    parser.add_argument(
        "--quantization",
        action="store_true",
        default=True,
        help="Use quantization for memory efficiency (default: True)"
    )
    
    parser.add_argument(
        "--no-quantization",
        dest="quantization",
        action="store_false",
        help="Disable quantization (useful for compatibility issues)"
    )
    
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    parser.add_argument(
        "--force_mps",
        action="store_true",
        help="Force MPS usage on Apple Silicon (experimental)"
    )
    
    return parser.parse_args()


def main():
    """Main function with CLI support and multimodal capabilities"""
    args = parse_arguments()
    
    # Determine if we should run in demo mode
    if not args.prompt and not args.demo:
        print("🚀 No prompt provided - running in demo mode")
        args.demo = True
    
    try:
        # Setup model and processor
        model_id = args.model_id
        
        # Try to load the model, with fallback options
        try:
            processor, model = setup_model(model_id, use_quantization=args.quantization, force_cpu=args.force_cpu, force_mps=args.force_mps)
        except (ImportError, OSError) as e:
            if "Qwen2.5-Omni" in str(e) or "AutoModel" in str(e):
                print(f"⚠️  Failed to load {model_id}: {e}")
                print("🔄 Trying fallback model...")
                # Fallback to a more widely supported model
                model_id = "Qwen/Qwen2.5-7B-Instruct"
                processor, model = setup_model(model_id, use_quantization=args.quantization, force_cpu=args.force_cpu, force_mps=args.force_mps)
            else:
                raise
        
        if args.demo:
            # Demo mode - run examples
            print("\n" + "="*60)
            print("🔤 DEMO MODE - TEXT-ONLY EXAMPLE")
            print("="*60)
            
            demo_text_prompt = "Write a Python function to calculate the area of a circle."
            response = generate_response(processor, model, demo_text_prompt, max_length=args.max_new_tokens)
            print_response(response)
            
            # Save text-only response if requested
            if args.save_history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_histories/multimodal_text_{timestamp}.json"
                os.makedirs("chat_histories", exist_ok=True)
                
                with open(filename, 'w') as f:
                    json.dump(response, f, indent=2)
                print(f"💾 Response saved to: {filename}")
            
            # Demo multimodal example (if image exists)
            sample_image_path = "sample_data/sample_image.jpg"
            if os.path.exists(sample_image_path):
                print("\n" + "="*60)
                print("🖼️  DEMO MODE - MULTIMODAL EXAMPLE")
                print("="*60)
                
                image = load_image(sample_image_path)
                demo_multimodal_prompt = "Describe what you see in this image in detail."
                
                multimodal_response = generate_response(
                    processor, model, demo_multimodal_prompt, image=image, max_length=args.max_new_tokens
                )
                print_response(multimodal_response)
                
                # Save multimodal response if requested
                if args.save_history:
                    mm_filename = f"chat_histories/multimodal_vision_{timestamp}.json"
                    with open(mm_filename, 'w') as f:
                        json.dump(multimodal_response, f, indent=2)
                    print(f"💾 Multimodal response saved to: {mm_filename}")
            else:
                print(f"\n⚠️  Sample image not found at {sample_image_path}")
                print("   Skipping multimodal demo.")
        
        else:
            # CLI mode - use provided arguments
            print("\n" + "="*60)
            if args.image_path:
                print("🖼️  MULTIMODAL GENERATION")
                print(f"📄 Prompt: {args.prompt}")
                print(f"🖼️  Image: {args.image_path}")
            else:
                print("🔤 TEXT-ONLY GENERATION")
                print(f"📄 Prompt: {args.prompt}")
            print("="*60)
            
            # Load image if provided
            image = None
            if args.image_path:
                try:
                    image = load_image(args.image_path)
                    print(f"✅ Image loaded: {args.image_path}")
                except Exception as e:
                    print(f"❌ Failed to load image: {e}")
                    sys.exit(1)
            
            # Generate response
            response = generate_response(
                processor, model, args.prompt, image=image, max_length=args.max_new_tokens
            )
            print_response(response)
            
            # Save response if requested
            if args.save_history:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                mode = "multimodal" if image else "text"
                filename = f"chat_histories/{mode}_cli_{timestamp}.json"
                os.makedirs("chat_histories", exist_ok=True)
                
                with open(filename, 'w') as f:
                    json.dump(response, f, indent=2)
                print(f"💾 Response saved to: {filename}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()