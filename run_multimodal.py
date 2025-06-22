import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from transformers.utils import cached_file
from huggingface_hub import snapshot_download, HfApi
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
except ImportError:
    Qwen2_5OmniForConditionalGeneration = None
    Qwen2_5OmniProcessor = None
from PIL import Image
import sys
import time
import threading
from datetime import datetime
import json
import os
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


def check_model_cached(model_id):
    """Check if model is already cached locally"""
    try:
        # Try to get config file to check if model is cached
        config_path = cached_file(model_id, "config.json", cache_dir=None, local_files_only=True)
        return config_path is not None
    except Exception:
        return False

def setup_model(model_id="Qwen/Qwen2.5-Omni-7B", force_download=False, quantization="int8"):
    """Setup model and processor with loading indicators"""
    print(f"🚀 Initializing Multi-Modal Model: {model_id}")
    print(f"⚙️  Quantization: {quantization}")
    print("=" * 60)
    
    # Check if model is cached
    is_cached = check_model_cached(model_id) and not force_download
    if is_cached:
        print("✅ Model found in cache - loading from local files")
    else:
        if force_download:
            print("🔄 Force download requested - will re-download model")
        else:
            print("📥 Model not cached - will download (this may take a while)")
        print("💡 Tip: Model will be cached after download for faster future loading")
    
    # Try to load processor first (for multi-modal models)
    def load_processor():
        # Special handling for Qwen2.5-Omni
        if "Qwen2.5-Omni" in model_id and Qwen2_5OmniProcessor:
            return Qwen2_5OmniProcessor.from_pretrained(model_id)
        try:
            return AutoProcessor.from_pretrained(model_id)
        except Exception:
            # Fallback to tokenizer for text-only models
            return AutoTokenizer.from_pretrained(model_id)
    
    processor = show_loading(load_processor, "Loading processor/tokenizer")
    
    # Load model with loading indicator
    def load_model():
        # Check available memory and adjust device mapping
        try:
            import psutil
            import platform
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            print(f"💾 Available RAM: {available_ram_gb:.1f} GB")
            
            # Force CPU for large models on macOS due to Metal limitations
            is_macos = platform.system() == "Darwin"
            is_large_model = any(size in model_id.upper() for size in ["3B", "7B", "13B", "OMNI"])
            
            if is_macos and is_large_model:
                device_map = "cpu"
                print("⚠️  Using CPU on macOS to avoid Metal GPU memory limits")
            elif "7B" in model_id.upper() and available_ram_gb < 16:
                device_map = "cpu"
                print("⚠️  Using CPU due to limited RAM")
            else:
                device_map = "auto"
        except ImportError:
            device_map = "auto"
        
        # Configure quantization
        quantization_config = None
        torch_dtype = torch.float16  # Default
        
        if quantization == "int8":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("🔧 Using 8-bit quantization for memory efficiency")
        elif quantization == "int4":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("🔧 Using 4-bit quantization for maximum memory efficiency")
        elif quantization == "nf4":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            print("🔧 Using NF4 quantization for optimal quality/memory balance")
        else:
            print("🔧 No quantization - using full precision")
        
        # Common loading parameters
        load_params = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": True,
            "resume_download": True,        # Resume interrupted downloads
            "low_cpu_mem_usage": True       # Reduce CPU memory usage during loading
        }
        
        # Add quantization config if specified
        if quantization_config:
            load_params["quantization_config"] = quantization_config
        
        # Only use local_files_only if we're confident the model is fully cached
        if is_cached and not force_download:
            print("🔍 Attempting to load from cache...")
            load_params["local_files_only"] = True
        
        # Special handling for Qwen2.5-Omni
        if "Qwen2.5-Omni" in model_id and Qwen2_5OmniForConditionalGeneration:
            try:
                return Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **load_params)
            except Exception as e:
                if "local_files_only" in str(e) or "does not appear to have files" in str(e):
                    print("⚠️  Cache incomplete, downloading missing files...")
                    load_params["local_files_only"] = False
                    return Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **load_params)
                raise e
        
        # Try multi-modal model first, then fall back to causal LM
        try:
            return AutoModelForVision2Seq.from_pretrained(model_id, **load_params)
        except (ValueError, OSError) as e:
            # If local_files_only failed, try downloading
            if "local_files_only" in str(e) or "does not appear to have files" in str(e):
                print("⚠️  Cache incomplete, downloading missing files...")
                load_params["local_files_only"] = False
                try:
                    return AutoModelForVision2Seq.from_pretrained(model_id, **load_params)
                except (ValueError, OSError):
                    pass
            
            # Fallback to causal LM for text-only models
            load_params.pop("local_files_only", None)  # Remove if present
            return AutoModelForCausalLM.from_pretrained(model_id, **load_params)
    
    model = show_loading(load_model, "Loading model")
    
    # Determine if this is a multi-modal model
    is_multimodal = hasattr(processor, 'image_processor')
    
    print("✅ Model and processor loaded successfully!")
    print(f"📱 Device: {model.device}")
    print(f"🔍 Multi-modal support: {'Yes' if is_multimodal else 'No'}")
    print("=" * 60)
    
    return processor, model, is_multimodal


def load_image(image_path):
    """Load and return PIL image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    print(f"📷 Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")
    return image


def extract_generated_content(full_response, original_prompt):
    """Extract just the generated content, removing prompt repetition"""
    if full_response.startswith(original_prompt):
        content = full_response[len(original_prompt):].strip()
    else:
        content = full_response.strip()
    
    lines = content.split('\n')
    clean_lines = []
    
    for line in lines:
        if line.strip().startswith("Write a Python function") and len(clean_lines) > 0:
            break
        clean_lines.append(line)
    
    return '\n'.join(clean_lines).strip()


def format_response(raw_response, prompt, model_id, image_path=None):
    """Format response in a consistent structure"""
    clean_content = extract_generated_content(raw_response, prompt)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "image_path": image_path,
        "response": clean_content,
        "model": model_id
    }


def generate_response(processor, model, text_prompt, image=None, max_length=256, model_id="unknown"):
    """Generate response with loading indicator and consistent formatting"""
    print(f"\n💭 Prompt: {text_prompt}")
    if image:
        print(f"🖼️  With image: {image.size[0]}x{image.size[1]}")
    print("-" * 40)
    
    def generate():
        if image and hasattr(processor, 'image_processor'):
            # Multi-modal generation
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ]}
            ]
            
            prompt = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(model.device)
        else:
            # Text-only generation
            inputs = processor(text_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.3,
            do_sample=True,
            pad_token_id=processor.eos_token_id if hasattr(processor, 'eos_token_id') else processor.tokenizer.eos_token_id,
            eos_token_id=processor.eos_token_id if hasattr(processor, 'eos_token_id') else processor.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # Handle different output formats and decoding methods
        try:
            # For multi-modal models, use processor decode if available
            if hasattr(processor, 'decode'):
                response = processor.decode(outputs[0], skip_special_tokens=True)
            elif hasattr(processor, 'batch_decode'):
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            else:
                # Fallback to tokenizer decode
                response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            # If all else fails, try different approaches
            print(f"⚠️  Decoding issue, trying alternative method: {e}")
            try:
                if hasattr(processor, 'tokenizer'):
                    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # Last resort - convert to list and decode
                    output_ids = outputs[0].tolist() if hasattr(outputs[0], 'tolist') else outputs[0]
                    response = processor.decode(output_ids, skip_special_tokens=True)
            except Exception as e2:
                print(f"❌ Decoding failed: {e2}")
                response = f"[Decoding Error: {e2}]"
        
        return response
    
    raw_response = show_loading(generate, "Generating response")
    formatted_response = format_response(raw_response, text_prompt, model_id, 
                                       image_path=getattr(image, 'filename', None) if image else None)
    
    return formatted_response


def print_response(formatted_response):
    """Print response in a clean, readable format"""
    print("\n📝 Generated Response:")
    print("=" * 60)
    print(formatted_response["response"])
    print("=" * 60)
    print(f"⏰ Generated at: {formatted_response['timestamp']}")
    print(f"🤖 Model: {formatted_response['model']}")
    if formatted_response.get('image_path'):
        print(f"🖼️  Image: {formatted_response['image_path']}")


def clear_model_cache(model_id):
    """Clear cached files for a specific model"""
    from huggingface_hub import scan_cache_dir
    import shutil
    
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if model_id in repo.repo_id:
                print(f"🗑️  Clearing cache for {repo.repo_id}")
                shutil.rmtree(repo.repo_path)
                print(f"✅ Cache cleared for {model_id}")
                return True
    except Exception as e:
        print(f"⚠️  Could not clear cache: {e}")
    return False

def main():
    """Main function with CLI support"""
    parser = argparse.ArgumentParser(description="Run LLM models locally with multi-modal support")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-7B", 
                       help="Hugging Face model ID to use")
    parser.add_argument("--image_path", 
                       help="Path to image file for multi-modal input")
    parser.add_argument("--prompt", default="Describe what you see in this image.", 
                       help="Text prompt to use")
    parser.add_argument("--force_download", action="store_true",
                       help="Force re-download model even if cached")
    parser.add_argument("--clear_cache", action="store_true",
                       help="Clear model cache before loading")
    parser.add_argument("--quantization", default="int8", 
                       choices=["none", "int8", "int4", "nf4"],
                       help="Model quantization level (default: int8)")
    
    args = parser.parse_args()
    
    try:
        # Clear cache if requested
        if args.clear_cache:
            clear_model_cache(args.model_id)
        
        # Setup model and processor
        processor, model, is_multimodal = setup_model(args.model_id, 
                                                     force_download=args.force_download,
                                                     quantization=args.quantization)
        
        # Load image if provided
        image = None
        if args.image_path:
            if not is_multimodal:
                print("❌ Error: Image provided but model doesn't support multi-modal input")
                print("💡 Suggestion: Try a multi-modal model like 'microsoft/kosmos-2-patch14-224' or 'Salesforce/blip2-opt-2.7b'")
                sys.exit(1)
            else:
                image = load_image(args.image_path)
        
        # Generate response
        response = generate_response(processor, model, args.prompt, image, model_id=args.model_id)
        print_response(response)
        
        # Save to file for history
        os.makedirs("chat_histories", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_histories/multimodal_response_{timestamp}.json"
        
        try:
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
        sys.exit(1)


if __name__ == "__main__":
    main()