#!/usr/bin/env python3
"""
Model Download Script

Downloads specific models for local use, based on run_multimodal.py logic.
Supports downloading models with different quantization configurations and 
provides detailed progress information.

Usage:
    python download_model.py --model_id microsoft/kosmos-2-patch14-224
    python download_model.py --model_id Qwen/Qwen2.5-Omni-7B --quantization int4
    python download_model.py --list-popular
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from transformers.utils import cached_file
from huggingface_hub import scan_cache_dir
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
except ImportError:
    Qwen2_5OmniForConditionalGeneration = None
    Qwen2_5OmniProcessor = None

import sys
import time
import threading
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
        config_path = cached_file(model_id, "config.json", cache_dir=None, local_files_only=True)
        return config_path is not None
    except Exception:
        return False


def get_cache_info():
    """Get information about cached models"""
    try:
        cache_info = scan_cache_dir()
        models = []
        total_size = 0
        
        for repo in cache_info.repos:
            repo_size = sum(revision.size_on_disk for revision in repo.revisions)
            models.append({
                "id": repo.repo_id,
                "size_gb": repo_size / (1024**3),
                "path": repo.repo_path
            })
            total_size += repo_size
        
        return models, total_size / (1024**3)
    except Exception as e:
        print(f"⚠️  Could not scan cache: {e}")
        return [], 0


def list_popular_models():
    """List popular models with their characteristics"""
    models = [
        {
            "id": "microsoft/kosmos-2-patch14-224",
            "type": "Multi-modal (Vision + Text)",
            "size": "~1.3GB",
            "description": "Small, fast vision-language model"
        },
        {
            "id": "Salesforce/blip2-opt-2.7b", 
            "type": "Multi-modal (Vision + Text)",
            "size": "~5.4GB",
            "description": "High-quality image captioning and VQA"
        },
        {
            "id": "Qwen/Qwen2.5-Omni-7B",
            "type": "Multi-modal (Vision + Text + Audio)",
            "size": "~15GB",
            "description": "Advanced multi-modal model with audio support"
        },
        {
            "id": "gpt2",
            "type": "Text-only",
            "size": "~500MB", 
            "description": "Classic text generation model"
        },
        {
            "id": "microsoft/DialoGPT-medium",
            "type": "Text-only (Conversational)",
            "size": "~850MB",
            "description": "Conversational AI model"
        },
        {
            "id": "deepseek-ai/deepseek-coder-1.3b-base",
            "type": "Text-only (Code)",
            "size": "~2.6GB",
            "description": "Code generation model"
        }
    ]
    
    print("🔥 Popular Models Available for Download:")
    print("=" * 80)
    
    for model in models:
        print(f"📦 {model['id']}")
        print(f"   Type: {model['type']}")
        print(f"   Size: {model['size']}")
        print(f"   Description: {model['description']}")
        print()


def download_model(model_id, quantization="int8", force_download=False):
    """Download a specific model with the given configuration"""
    print(f"🚀 Downloading Model: {model_id}")
    print(f"⚙️  Quantization: {quantization}")
    print("=" * 60)
    
    # Check if already cached
    is_cached = check_model_cached(model_id) and not force_download
    if is_cached and not force_download:
        print("✅ Model already cached locally")
        return True
    
    if force_download:
        print("🔄 Force download requested - will re-download model")
    else:
        print("📥 Starting model download...")
    
    try:
        # Configure quantization for size estimation
        quantization_config = None
        if quantization == "int8":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                print("🔧 Configured for 8-bit quantization (50% memory reduction)")
            except ImportError:
                print("⚠️  bitsandbytes not available, downloading in full precision")
                quantization = "none"
        elif quantization in ["int4", "nf4"]:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("🔧 Configured for 4-bit quantization (75% memory reduction)")
            except ImportError:
                print("⚠️  bitsandbytes not available, downloading in full precision")
                quantization = "none"
        
        if quantization == "none":
            print("🔧 Full precision download")
        
        # Download processor/tokenizer first
        def download_processor():
            if "Qwen2.5-Omni" in model_id and Qwen2_5OmniProcessor:
                return Qwen2_5OmniProcessor.from_pretrained(model_id)
            try:
                return AutoProcessor.from_pretrained(model_id)
            except Exception:
                return AutoTokenizer.from_pretrained(model_id)
        
        processor = show_loading(download_processor, "Downloading processor/tokenizer")
        
        # Download model
        def download_model_weights():
            load_params = {
                "torch_dtype": torch.float16,
                "device_map": "cpu",  # Use CPU for download to avoid device issues
                "trust_remote_code": True,
                "resume_download": True,
                "low_cpu_mem_usage": True
            }
            
            # Only add quantization config if we have bitsandbytes and it's not causing issues
            if quantization_config:
                try:
                    load_params["quantization_config"] = quantization_config
                except Exception as e:
                    print(f"⚠️  Quantization config failed, using full precision: {e}")
                    load_params.pop("quantization_config", None)
            
            # Special handling for Qwen2.5-Omni
            if "Qwen2.5-Omni" in model_id and Qwen2_5OmniForConditionalGeneration:
                return Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **load_params)
            
            # Try multi-modal first, then text-only
            try:
                return AutoModelForVision2Seq.from_pretrained(model_id, **load_params)
            except (ValueError, OSError):
                return AutoModelForCausalLM.from_pretrained(model_id, **load_params)
        
        show_loading(download_model_weights, "Downloading model weights")
        
        # Determine model type
        is_multimodal = hasattr(processor, 'image_processor')
        
        print("✅ Model downloaded successfully!")
        print(f"🔍 Type: {'Multi-modal' if is_multimodal else 'Text-only'}")
        print("💾 Cached at: ~/.cache/huggingface/hub/")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main():
    """Main function with CLI support"""
    parser = argparse.ArgumentParser(description="Download specific models for local use")
    parser.add_argument("--model_id", 
                       help="Hugging Face model ID to download")
    parser.add_argument("--quantization", default="int8",
                       choices=["none", "int8", "int4", "nf4"],
                       help="Quantization level for download (default: int8)")
    parser.add_argument("--force_download", action="store_true",
                       help="Force re-download even if cached")
    parser.add_argument("--list-popular", action="store_true",
                       help="List popular models available for download")
    parser.add_argument("--list-cached", action="store_true", 
                       help="List currently cached models")
    parser.add_argument("--batch-download", nargs="+",
                       help="Download multiple models in batch")
    
    args = parser.parse_args()
    
    try:
        # List popular models
        if args.list_popular:
            list_popular_models()
            return
        
        # List cached models
        if args.list_cached:
            models, total_size = get_cache_info()
            if not models:
                print("📭 No models currently cached")
            else:
                print(f"💾 Cached Models (Total: {total_size:.1f} GB):")
                print("=" * 60)
                for model in models:
                    print(f"📦 {model['id']} ({model['size_gb']:.1f} GB)")
            return
        
        # Batch download
        if args.batch_download:
            print(f"📦 Batch downloading {len(args.batch_download)} models...")
            success_count = 0
            for model_id in args.batch_download:
                print(f"\n🔄 Processing {model_id}...")
                if download_model(model_id, args.quantization, args.force_download):
                    success_count += 1
                print()
            
            print(f"✅ Successfully downloaded {success_count}/{len(args.batch_download)} models")
            return
        
        # Single model download
        if not args.model_id:
            print("❌ Error: --model_id is required")
            print("💡 Try: python download_model.py --list-popular")
            sys.exit(1)
        
        success = download_model(args.model_id, args.quantization, args.force_download)
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()