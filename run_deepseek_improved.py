import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import time
import threading
from datetime import datetime
import json
import re


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


def setup_deepseek():
    """Setup DeepSeek model and tokenizer with loading indicators"""
    model_id = "deepseek-ai/deepseek-coder-1.3b-base"
    
    print(f"🚀 Initializing DeepSeek Coder Model: {model_id}")
    print("=" * 60)
    
    # Load tokenizer with loading indicator
    def load_tokenizer():
        return AutoTokenizer.from_pretrained(model_id)
    
    tokenizer = show_loading(load_tokenizer, "Loading tokenizer")
    
    # Load model with loading indicator
    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    model = show_loading(load_model, "Loading model")
    
    print("✅ Model and tokenizer loaded successfully!")
    print(f"📱 Device: {model.device}")
    print(f"💾 Model size: ~1.3B parameters")
    print("=" * 60)
    
    return tokenizer, model


def extract_generated_content(full_response, original_prompt):
    """Extract just the generated content, removing prompt repetition"""
    # Remove the original prompt from the beginning if it's repeated
    if full_response.startswith(original_prompt):
        content = full_response[len(original_prompt):].strip()
    else:
        content = full_response.strip()
    
    # Split by newlines and take until we see another unrelated prompt
    lines = content.split('\n')
    clean_lines = []
    
    for line in lines:
        # Stop if we encounter what looks like another prompt
        if line.strip().startswith("Write a Python function") and len(clean_lines) > 0:
            break
        clean_lines.append(line)
    
    return '\n'.join(clean_lines).strip()


def format_response(raw_response, prompt):
    """Format response in a consistent structure"""
    clean_content = extract_generated_content(raw_response, prompt)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": clean_content,
        "model": "deepseek-coder-1.3b-base"
    }


def generate_response(tokenizer, model, prompt, max_length=256):
    """Generate response with loading indicator and consistent formatting"""
    print(f"\n💭 Prompt: {prompt}")
    print("-" * 40)
    
    def generate():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.3,  # Lower temperature for more focused output
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Reduce repetition
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    raw_response = show_loading(generate, "Generating response")
    formatted_response = format_response(raw_response, prompt)
    
    return formatted_response


def print_response(formatted_response):
    """Print response in a clean, readable format"""
    print("\n📝 Generated Response:")
    print("=" * 60)
    print(formatted_response["response"])
    print("=" * 60)
    print(f"⏰ Generated at: {formatted_response['timestamp']}")
    print(f"🤖 Model: {formatted_response['model']}")


def main():
    """Main function with better error handling"""
    try:
        # Setup model and tokenizer
        tokenizer, model = setup_deepseek()
        
        # Example usage
        prompt = "Write a Python function to calculate fibonacci sequence"
        response = generate_response(tokenizer, model, prompt)
        print_response(response)
        
        # Save to file for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_histories/deepseek_response_{timestamp}.json"
        
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