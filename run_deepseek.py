import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_deepseek():
    # Model ID from Hugging Face
    model_id = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load model with lower precision to reduce memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Setup model and tokenizer
    tokenizer, model = setup_deepseek()
    
    # Example usage
    prompt = "Write a Python function to calculate fibonacci sequence"
    response = generate_response(tokenizer, model, prompt)
    print("Response:", response)