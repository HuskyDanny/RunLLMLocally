import unittest
from unittest.mock import patch, MagicMock
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import time


class TestDeepSeekImproved(unittest.TestCase):
    
    def test_setup_shows_loading_indicator(self):
        """Test that setup_deepseek shows loading progress"""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('run_deepseek_improved.AutoTokenizer') as mock_tokenizer:
                with patch('run_deepseek_improved.AutoModelForCausalLM') as mock_model:
                    # Mock the loading to be instant for testing
                    mock_tokenizer.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    
                    from run_deepseek_improved import setup_deepseek
                    setup_deepseek()
                    
                    output = mock_stdout.getvalue()
                    self.assertIn("Loading", output)
                    self.assertIn("✓", output)  # Success indicator
    
    def test_generate_response_has_consistent_format(self):
        """Test that generate_response returns consistent, clean format"""
        from run_deepseek_improved import format_response
        
        # Test with a sample model response
        raw_response = "Write a Python function to calculate fibonacci sequence def fibonacci(n): ..."
        formatted = format_response(raw_response, "Write a Python function to calculate fibonacci sequence")
        
        self.assertIsInstance(formatted, dict)
        self.assertIn("prompt", formatted)
        self.assertIn("response", formatted)
        self.assertIn("timestamp", formatted)
    
    def test_response_extraction_removes_prompt_repetition(self):
        """Test that we properly extract just the generated code"""
        from run_deepseek_improved import extract_generated_content
        
        prompt = "Write a Python function to calculate fibonacci sequence"
        full_response = "Write a Python function to calculate fibonacci sequence\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        
        extracted = extract_generated_content(full_response, prompt)
        self.assertNotIn(prompt, extracted)
        self.assertIn("def fibonacci", extracted)
    
    def test_loading_spinner_functionality(self):
        """Test that loading spinner shows progress"""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            from run_deepseek_improved import show_loading
            
            # Mock a quick operation
            def quick_operation():
                time.sleep(0.1)
                return "done"
            
            result = show_loading(quick_operation, "Testing")
            output = mock_stdout.getvalue()
            
            self.assertEqual(result, "done")
            self.assertIn("Testing", output)


if __name__ == "__main__":
    unittest.main() 