import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import io
from contextlib import redirect_stdout
import os
from PIL import Image
import torch


class TestMultiModalModel(unittest.TestCase):
    """Test cases for multi-modal model functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_image_path = "sample_data/sample_image.jpg"
        self.sample_text = "Describe what you see in this image"
        self.test_model_id = "Qwen/Qwen2.5-Omni-7B"
        
    def test_setup_model_returns_processor_and_model(self):
        """Test that setup_model returns both processor and model objects"""
        with patch('run_multimodal.AutoProcessor') as mock_processor:
            with patch('run_multimodal.AutoModelForCausalLM') as mock_model:
                mock_processor.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()
                
                from run_multimodal import setup_model
                processor, model = setup_model(self.test_model_id)
                
                self.assertIsNotNone(processor)
                self.assertIsNotNone(model)
                mock_processor.from_pretrained.assert_called_once_with(self.test_model_id)
                mock_model.from_pretrained.assert_called_once()
    
    def test_load_image_returns_pil_image(self):
        """Test that load_image returns a PIL Image object"""
        from run_multimodal import load_image
        
        # Test with existing sample image
        if os.path.exists(self.sample_image_path):
            image = load_image(self.sample_image_path)
            self.assertIsInstance(image, Image.Image)
        else:
            # Test with mock if sample doesn't exist
            with patch('PIL.Image.open') as mock_open:
                mock_image = MagicMock(spec=Image.Image)
                mock_open.return_value = mock_image
                
                image = load_image("test_path.jpg")
                self.assertEqual(image, mock_image)
    
    def test_load_image_handles_invalid_path(self):
        """Test that load_image handles invalid image paths gracefully"""
        from run_multimodal import load_image
        
        with self.assertRaises(FileNotFoundError):
            load_image("nonexistent_image.jpg")
    
    def test_is_multimodal_model_detection(self):
        """Test detection of multi-modal vs text-only models"""
        from run_multimodal import is_multimodal_model
        
        # Mock processor with image processing capability
        mock_processor = MagicMock()
        mock_processor.image_processor = MagicMock()
        self.assertTrue(is_multimodal_model(mock_processor))
        
        # Mock processor without image processing capability
        mock_processor_text_only = MagicMock()
        del mock_processor_text_only.image_processor
        self.assertFalse(is_multimodal_model(mock_processor_text_only))
    
    def test_prepare_inputs_text_only(self):
        """Test input preparation for text-only models"""
        from run_multimodal import prepare_inputs
        
        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        
        inputs = prepare_inputs(mock_processor, self.sample_text, None)
        
        self.assertIn("input_ids", inputs)
        mock_processor.assert_called_once_with(text=self.sample_text, return_tensors="pt")
    
    def test_prepare_inputs_multimodal(self):
        """Test input preparation for multi-modal models with text and image"""
        from run_multimodal import prepare_inputs
        
        mock_processor = MagicMock()
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.tensor([[[[0.5]]]])
        }
        mock_image = MagicMock(spec=Image.Image)
        
        inputs = prepare_inputs(mock_processor, self.sample_text, mock_image)
        
        self.assertIn("input_ids", inputs)
        self.assertIn("pixel_values", inputs)
        mock_processor.assert_called_once_with(text=self.sample_text, images=mock_image, return_tensors="pt")
    
    def test_generate_response_text_only(self):
        """Test response generation for text-only input"""
        with patch('run_multimodal.load_image') as mock_load_image:
            with patch('run_multimodal.setup_model') as mock_setup:
                with patch('run_multimodal.prepare_inputs') as mock_prepare:
                    with patch('run_multimodal.is_multimodal_model') as mock_is_multimodal:
                        
                        # Setup mocks
                        mock_processor = MagicMock()
                        mock_model = MagicMock()
                        mock_setup.return_value = (mock_processor, mock_model)
                        mock_is_multimodal.return_value = False
                        mock_prepare.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
                        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                        mock_processor.decode.return_value = "Generated response"
                        
                        from run_multimodal import generate_response
                        response = generate_response(self.test_model_id, self.sample_text, None)
                        
                        self.assertIsInstance(response, dict)
                        self.assertIn("response", response)
                        self.assertIn("timestamp", response)
                        self.assertIn("model", response)
                        self.assertEqual(response["model"], self.test_model_id)
    
    def test_generate_response_multimodal(self):
        """Test response generation for multi-modal input"""
        with patch('run_multimodal.load_image') as mock_load_image:
            with patch('run_multimodal.setup_model') as mock_setup:
                with patch('run_multimodal.prepare_inputs') as mock_prepare:
                    with patch('run_multimodal.is_multimodal_model') as mock_is_multimodal:
                        
                        # Setup mocks
                        mock_processor = MagicMock()
                        mock_model = MagicMock()
                        mock_image = MagicMock(spec=Image.Image)
                        mock_setup.return_value = (mock_processor, mock_model)
                        mock_load_image.return_value = mock_image
                        mock_is_multimodal.return_value = True
                        mock_prepare.return_value = {
                            "input_ids": torch.tensor([[1, 2, 3]]),
                            "pixel_values": torch.tensor([[[[0.5]]]])
                        }
                        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                        mock_processor.decode.return_value = "I can see shapes and colors in this image"
                        
                        from run_multimodal import generate_response
                        response = generate_response(self.test_model_id, self.sample_text, self.sample_image_path)
                        
                        self.assertIsInstance(response, dict)
                        self.assertIn("response", response)
                        self.assertIn("timestamp", response)
                        self.assertIn("model", response)
                        self.assertIn("input_type", response)
                        self.assertEqual(response["input_type"], "multimodal")
    
    def test_generate_response_shows_loading_indicator(self):
        """Test that generate_response shows loading indicators"""
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch('run_multimodal.setup_model') as mock_setup:
                with patch('run_multimodal.prepare_inputs') as mock_prepare:
                    with patch('run_multimodal.is_multimodal_model') as mock_is_multimodal:
                        
                        # Setup mocks for quick execution
                        mock_processor = MagicMock()
                        mock_model = MagicMock()
                        mock_setup.return_value = (mock_processor, mock_model)
                        mock_is_multimodal.return_value = False
                        mock_prepare.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
                        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                        mock_processor.decode.return_value = "Test response"
                        
                        from run_multimodal import generate_response
                        generate_response(self.test_model_id, self.sample_text, None)
                        
                        output = mock_stdout.getvalue()
                        # Check for any of the loading messages used in the implementation
                        self.assertTrue(any(msg in output for msg in ["Loading", "Setting up model", "Generating response"]))
                        self.assertIn("✓", output)
    
    def test_format_multimodal_response(self):
        """Test formatting of multi-modal responses"""
        from run_multimodal import format_multimodal_response
        
        raw_response = "I can see a colorful image with geometric shapes"
        prompt = "Describe what you see in this image"
        model_id = self.test_model_id
        input_type = "multimodal"
        
        formatted = format_multimodal_response(raw_response, prompt, model_id, input_type)
        
        self.assertIsInstance(formatted, dict)
        self.assertIn("timestamp", formatted)
        self.assertIn("prompt", formatted)
        self.assertIn("response", formatted)
        self.assertIn("model", formatted)
        self.assertIn("input_type", formatted)
        self.assertEqual(formatted["input_type"], input_type)
        self.assertEqual(formatted["model"], model_id)
    
    def test_error_handling_for_unsupported_model(self):
        """Test error handling for unsupported model types"""
        with patch('run_multimodal.AutoProcessor') as mock_processor:
            with patch('run_multimodal.AutoModelForCausalLM') as mock_model:
                # Simulate model loading failure
                mock_processor.from_pretrained.side_effect = Exception("Model not found")
                
                from run_multimodal import setup_model
                
                with self.assertRaises(Exception):
                    setup_model("invalid/model-id")
    
    def test_command_line_argument_parsing(self):
        """Test command line argument parsing for image input"""
        from run_multimodal import parse_arguments
        
        # Test text-only mode
        args = parse_arguments(["--prompt", "Test prompt"])
        self.assertEqual(args.prompt, "Test prompt")
        self.assertIsNone(args.image)
        
        # Test multi-modal mode
        args = parse_arguments(["--prompt", "Test prompt", "--image", "test.jpg"])
        self.assertEqual(args.prompt, "Test prompt")
        self.assertEqual(args.image, "test.jpg")


if __name__ == "__main__":
    unittest.main()