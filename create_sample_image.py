#!/usr/bin/env python3
"""Script to create a sample image for testing multi-modal functionality"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_image():
    """Create a simple sample image with text"""
    # Create a 400x300 image with white background
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some simple shapes and text
    draw.rectangle([50, 50, 350, 250], outline='blue', width=3)
    draw.ellipse([100, 100, 200, 200], fill='lightblue', outline='darkblue')
    draw.ellipse([200, 150, 300, 250], fill='lightgreen', outline='darkgreen')
    
    # Add text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((120, 120), "Sample Image", fill='black', font=font)
    draw.text((120, 140), "for Multi-Modal", fill='black', font=font)
    draw.text((120, 160), "Testing", fill='black', font=font)
    
    # Save the image
    os.makedirs('sample_data', exist_ok=True)
    img.save('sample_data/sample_image.jpg', 'JPEG')
    print("Sample image created at: sample_data/sample_image.jpg")

if __name__ == "__main__":
    create_sample_image()