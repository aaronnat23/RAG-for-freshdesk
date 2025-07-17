#!/usr/bin/env python3
"""
Simple OCR script using PaddleOCR
Usage: python simple_ocr.py <image_file>
"""

import sys
import os
from paddleocr import PaddleOCR
from PIL import Image

def resize_if_large(image_path, max_pixels=5_000_000):
    """Resize image if too large"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            total_pixels = width * height
            
            if total_pixels > max_pixels:
                print(f"Large image detected ({width}x{height}), resizing...")
                scale_factor = (max_pixels / total_pixels) ** 0.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_path = f"resized_{os.path.basename(image_path)}"
                resized_img.save(resized_path, quality=95)
                print(f"Resized and saved as: {resized_path}")
                return resized_path
        
        return image_path
    except Exception as e:
        print(f"Resize failed: {e}")
        return image_path

def extract_text_from_image(image_path):
    """Extract text from image using PaddleOCR"""
    
    # Check for local models first
    det_model_path = "./PP-OCRv5_mobile_det_infer"
    rec_model_path = "./PP-OCRv5_mobile_rec_infer"
    use_local = os.path.exists(det_model_path) and os.path.exists(rec_model_path)
    
    ocr = None
    
    if use_local:
        print("Attempting to use local models...")
        try:
            ocr = PaddleOCR(
                text_detection_model_dir=det_model_path,
                text_recognition_model_dir=rec_model_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device='cpu'  # Force CPU usage
                # Removed lang='en' as it conflicts with local models
            )
            print("✅ Local models loaded successfully!")
        except Exception as e:
            print(f"❌ Local models failed: {e}")
            print("🔄 Falling back to downloading models...")
            ocr = None
    
    # Fallback to downloaded models if local models failed
    if ocr is None:
        print("Using downloadable mobile models...")
        print("💡 Note: Models will be downloaded once and cached for future use")
        try:
            ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device='cpu',  # Force CPU usage
                lang='en'
            )
            print("✅ Downloaded models loaded successfully!")
        except Exception as e:
            print(f"❌ Downloaded models also failed: {e}")
            # Last resort - use default models
            print("🔄 Using default models...")
            ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device='cpu',  # Force CPU usage
                lang='en'
            )
    
    # Perform OCR
    result = ocr.predict(input=image_path)
    
    # Extract text from results
    all_text = []
    for res in result:
        json_data = res.json
        if 'res' in json_data and 'rec_texts' in json_data['res']:
            rec_texts = json_data['res']['rec_texts']
            all_text.extend([text.strip() for text in rec_texts if text.strip()])
    
    return " ".join(all_text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_ocr.py <image_file>")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    try:
        # Resize if needed
        processed_image = resize_if_large(image_path)
        
        # Extract text
        extracted_text = extract_text_from_image(processed_image)
        print("\nExtracted text:")
        print(extracted_text)
        
        # Clean up resized file if created
        if processed_image != image_path:
            os.remove(processed_image)
            print(f"\nCleaned up temporary file: {processed_image}")
            
    except Exception as e:
        print(f"OCR failed: {e}")

if __name__ == "__main__":
    main()