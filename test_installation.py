#!/usr/bin/env python3
"""
Installation Test Script for Freshdesk Ticket Analyzer
Run this script to verify all dependencies are installed correctly.

Usage: python test_installation.py
"""

import sys
import importlib

def test_import(module_name, package_name=None, description=""):
    """Test if a module can be imported successfully"""
    try:
        if package_name:
            module = importlib.import_module(module_name)
            # Test specific attribute if provided
            if hasattr(module, package_name):
                getattr(module, package_name)
        else:
            importlib.import_module(module_name)
        
        print(f"✅ {module_name:<25} - OK {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name:<25} - FAILED: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name:<25} - WARNING: {e}")
        return False

def test_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    print(f"\n🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False

def test_optional_features():
    """Test optional/advanced features"""
    print("\n🔬 Testing Optional Features:")
    
    # Test GPU support
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA GPU support available")
        else:
            print("ℹ️  CUDA GPU not available (CPU mode will be used)")
    except ImportError:
        print("ℹ️  PyTorch not installed (not required)")
    
    # Test advanced FAISS features
    try:
        import faiss
        if hasattr(faiss, 'StandardGpuResources'):
            print("ℹ️  FAISS GPU support detected")
        else:
            print("ℹ️  FAISS CPU version (recommended for most users)")
    except:
        pass

def main():
    print("🧪 Freshdesk Ticket Analyzer - Installation Test")
    print("=" * 55)
    
    # Test Python version
    python_ok = test_python_version()
    
    print("\n📦 Testing Core Dependencies:")
    
    # Core dependencies
    tests = [
        ("streamlit", None, "- Web interface"),
        ("requests", None, "- HTTP requests"),
        ("pandas", None, "- Data processing"),
        ("numpy", None, "- Numerical operations"),
    ]
    
    print("\n🤖 Testing AI/ML Dependencies:")
    ai_tests = [
        ("google.generativeai", None, "- Gemini AI"),
        ("faiss", None, "- Vector database"),
        ("sentence_transformers", "SentenceTransformer", "- Text embeddings"),
        ("transformers", None, "- Transformer models"),
    ]
    
    print("\n👁️ Testing OCR Dependencies:")
    ocr_tests = [
        ("paddleocr", "PaddleOCR", "- OCR engine"),
        ("PIL", "Image", "- Image processing"),
    ]
    
    print("\n📄 Testing Document Processing:")
    doc_tests = [
        ("PyPDF2", None, "- PDF processing"),
        ("docx", None, "- Word documents"),
    ]
    
    # Run all tests
    all_tests = tests + ai_tests + ocr_tests + doc_tests
    passed = 0
    total = len(all_tests)
    
    for module, attr, desc in all_tests:
        if test_import(module, attr, desc):
            passed += 1
    
    # Test optional features
    test_optional_features()
    
    # Summary
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}/{total} dependencies")
    
    if passed == total:
        print("🎉 All core dependencies installed successfully!")
        print("🚀 You're ready to run the Freshdesk Ticket Analyzer!")
        
        print(f"\n💡 Next steps:")
        print(f"1. Configure your API keys in .streamlit/secrets.toml")
        print(f"2. Run: streamlit run freshdesk_analyzer.py")
        
    else:
        print(f"⚠️  {total - passed} dependencies missing.")
        print(f"📋 Run: pip install -r requirements.txt")
        
        if not python_ok:
            print(f"🔧 Please upgrade to Python 3.8 or higher")
    
    print(f"\n🔗 For detailed setup instructions, see the README file.")

if __name__ == "__main__":
    main()