# 🎯 Freshdesk Ticket Analyzer - Setup & Running Instructions

## 📋 Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **Git** (to clone the repository)
- **Freshdesk API access** (Admin or Agent with API permissions)
- **Google AI Studio API key** (for Gemini AI)

## 🚀 Quick Start

### Step 1: Clone/Download the Code
```bash
# If using git
git clone <your-repository-url>
cd freshdesk-ticket-analyzer

# Or download and extract the files to a folder
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues, try updating pip first:
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Create a `.streamlit` folder and `secrets.toml` file:

```bash
# Create the .streamlit directory
mkdir .streamlit

# Create secrets.toml file (use your preferred text editor)
# On Windows:
notepad .streamlit\secrets.toml
# On macOS/Linux:
nano .streamlit/secrets.toml
```

Add your API credentials to `.streamlit/secrets.toml`:

```toml
# Freshdesk Configuration
FRESHDESK_DOMAIN = "your-company.freshdesk.com"
FRESHDESK_API_KEY = "your_freshdesk_api_key_here"

# Google AI (Gemini) Configuration  
GEMINI_KEY = "your_gemini_api_key_here"
```

### Step 5: Run the Application
```bash
streamlit run freshdesk_analyzer.py
```

The application will open in your browser at `http://localhost:8501`

## 🔑 Getting API Keys

### Freshdesk API Key
1. Log into your Freshdesk account
2. Go to **Profile Settings** → **View contact details**
3. Find your **API Key** on the right side
4. Copy the API key (it should look like: `aBcDeFgHiJkLmNoPqRsTuVwXyZ`)

### Google AI Studio API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** in the top navigation
4. Create a new project or select existing one
5. Generate API key and copy it

## 📁 Project Structure
```
freshdesk-ticket-analyzer/
├── freshdesk_analyzer.py          # Main Streamlit application
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── secrets.toml              # API keys and configuration
├── ocr_extracted_text.log        # Generated when processing tickets
├── ticket_vector_db.index        # Generated vector database files
└── ticket_vector_db_metadata.pkl # Generated metadata
```

## 🛠️ Troubleshooting

### Common Installation Issues

#### Issue: `paddlepaddle` installation fails
```bash
# Try installing CPU version specifically
pip install paddlepaddle==2.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# Or for specific Python version
pip install paddlepaddle==2.5.1 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

#### Issue: `faiss` installation fails
```bash
# Try installing specific version
pip install faiss-cpu==1.7.4

# On some systems, you might need:
conda install -c conda-forge faiss-cpu
```

#### Issue: `torch` dependency conflicts
```bash
# Install specific compatible versions
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### Common Runtime Issues

#### Issue: "Failed to create vector database"
- **Solution**: Check that FAISS and sentence-transformers are properly installed
- Try: `pip install --upgrade sentence-transformers faiss-cpu`

#### Issue: "OCR model loading failed"
- **Solution**: PaddleOCR will download models on first run (can take 5-10 minutes)
- Ensure stable internet connection
- Check available disk space (models are ~100MB)

#### Issue: "Error fetching tickets"
- **Solution**: Verify Freshdesk API key and domain in secrets.toml
- Check Freshdesk API rate limits (may need to wait)
- Ensure account has API access permissions

#### Issue: "Gemini API error"
- **Solution**: Verify Google AI Studio API key
- Check API quotas and billing settings
- Ensure API is enabled for your project

## ⚡ Performance Optimization

### For Better OCR Performance
```bash
# Install GPU version of PaddlePaddle (if you have NVIDIA GPU)
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

### For Faster Vector Operations
```bash
# Install GPU version of FAISS (if you have NVIDIA GPU)
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Memory Optimization
- Reduce batch size for large ticket processing
- Close browser tabs when processing large datasets
- Consider processing tickets in smaller chunks (50-100 at a time)

## 🔒 Security Notes

1. **Never commit secrets.toml to version control**
2. **Add .streamlit/secrets.toml to .gitignore**
3. **Use environment variables in production**
4. **Regularly rotate API keys**

## 📊 Usage Tips

### First Time Setup
1. Start with **Vector Database** tab
2. Process 10-20 tickets first to test setup
3. Check `ocr_extracted_text.log` for attachment processing results
4. Gradually increase ticket count once everything works

### Best Practices
- **OCR**: Works best with clear text images (screenshots, documents)
- **Vector Database**: Create separate databases for different time periods
- **Analysis**: Use Semantic Search for quick insights, Map-Reduce for deep analysis
- **Performance**: Save/load vector databases to avoid reprocessing

## 🆘 Getting Help

### Log Files to Check
- Streamlit logs in terminal
- `ocr_extracted_text.log` for attachment processing
- Browser developer console for JavaScript errors

### Useful Debug Commands
```bash
# Check installed packages
pip list | grep -E "(streamlit|faiss|paddle|sentence)"

# Test imports
python -c "import streamlit, faiss, paddleocr, sentence_transformers; print('All imports successful')"

# Check Streamlit version
streamlit version
```

### Performance Monitoring
- Monitor memory usage during large batch processing
- Check available disk space for model downloads and vector databases
- Watch API rate limits in Freshdesk admin panel

## 🚀 Ready to Go!

Once setup is complete, you should see:
- ✅ OCR libraries loaded successfully
- ✅ Vector database creation working
- ✅ Ticket analysis functional
- ✅ Similar ticket search operational

Start with a small batch of tickets to verify everything works, then scale up!