# DDR Report Generator

AI-powered Detailed Diagnostic Report generator. Uploads inspection + thermal documents → generates structured DDR report → download as PDF.

## Stack
- **Backend**: FastAPI (Python)
- **AI**: Llama 3 via Groq API (free)
- **PDF extraction**: PyMuPDF
- **Frontend**: HTML/CSS/JS

## Setup

### 1. Get free Groq API key
Go to https://console.groq.com → Sign up free → API Keys → Create key

### 2. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Set API key and run
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_key_here"
python -m uvicorn main:app --reload --port 8000

# Mac/Linux
export GROQ_API_KEY="your_key_here"
python -m uvicorn main:app --reload --port 8000
```

### 4. Open
```
http://localhost:8000
```

## Share live link (ngrok)
```bash
ngrok http 8000
```

## Features
- Extracts text + images from PDFs
- OCR fallback for scanned PDFs
- Cross-references inspection + thermal findings by area
- 7-section DDR report
- Severity assessment table
- Downloadable PDF
- Handles missing/conflicting data
