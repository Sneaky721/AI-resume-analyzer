# AI Resume Analyzer (Flask + NLP)

A beginner-friendly AI-based Resume Analyzer web application built with:

- Python + Flask
- spaCy (text preprocessing)
- scikit-learn (TF-IDF + cosine similarity)
- PyPDF2 (PDF text extraction)
- HTML/CSS frontend

The app compares an uploaded resume with a pasted job description and provides:

- Resume-job match score
- Extracted resume skills
- Extracted job description skills
- Common and missing skills
- Keyword optimization suggestions
- Practical improvement feedback

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- templates/
|   |-- index.html
|   `-- result.html
`-- static/
    `-- style.css
```

## Features

1. **Resume Upload**
   - Supports `.pdf` and `.txt` files
   - Extracts content using PyPDF2 (PDF) or text decoding (TXT)

2. **Job Description Input**
   - Multi-line text area for pasted JD content

3. **NLP Preprocessing**
   - Lowercasing, tokenization, stopword and punctuation removal
   - Lemmatization through spaCy (with fallback support)

4. **Match Scoring**
   - TF-IDF vectorization and cosine similarity using scikit-learn
   - Outputs percentage match score

5. **Skill Comparison**
   - Infers skills from the job description + resume using TF-IDF n-grams and sentence embeddings
   - Falls back to a predefined skill dictionary only if embeddings aren't available
   - Shows resume skills, JD skills, common skills, missing skills, and extra resume skills

6. **Suggestions**
   - Recommends missing keywords
   - Gives practical, ATS-friendly improvements

## Setup and Run

### 1) Create and activate virtual environment (recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Install spaCy English model

```bash
python -m spacy download en_core_web_sm
```

> If the model is not installed, the app still runs using a fallback tokenizer.

### 4) (Optional) Enable LLM-generated tips

By default, the app uses deterministic tips (fast and reliable).
If you want LLM-generated suggestions, enable it with an environment variable:

**Windows (PowerShell):**
```powershell
$env:ENABLE_LLM="1"
python app.py
```

First time you enable this, it may download models from Hugging Face (local, free, but can take a few minutes).

### 5) Run the app

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Notes
- No database is used (local, stateless app).
- Maximum upload size is 5 MB.
- For best results, upload text-based PDFs (not scanned image PDFs).
