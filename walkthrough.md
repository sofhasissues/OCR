# Project Walkthrough

## Overview

This project scans documents, extracts text using OCR, generates study questions from the extracted text, and optionally converts the text to speech. The web interface is built with Streamlit and the results can be saved to MongoDB.

## Structure

- `core/` — core OCR, question generation, TTS, and MongoDB helper code
- `pages/` — reserved for future Streamlit pages or app modules
- `uploads/` — reserved for temporary document uploads
- `Home.py` — main Streamlit app page
- `run_app.py` — helper script to launch the Streamlit UI
- `groq_check.py` — placeholder for future validation or model checks
- `.env` — environment variables
- `.gitignore` — ignored files and folders
- `.streamlit/` — Streamlit configuration

## Setup

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Install Poppler for PDF support and add it to PATH.

- Windows: download from https://poppler.freedesktop.org/ and add the `bin` folder to your PATH.
- MacOS: `brew install poppler`
- Linux: `sudo apt install poppler-utils`

3. Set up local environment variables in `.env` if needed, e.g.:

```bash
MONGODB_URI=mongodb://localhost:27017
MODEL_NAME=t5-small
```

4. Run the UI:

```bash
python run_app.py
```

or:

```bash
streamlit run Home.py
```

## Notes

- Use `uploads/` for temporary document storage if you want to persist uploads locally.
- The Streamlit app can generate a temporary WAV file for TTS playback.
- MongoDB storage is optional and controlled through the sidebar.
