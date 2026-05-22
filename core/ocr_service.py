import os
import re
import json
import tempfile
import urllib.request
import urllib.error
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes, convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


# ---------------------------------------------------------------------------
# API config
# Priority: Groq (llama3-8b-8192) → Gemini (gemini-1.5-flash) → templates
#
# Groq  — free key at https://console.groq.com
# Gemini — free key at https://aistudio.google.com/app/apikey
# ---------------------------------------------------------------------------
GROQ_API_URL    = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL      = "llama3-8b-8192"

GEMINI_API_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_MODEL    = "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# Groq call
# ---------------------------------------------------------------------------
def _groq_chat(system: str, user: str, api_key: str, max_tokens: int = 1024) -> str:
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Shared JSON parser — used by both Groq and Gemini responses
# ---------------------------------------------------------------------------
def _parse_questions_json(raw: str) -> Dict[str, List[str]]:
    """Extract and parse a JSON object from a model response."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("Response did not contain a JSON object")
    data = json.loads(match.group())
    result: Dict[str, List[str]] = {}
    for skill, qs in data.items():
        if isinstance(qs, list):
            clean = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
            if clean:
                result[skill] = clean
    return result


def _parse_skills_json(raw: str) -> List[str]:
    """Extract and parse a JSON array of skill strings from a model response."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        skills = json.loads(match.group())
        return [s.strip() for s in skills if isinstance(s, str) and 2 <= len(s) <= 40]
    return []


# ---------------------------------------------------------------------------
# Skill keyword bank
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Template fallback pool — used only when both APIs fail/unavailable
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Skill extraction using ONLY Groq
# ---------------------------------------------------------------------------

def extract_skills(
    text: str,
    groq_key: Optional[str] = None,
) -> List[str]:

    gk = groq_key or os.getenv("GROQ_API_KEY", "")

    if not gk:
        print("Missing GROQ_API_KEY")
        return []

    system = """
You are an ATS resume parser.

Extract ONLY technical skills from the resume.

RULES:
- Return ONLY a valid JSON array
- No explanation
- No markdown
- No duplicate skills
- Only technical skills
- Ignore soft skills

Example:
["Python", "Django", "MongoDB", "Docker"]
"""

    user = f"""
Resume Text:

{text[:5000]}
"""

    try:
        raw = _groq_chat(
            system=system,
            user=user,
            api_key=gk,
            max_tokens=500
        )

        print("RAW SKILLS RESPONSE:", raw)

        skills = _parse_skills_json(raw)

        # Remove duplicates
        cleaned = []
        seen = set()

        for skill in skills:
            s = skill.strip()

            if s and s.lower() not in seen:
                cleaned.append(s)
                seen.add(s.lower())

        return cleaned

    except Exception as e:
        print("Skill extraction error:", e)
        return []
# ---------------------------------------------------------------------------
# Question generation — shared prompt builder
# ---------------------------------------------------------------------------
def _build_question_prompt(skills: List[str], total: int) -> Tuple[str, str]:
    """Returns (system_prompt, user_prompt) for question generation."""
    skills_str = ", ".join(skills)
    system = (
        "You are an expert technical interviewer. "
        "Generate unique, non-repetitive interview questions. "
        "Return ONLY a valid JSON object — no explanation, no markdown fences. "
        "Format: {\"SkillName\": [\"question1\", \"question2\", ...], ...}"
    )
    user = (
        f"The candidate has the following skills: {skills_str}.\n\n"
        f"Generate a total of exactly {total} interview questions spread across all skills. "
        f"Distribute them proportionally — skills the candidate emphasises more get more questions. "
        f"For each skill include a mix of technical, conceptual, and situational questions. "
        f"Every question must be unique. Do not repeat any question across skills.\n\n"
        f"Return JSON only."
    )
    return system, user


def _generate_via_groq(skills: List[str], total: int, api_key: str) -> Dict[str, List[str]]:
    system, user = _build_question_prompt(skills, total)
    raw = _groq_chat(system, user, api_key, max_tokens=2048)
    return _parse_questions_json(raw)



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_skill_based_questions(
    text: str,
    max_questions: int,
    skills: Optional[List[str]] = None,
    groq_key: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], str]:

    gk = groq_key or os.getenv("GROQ_API_KEY", "")

    # Extract skills using Groq
    if skills is None:
        skills = extract_skills(text, groq_key=gk)

    print("Detected Skills:", skills)

    # If no skills found
    if not skills:
        print("No specific skills detected. Questions will be generated generically.")

        skills = ["General Programming"]

    # Generate questions using Groq
    try:
        result = _generate_via_groq(skills, max_questions, gk)

        if result:
            return result, "groq"

    except Exception as e:
        print("Question generation error:", e)

    # Fallback return
    return {
        "General Programming": [
            "Tell me about yourself.",
            "Explain a challenging project you worked on.",
            "What are your strengths in programming?",
            "How do you debug code?",
            "Explain OOP concepts."
        ]
    }, "fallback"
def format_skill_questions(skill_questions: Dict[str, List[str]]) -> str:
    lines = []
    for skill, qs in skill_questions.items():
        lines.append(f"--- {skill} ---")
        for i, q in enumerate(qs, 1):
            lines.append(f"{i}. {q}")
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------
def extract_text_from_image(image_path: str) -> str:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    return extract_text_from_image_array(image)


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image bytes")
    return extract_text_from_image_array(image)


def extract_text_from_image_array(image: Any) -> str:
    if image is None:
        raise ValueError("Invalid image data")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cleaned = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    pil_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    text = pytesseract.image_to_string(pil_image, lang="eng")
    return text.strip()


def _render_pdf_with_fitz(pdf_source: bytes, dpi: int = 150):
    if fitz is None:
        raise RuntimeError(
            "PDF rendering via PyMuPDF is unavailable. Install with `pip install pymupdf`."
        )
    doc = fitz.open(stream=pdf_source, filetype="pdf")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img)
    return images


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        pages = convert_from_path(pdf_path, dpi=300)
    except (PDFInfoNotInstalledError, OSError):
        if fitz is not None:
            with open(pdf_path, "rb") as f:
                pages = _render_pdf_with_fitz(f.read(), dpi=150)
        else:
            raise RuntimeError("Poppler not installed and PyMuPDF not available.")
    texts = []
    for page in pages:
        text = pytesseract.image_to_string(page, lang="eng")
        texts.append(text)
    return "\n\n".join(texts).strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300)
    except (PDFInfoNotInstalledError, OSError):
        pages = _render_pdf_with_fitz(pdf_bytes, dpi=150)

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    texts = []
    for page in pages:
        text = pytesseract.image_to_string(page, lang="eng")
        texts.append(text)
    return "\n\n".join(texts).strip()


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def create_tts_audio(text: str, output_path: Optional[str] = None) -> str:
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 is not installed. Run: pip install pyttsx3")
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(), f"ocr_tts_{int(datetime.now().timestamp())}.wav"
        )
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
def save_to_mongo(
    record: Dict[str, Any],
    mongo_uri: Optional[str] = None,
    db_name: str = "ocr_qna",
    collection_name: str = "sessions",
) -> str:
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed. Run: pip install pymongo")
    mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    collection = client[db_name][collection_name]
    insert_result = collection.insert_one(record)
    return str(insert_result.inserted_id)


def build_session_record(
    filename: str,
    source_type: str,
    extracted_text: str,
    questions: str,
    model_name: str,
    max_questions: int,
    saved_audio_path: Optional[str] = None,
    detected_skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    record = {
        "filename": filename,
        "source_type": source_type,
        "model_name": model_name,
        "requested_questions": max_questions,
        "extracted_text": extracted_text,
        "generated_questions": questions,
        "detected_skills": detected_skills or [],
        "created_at": datetime.utcnow(),
    }
    if saved_audio_path:
        record["audio_path"] = saved_audio_path
    return record


# Backwards compatibility
def generate_questions(text: str, model_name: str, max_questions: int) -> str:
    skill_questions, _ = generate_skill_based_questions(text, max_questions)
    return format_skill_questions(skill_questions)
