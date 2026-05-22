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

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"


def _groq_chat(system: str, user: str, api_key: str, max_tokens: int = 1024) -> str:
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,  # Lowered temperature for more stable JSON output
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


def _clean_and_parse_json(raw: str, default_factory: Any) -> Any:
    """Robust helper to extract and clean JSON arrays or objects from LLM text."""
    # Remove markdown code blocks
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```", "", raw).strip()
    
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if not match:
        return default_factory()
        
    json_content = match.group(1)
    
    try:
        return json.loads(json_content)
    except json.JSONDecodeError:
        # Fallback repair: convert single quotes to double quotes if LLM used them
        try:
            fixed_json = json_content.replace("'", '"')
            return json.loads(fixed_json)
        except Exception:
            return default_factory()


def _parse_questions_json(raw: str) -> Dict[str, List[str]]:
    data = _clean_and_parse_json(raw, dict)
    if not isinstance(data, dict):
        return {}
    
    result: Dict[str, List[str]] = {}
    for skill, qs in data.items():
        if isinstance(qs, list):
            clean = [str(q).strip() for q in qs if q and str(q).strip()]
            if clean:
                result[str(skill)] = clean
    return result


def _parse_skills_json(raw: str) -> List[str]:
    data = _clean_and_parse_json(raw, list)
    if not isinstance(data, list):
        return []
    return [str(s).strip() for s in data if s and (2 <= len(str(s).strip()) <= 40)]


def extract_skills(text: str, groq_key: Optional[str] = None) -> List[str]:
    gk = groq_key or os.getenv("GROQ_API_KEY", "")
    if not gk:
        print("Missing GROQ_API_KEY")
        return []

    system = """You are an ATS resume parser.
Extract ONLY technical skills from the resume text.
Return ONLY a valid JSON array of strings. No conversation, no explanations, no markdown markdown fences.

Example Output:
["Python", "Django", "MongoDB", "Docker"]"""

    user = f"Resume Text:\n\n{text[:5000]}"

    try:
        raw = _groq_chat(system=system, user=user, api_key=gk, max_tokens=500)
        skills = _parse_skills_json(raw)

        cleaned = []
        seen = set()
        for skill in skills:
            if skill and skill.lower() not in seen:
                cleaned.append(skill)
                seen.add(skill.lower())
        return cleaned
    except Exception as e:
        print("Skill extraction error:", e)
        return []


def _build_question_prompt(skills: List[str], total: int) -> Tuple[str, str]:
    skills_str = ", ".join(skills)
    system = (
        "You are an expert technical interviewer. "
        "Generate unique, non-repetitive interview questions based on the candidate's skills. "
        "Return ONLY a valid JSON object map matching the exact formatting rule. No explanations. "
        "Format: {\"SkillName\": [\"question1\", \"question2\"], \"AnotherSkill\": [\"question3\"]}"
    )
    user = (
        f"The candidate has the following skills: {skills_str}.\n\n"
        f"Generate a total of exactly {total} interview questions distributed across these skills. "
        f"Include a mix of technical, conceptual, and situational questions. "
        f"Return valid JSON only."
    )
    return system, user


def _generate_via_groq(skills: List[str], total: int, api_key: str) -> Dict[str, List[str]]:
    system, user = _build_question_prompt(skills, total)
    raw = _groq_chat(system, user, api_key, max_tokens=2048)
    return _parse_questions_json(raw)


def generate_skill_based_questions(
    text: str,
    max_questions: int,
    skills: Optional[List[str]] = None,
    groq_key: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], str]:
    gk = groq_key or os.getenv("GROQ_API_KEY", "")

    if skills is None:
        skills = extract_skills(text, groq_key=gk)

    if not skills:
        skills = ["General Programming"]

    try:
        result = _generate_via_groq(skills, max_questions, gk)
        if result:
            return result, "groq"
    except Exception as e:
        print("Question generation error:", e)

    return {}, "groq"


def format_skill_questions(skill_questions: Dict[str, List[str]]) -> str:
    lines = []
    for skill, qs in skill_questions.items():
        lines.append(f"--- {skill} ---")
        for i, q in enumerate(qs, 1):
            lines.append(f"{i}. {q}")
        lines.append("")
    return "\n".join(lines).strip()


# ── OCR & Document Processing ──────────────────────────────────────────────
def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image bytes")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cleaned = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    pil_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return pytesseract.image_to_string(pil_image, lang="eng").strip()


def _render_pdf_with_fitz(pdf_source: bytes, dpi: int = 150):
    if fitz is None:
        raise RuntimeError("PDF rendering via PyMuPDF is unavailable.")
    doc = fitz.open(stream=pdf_source, filetype="pdf")
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append(img)
    return images


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=300)
    except (PDFInfoNotInstalledError, OSError):
        pages = _render_pdf_with_fitz(pdf_bytes, dpi=150)

    # Fallback to system path default execution if your configuration differs
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except Exception:
        pass

    texts = []
    for page in pages:
        text = pytesseract.image_to_string(page, lang="eng")
        texts.append(text)
    return "\n\n".join(texts).strip()


def clean_text(text: str) -> str:
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])



def create_tts_audio(text: str, output_path: Optional[str] = None) -> str:
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 is not installed.")
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(), f"ocr_tts_{int(datetime.now().timestamp())}.wav"
        )
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path


def save_to_mongo(record: Dict[str, Any], mongo_uri: Optional[str] = None, db_name: str = "ocr_qna", collection_name: str = "sessions") -> str:
    if MongoClient is None:
        raise RuntimeError("pymongo is not installed.")
    mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    collection = client[db_name][collection_name]
    return str(collection.insert_one(record).inserted_id)


def build_session_record(filename: str, source_type: str, extracted_text: str, questions: str, model_name: str, max_questions: int, saved_audio_path: Optional[str] = None, detected_skills: Optional[List[str]] = None) -> Dict[str, Any]:
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
