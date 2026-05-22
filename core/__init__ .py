"""Core OCR and question generation utilities."""

from .ocr_service import (
    build_session_record,
    clean_text,
    create_tts_audio,
    extract_skills,
    extract_text_from_image,
    extract_text_from_image_bytes,
    extract_text_from_pdf,
    extract_text_from_pdf_bytes,
    format_skill_questions,
    generate_questions,
    generate_skill_based_questions,
    save_to_mongo,
)
