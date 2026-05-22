import os
import streamlit as st

from core.ocr_service import (
    build_session_record,
    clean_text,
    create_tts_audio,
    extract_skills,
    extract_text_from_image_bytes,
    extract_text_from_pdf_bytes,
    format_skill_questions,
    generate_skill_based_questions,
    save_to_mongo,
)

MONGODB_URI        = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB         = os.getenv("MONGODB_DB", "ocr_qna")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "sessions")

PROVIDER_LABELS = {
    "groq":     "Groq  (llama3-8b-8192)",
    "gemini":   "Gemini  (gemini-1.5-flash)",
    "template": "Template fallback",
}

st.set_page_config(page_title="Upload & Process",layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("**Model settings**")

groq_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
    help="Free key → https://console.groq.com  |  Model: llama3-8b-8192",
)
gemini_key = st.sidebar.text_input(
    "Gemini API Key  (fallback)",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password",
    help="Free key → https://aistudio.google.com/app/apikey  |  Model: gemini-1.5-flash",
)

# Status indicators
if groq_key:
    st.sidebar.success("Groq ✓  (primary)")
elif gemini_key:
    st.sidebar.warning("Groq key missing — Gemini will be used")
else:
    st.sidebar.error("No API keys — template fallback will be used")

max_questions = st.sidebar.slider(
    "Total questions", min_value=3, max_value=30, value=9,
    help="Total questions spread across ALL detected skills.",
)
use_tts    = st.sidebar.checkbox("Enable TTS playback", value=True)
save_to_db = st.sidebar.checkbox("Save to MongoDB", value=False)
if save_to_db:
    st.sidebar.caption(f"URI: `{MONGODB_URI}`")
    st.sidebar.caption(f"DB: `{MONGODB_DB}` · `{MONGODB_COLLECTION}`")

# ── Page ───────────────────────────────────────────────────────────────────
st.title("Upload & Process")
st.write("Upload a resume or scanned document to extract skills and generate interview questions.")

uploaded_file = st.file_uploader(
    "Upload document", type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf"]
)

# Image preview
if uploaded_file is not None:
    if not uploaded_file.name.lower().endswith(".pdf"):
        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
    else:
        st.info(f"PDF uploaded: **{uploaded_file.name}**")

if st.button("Process Document", type="primary", disabled=(uploaded_file is None)):
    try:
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        filename   = uploaded_file.name
        file_type  = uploaded_file.type
        gk         = groq_key   or None
        gmk        = gemini_key or None

        # Step 1: OCR
        with st.spinner("Extracting text from document..."):
            if filename.lower().endswith(".pdf") or file_type == "application/pdf":
                source_type    = "pdf"
                extracted_text = extract_text_from_pdf_bytes(file_bytes)
            else:
                source_type    = "image"
                extracted_text = extract_text_from_image_bytes(file_bytes)

        cleaned_text = clean_text(extracted_text)
        if not cleaned_text:
            st.error("No text could be extracted from the document.")
            st.stop()

        with st.expander("Extracted text", expanded=False):
            st.text_area("", cleaned_text, height=220, label_visibility="collapsed")

        # Step 2: Skill extraction
        with st.spinner("Detecting skills..."):
            detected_skills = extract_skills(cleaned_text, gk)

        st.subheader("Detected skills")
        if detected_skills:
            tag_html = " ".join(
                f'<span style="background:#1f77b4;color:white;padding:4px 10px;'
                f'border-radius:12px;margin:3px;display:inline-block;font-size:0.85rem">'
                f'{skill}</span>'
                for skill in detected_skills
            )
            st.markdown(tag_html, unsafe_allow_html=True)
            st.markdown("")
            edited = st.text_input(
                "Edit detected skills (comma-separated)",
                value=", ".join(detected_skills),
                help="Add, remove, or rename skills before generating questions.",
            )
            final_skills = [s.strip() for s in edited.split(",") if s.strip()]
        else:
            st.warning("No specific skills detected. Questions will be generated generically.")
            final_skills = []

        # Step 3: Question generation
        with st.spinner("Generating interview questions..."):
            try:
                skill_questions, provider = generate_skill_based_questions(
                    text=cleaned_text,
                    max_questions=max_questions,
                    skills=final_skills or None,
                    groq_key=gk,
                    gemini_key=gmk,
                )
            except Exception as exc:
                st.error(f"Question generation error: {exc}")
                skill_questions, provider = {}, "template"

        total_generated = sum(len(qs) for qs in skill_questions.values())
        provider_label  = PROVIDER_LABELS.get(provider, provider)

        st.subheader(f"Interview questions by skill  ({total_generated} total)")
        st.caption(f"Generated using: {provider_label}")

        if skill_questions:
            for skill, questions in skill_questions.items():
                with st.expander(f"🔹 {skill}  ({len(questions)} questions)", expanded=True):
                    for i, q in enumerate(questions, 1):
                        st.markdown(f"**Q{i}.** {q}")
        else:
            st.info("No questions were generated.")

        flat_questions = format_skill_questions(skill_questions)

        # Step 4: TTS
        audio_path = None
        if use_tts:
            st.subheader("Text-to-speech playback")
            try:
                tts_text = (
                    "The following interview questions were generated based on "
                    "the candidate's skills. " + flat_questions
                )
                audio_path = create_tts_audio(tts_text)
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/wav")
            except Exception as exc:
                st.warning(f"TTS failed: {exc}")

        # Step 5: MongoDB
        if save_to_db:
            record = build_session_record(
                filename=filename,
                source_type=source_type,
                extracted_text=cleaned_text,
                questions=flat_questions,
                model_name=provider,
                max_questions=max_questions,
                saved_audio_path=audio_path,
                detected_skills=final_skills,
            )
            try:
                inserted_id = save_to_mongo(
                    record,
                    mongo_uri=MONGODB_URI,
                    db_name=MONGODB_DB,
                    collection_name=MONGODB_COLLECTION,
                )
                st.success(f"Saved to MongoDB — ID: `{inserted_id}`")
            except Exception as exc:
                st.warning("Could not save to MongoDB. Processing still completed.")
                st.error(f"MongoDB error: {exc}")

    except Exception as exc:
        st.error(f"Error processing document: {exc}")
        st.exception(exc)

elif uploaded_file is None:
    st.info("Upload a document above to get started.")
