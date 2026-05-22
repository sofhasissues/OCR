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

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "ocr_qna")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "sessions")

st.set_page_config(page_title="Upload & Process", layout="wide")

# ── Session State Initialization ───────────────────────────────────────────
# We use session_state so the app doesn't forget the skills when you type in the text box.
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = None
if "detected_skills" not in st.session_state:
    st.session_state.detected_skills = []
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "source_type" not in st.session_state:
    st.session_state.source_type = None

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("**Model Settings**")

groq_key = st.sidebar.text_input(
    "Groq API Key",
    value=os.getenv("GROQ_API_KEY", ""),
    type="password",
    help="Free key → https://console.groq.com | Model: llama3-8b-8192",
)

if groq_key:
    st.sidebar.success("Groq key loaded ✓")
else:
    st.sidebar.error("Missing Groq API Key")

max_questions = st.sidebar.slider(
    "Total Questions", min_value=3, max_value=30, value=9,
    help="Total questions spread across all detected skills.",
)
use_tts = st.sidebar.checkbox("Enable TTS playback", value=True)
save_to_db = st.sidebar.checkbox("Save to MongoDB", value=False)

if save_to_db:
    st.sidebar.caption(f"URI: `{MONGODB_URI}`")
    st.sidebar.caption(f"DB: `{MONGODB_DB}` · `{MONGODB_COLLECTION}`")

# ── Page Content ───────────────────────────────────────────────────────────
st.title("Upload & Process")
st.write("Upload a resume or scanned document to extract skills and generate interview questions exclusively via Groq.")

uploaded_file = st.file_uploader(
    "Upload document", type=["png", "jpg", "jpeg", "tiff", "bmp", "pdf"]
)

# Reset state if a new file is uploaded
if uploaded_file is not None:
    if st.session_state.current_filename != uploaded_file.name:
        st.session_state.extracted_text = None
        st.session_state.detected_skills = []
        st.session_state.current_filename = uploaded_file.name
        
    if not uploaded_file.name.lower().endswith(".pdf"):
        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
    else:
        st.info(f"PDF uploaded: **{uploaded_file.name}**")
else:
    st.info("Drop your file above to begin scanning metrics.")
    st.stop() # Stop execution until a file is uploaded

# ==========================================
# STEP 1: OCR & SKILL EXTRACTION
# ==========================================
if st.session_state.extracted_text is None:
    if st.button("Step 1: Extract Text & Skills", type="primary", disabled=not groq_key):
        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            file_type = uploaded_file.type

            # OCR Processing
            with st.spinner("Extracting text from document..."):
                if filename.lower().endswith(".pdf") or file_type == "application/pdf":
                    st.session_state.source_type = "pdf"
                    raw_text = extract_text_from_pdf_bytes(file_bytes)
                else:
                    st.session_state.source_type = "image"
                    raw_text = extract_text_from_image_bytes(file_bytes)

            cleaned_text = clean_text(raw_text)
            
            if not cleaned_text:
                st.error("No text could be extracted. Please check the scanner quality.")
                st.stop()

            st.session_state.extracted_text = cleaned_text

            # Skill Extraction
            with st.spinner("Detecting skills using Groq..."):
                skills = extract_skills(cleaned_text, groq_key)
                st.session_state.detected_skills = skills if skills else ["General Programming"]
                
            st.rerun() # Refresh the page to show Step 2

        except Exception as exc:
            st.error(f"Error during extraction: {exc}")

# ==========================================
# STEP 2: EDIT SKILLS & GENERATE QUESTIONS
# ==========================================
if st.session_state.extracted_text is not None:
    st.success("✅ Text and skills extracted successfully.")
    
    with st.expander("Preview Extracted Text", expanded=False):
        st.text_area("", st.session_state.extracted_text, height=200, label_visibility="collapsed")

    st.subheader("Detected Skills")
    
    # Display pills for visual feedback
    tag_html = " ".join(
        f'<span style="background:#1f77b4;color:white;padding:4px 10px;'
        f'border-radius:12px;margin:3px;display:inline-block;font-size:0.85rem">'
        f'{skill}</span>'
        for skill in st.session_state.detected_skills
    )
    st.markdown(tag_html, unsafe_allow_html=True)
    st.markdown("")
    
    # Editable skills input (This is safe now because it's no longer inside the Step 1 button loop)
    edited_skills_str = st.text_input(
        "Edit detected skills (comma-separated)",
        value=", ".join(st.session_state.detected_skills),
        help="Modify extracted skills directly before generating questions."
    )
    final_skills = [s.strip() for s in edited_skills_str.split(",") if s.strip()]

    # Generate Questions Button
    if st.button("Step 2: Generate Interview Questions", type="primary"):
        with st.spinner("Generating interview questions via Groq..."):
            try:
                skill_questions, provider = generate_skill_based_questions(
                    text=st.session_state.extracted_text,
                    max_questions=max_questions,
                    skills=final_skills,
                    groq_key=groq_key,
                )

                total_generated = sum(len(qs) for qs in skill_questions.values())
                st.subheader(f"Interview Questions By Skill ({total_generated} total)")

                if skill_questions:
                    for skill, questions in skill_questions.items():
                        with st.expander(f"🔹 {skill} ({len(questions)} questions)", expanded=True):
                            for i, q in enumerate(questions, 1):
                                st.markdown(f"**Q{i}.** {q}")
                else:
                    st.error("Failed to generate structural questions from the model output.")

                flat_questions = format_skill_questions(skill_questions)

                # Text-To-Speech
                audio_path = None
                if use_tts and flat_questions:
                    st.subheader("Audio Playback")
                    try:
                        tts_text = f"The following interview questions were generated. {flat_questions}"
                        audio_path = create_tts_audio(tts_text)
                        with open(audio_path, "rb") as f:
                            st.audio(f.read(), format="audio/wav")
                    except Exception as exc:
                        st.warning(f"TTS playback synthesis failed: {exc}")

                # Save to MongoDB
                if save_to_db and flat_questions:
                    record = build_session_record(
                        filename=st.session_state.current_filename,
                        source_type=st.session_state.source_type,
                        extracted_text=st.session_state.extracted_text,
                        questions=flat_questions,
                        model_name="groq",
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
                        st.success(f"Session safely synced to MongoDB. Index ID: `{inserted_id}`")
                    except Exception as exc:
                        st.error(f"MongoDB storage transaction failed: {exc}")

            except Exception as exc:
                st.error(f"Critical execution error during generation routine: {exc}")
                st.exception(exc)
