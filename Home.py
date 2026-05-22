import os
import streamlit as st

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "ocr_qna")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "sessions")

st.set_page_config(
    page_title="OCR Interview Generator",
    layout="wide",
)

st.sidebar.caption(f"DB: `{MONGODB_DB}` · `{MONGODB_COLLECTION}`")

st.title("Resume Scanner")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("Upload & Process")
    if st.button("Go to Upload", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Upload.py")

with col2:
    st.markdown("Past Results")
    if st.button("Go to Results", use_container_width=True,type="primary"):
        st.switch_page("pages/2_Results.py")

