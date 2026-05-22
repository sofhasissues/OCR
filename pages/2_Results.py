import os
import streamlit as st

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "ocr_qna")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "sessions")

st.set_page_config(page_title="Past Results",layout="wide")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.caption(f"DB: `{MONGODB_DB}` · `{MONGODB_COLLECTION}`")

# ── Page ───────────────────────────────────────────────────────────────────
st.title("Past Results")
st.write("Browse previously processed documents saved to MongoDB.")

try:
    from pymongo import MongoClient
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    collection = client[MONGODB_DB][MONGODB_COLLECTION]

    docs = list(collection.find({}, sort=[("created_at", -1)], limit=50))

    if not docs:
        st.info("No sessions saved yet. Process a document and enable 'Save to MongoDB'.")
    else:
        st.success(f"{len(docs)} session(s) found.")

        for doc in docs:
            created  = doc.get("created_at", "Unknown date")
            filename = doc.get("filename", "Unnamed")
            skills   = doc.get("detected_skills", [])

            with st.expander(f"{filename}  —  {created}", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**File info**")
                    st.write(f"Source type: `{doc.get('source_type', 'N/A')}`")
                    st.write(f"Model: `{doc.get('model_name', 'N/A')}`")
                    st.write(f"Questions requested: `{doc.get('requested_questions', 'N/A')}`")

                with col2:
                    st.markdown("**Detected skills**")
                    if skills:
                        tag_html = " ".join(
                            f'<span style="background:#1f77b4;color:white;padding:3px 9px;'
                            f'border-radius:10px;margin:2px;display:inline-block;font-size:0.82rem">'
                            f'{s}</span>'
                            for s in skills
                        )
                        st.markdown(tag_html, unsafe_allow_html=True)
                    else:
                        st.write("None recorded")

                st.markdown("**Generated questions**")
                st.text_area(
                    "", doc.get("generated_questions", ""),
                    height=180, key=str(doc["_id"]),
                    label_visibility="collapsed"
                )

                with st.expander("Show extracted text", expanded=False):
                    st.text_area(
                        "", doc.get("extracted_text", ""),
                        height=150, key=f"txt_{doc['_id']}",
                        label_visibility="collapsed"
                    )

except ImportError:
    st.error("pymongo is not installed. Run: `pip install pymongo`")
except Exception as exc:
    st.warning("Could not connect to MongoDB.")
    st.error(str(exc))
    st.info("Make sure MongoDB is running and MONGODB_URI is set correctly in your .env file.")
