"""Interactive Web Interface & Demo Dashboard for ZentroMart VoiceOps 2026.

Run with:
    streamlit run app.py
"""
import os
import json
import time
import streamlit as st
from pathlib import Path
from src.agent import ZentroMartAgent
from src.retriever import ZentroMartRetriever
from src.voice import (
    VoicePipeline,
    normalize_text_for_speech,
    GNANI_STT_LANGUAGES,
    GNANI_TTS_LANGUAGES
)
from src.config import REFERENCE_DATE_STR, OUTPUTS_DIR

st.set_page_config(
    page_title="ZentroMart VoiceOps 2026",
    page_icon="🎙️",
    layout="wide"
)

# Custom CSS for polished, modern look
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #64748B; margin-bottom: 1.5rem; }
    .card { background: #FFFFFF; border-radius: 10px; padding: 1.2rem; border: 1px solid #E2E8F0; margin-bottom: 1rem; }
    .badge-verified { background-color: #DCFCE7; color: #166534; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .badge-pending { background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .badge-failed { background-color: #FEE2E2; color: #991B1B; padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "agent" not in st.session_state:
    st.session_state.retriever = ZentroMartRetriever()
    st.session_state.agent = ZentroMartAgent(mode="rag", retriever=st.session_state.retriever)
    st.session_state.voice = VoicePipeline()
    st.session_state.messages = []
    st.session_state.retrieved_history = []

# Sidebar Navigation and Configuration
st.sidebar.title("🎙️ VoiceOps 2026")
st.sidebar.markdown(f"**Reference Date:** `{REFERENCE_DATE_STR}`")

mode_selection = st.sidebar.radio("Agent Engine Mode", ["Grounded RAG (Step 2/3)", "Baseline Prompt-Only (Step 1)"])
st.session_state.agent.mode = "rag" if "RAG" in mode_selection else "baseline"

st.sidebar.markdown("---")
st.sidebar.subheader("Gnani Voice Settings")
voice_model = st.sidebar.selectbox("TTS Model", ["timbre-v2.5", "timbre-v3.0"])
voice_name = st.sidebar.selectbox("Voice", ["Kaveri", "Trupti", "Pranav", "Shlok"])

# Voice Languages
stt_lang = st.sidebar.selectbox(
    "Customer Input Language (STT)",
    options=list(GNANI_STT_LANGUAGES.keys()),
    index=list(GNANI_STT_LANGUAGES.keys()).index("en-IN") if "en-IN" in GNANI_STT_LANGUAGES else 0,
    format_func=lambda x: GNANI_STT_LANGUAGES.get(x, x)
)
tts_lang = st.sidebar.selectbox(
    "Agent Reply Language (TTS)",
    options=list(GNANI_TTS_LANGUAGES.keys()),
    index=list(GNANI_TTS_LANGUAGES.keys()).index("en-IN") if "en-IN" in GNANI_TTS_LANGUAGES else 0,
    format_func=lambda x: GNANI_TTS_LANGUAGES.get(x, x)
)

if st.session_state.voice.use_live_gnani:
    st.sidebar.success("🟢 Gnani API Connected")
else:
    st.sidebar.info("🟡 Voice: Offline Simulation Mode (Add `GNANI_API_KEY` in `.env` to enable live Kaveri TTS voice)")

# One-Click Downloads for Submission
st.sidebar.markdown("---")
st.sidebar.subheader("📦 Submission Downloads")
zip_path = Path("team_voiceops_2026_voiceops.zip")
if zip_path.exists():
    st.sidebar.download_button(
        label="📥 Download Submission ZIP",
        data=zip_path.read_bytes(),
        file_name="team_voiceops_2026_voiceops.zip",
        mime="application/zip"
    )

pdf_path = Path("report.pdf")
if pdf_path.exists():
    st.sidebar.download_button(
        label="📄 Download Report PDF",
        data=pdf_path.read_bytes(),
        file_name="report.pdf",
        mime="application/pdf"
    )

if st.sidebar.button("▶️ Test Voice Audio (Sample)"):
    test_wav = OUTPUTS_DIR / f"test_sample_{int(time.time())}.wav"
    st.session_state.voice.synthesize(
        "Hello, this is ZentroMart customer support voice assistant. How may I help you with your order today?",
        str(test_wav),
        voice=voice_name,
        model=voice_model,
        language=tts_lang
    )
    if test_wav.exists():
        st.sidebar.audio(test_wav.read_bytes(), format="audio/wav")
        st.sidebar.caption("✅ Audio generated successfully!")

if st.sidebar.button("🔄 Reset Conversation Session"):
    st.session_state.agent.reset_conversation()
    st.session_state.messages = []
    st.session_state.retrieved_history = []
    st.rerun()

# Header
st.markdown('<div class="main-header">ZentroMart AI Customer Support Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Consulting & Analytics Club, IIT Guwahati × Gnani.ai Capstone Project</div>', unsafe_allow_html=True)

# Main layout: 2 Columns (Left: Chat & Voice, Right: Inspection & Tools)
col_chat, col_inspect = st.columns([1.3, 1.0])

with col_chat:
    st.subheader("💬 Support Conversation")

    # Display chat message history
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio_path") and Path(msg["audio_path"]).exists():
                st.audio(msg["audio_path"])

    # Customer input options: Text input or Audio test
    user_input = st.chat_input("Type your message here (or share order ID, e.g. 8978 2837)...")

    uploaded_audio = st.file_uploader("Or upload customer voice clip (.wav / .mp3):", type=["wav", "mp3", "ogg"], key="audio_upload")
    if uploaded_audio and st.button("🎙️ Process Voice Clip"):
        temp_in = OUTPUTS_DIR / f"temp_{uploaded_audio.name}"
        temp_in.write_bytes(uploaded_audio.read())

        with st.spinner("Transcribing via Gnani STT and processing..."):
            temp_out = OUTPUTS_DIR / f"reply_{Path(uploaded_audio.name).stem}.wav"
            heard, reply, out_wav = st.session_state.voice.voice_turn(
                str(temp_in),
                st.session_state.agent.respond,
                language_code=stt_lang,
                tts_language=tts_lang,
                voice=voice_name,
                model=voice_model,
                output_wav_path=str(temp_out)
            )
            st.session_state.messages.append({"role": "user", "content": f"🎤 *[Voice In ({stt_lang})]*: {heard}"})
            st.session_state.messages.append({"role": "assistant", "content": reply, "audio_path": out_wav})
            st.rerun()

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("ZentroMart Agent thinking..."):
            reply = st.session_state.agent.respond(user_input)

            # Generate spoken audio reply
            out_wav_path = str(OUTPUTS_DIR / f"reply_turn_{len(st.session_state.messages)}.wav")
            st.session_state.voice.synthesize(
                reply,
                out_wav_path,
                voice=voice_name,
                model=voice_model,
                language=tts_lang
            )
            st.session_state.messages.append({"role": "assistant", "content": reply, "audio_path": out_wav_path})
            st.rerun()

with col_inspect:
    st.subheader("🔍 Brain & State Inspector")

    # 1. Identity Verification & Order State
    tm = st.session_state.agent.tool_manager
    st.markdown("### 🛡️ Caller Identity Verification")
    if tm.state == "verified":
        st.markdown('<span class="badge-verified">✓ VERIFIED CALLER</span>', unsafe_allow_html=True)
    elif tm.state == "pending_verification_proof":
        st.markdown('<span class="badge-pending">⏳ PENDING PHONE LAST-4 OR PINCODE</span>', unsafe_allow_html=True)
    elif tm.state == "failed":
        st.markdown('<span class="badge-failed">✗ VERIFICATION FAILED (TERMINAL)</span>', unsafe_allow_html=True)
    else:
        st.markdown("`No active order authentication session`")

    if tm.verified_order:
        st.markdown("#### 📦 Verified Order Details")
        st.json({
            "order_id": tm.verified_order.get("order_id"),
            "customer_name": tm.verified_order.get("customer_first_name"),
            "status": tm.verified_order.get("status"),
            "total_inr": tm.verified_order.get("total_inr"),
            "is_plus_member": tm.verified_order.get("is_plus"),
            "delivery_date": tm.verified_order.get("delivery_date"),
            "items_count": len(tm.verified_order.get("items", []))
        })

    # 2. Retrieved Document Sections
    st.markdown("### 📚 Retrieved Knowledge Chunks")
    ev_ids = st.session_state.agent.last_retrieved_evidence
    if ev_ids:
        st.markdown(f"**Retrieved Section IDs:** `{', '.join(ev_ids)}`")
        for eid in ev_ids[:3]:
            chunk = st.session_state.retriever.kb.get_chunk_by_section_id(eid)
            if chunk:
                with st.expander(f"[{chunk.section_id}] {chunk.doc_title} - {chunk.heading}"):
                    st.markdown(f"**Status:** `{chunk.status}` | **Precedence:** `{chunk.doc_type}`")
                    st.markdown(chunk.content)
    else:
        st.info("No documents retrieved for current turn (or running in baseline mode).")

    # 3. Created Escalation Tickets
    if tm.tickets_created:
        st.markdown("### 🎫 Escalation Tickets Logged")
        for tkt in tm.tickets_created:
            st.success(f"Ticket ID: **{tkt['ticket_id']}** (created=True)")
