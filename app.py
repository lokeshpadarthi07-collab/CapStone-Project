"""Interactive Web Interface & Demo Dashboard for ZentroMart VoiceOps 2026.

Run locally with:
    streamlit run app.py

Deployable on Vercel via exported WSGI entrypoint (`app` / `handler`).
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


# =========================================================================
# 1. STREAMLIT APPLICATION (Runs when invoked with `streamlit run app.py`)
# =========================================================================
def run_streamlit():
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


# Run Streamlit UI if in Streamlit execution context
if hasattr(st, "runtime") and st.runtime.exists():
    run_streamlit()


# =========================================================================
# 2. VERCEL SERVERLESS ENTRYPOINT EXPORT (Fixes PYTHON_ENTRYPOINT_NOT_FOUND)
# =========================================================================
_vercel_agent_instance = None

def _get_vercel_agent():
    global _vercel_agent_instance
    if _vercel_agent_instance is None:
        _vercel_agent_instance = ZentroMartAgent(mode="rag")
    return _vercel_agent_instance

_VERCEL_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ZentroMart VoiceOps 2026 - AI Support Agent</title>
    <style>
        :root { --primary: #2563EB; --bg: #F8FAFC; --card: #FFFFFF; --text: #1E293B; --border: #E2E8F0; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
        .container { max-width: 860px; margin: 0 auto; }
        .header { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .header h1 { margin: 0 0 8px 0; font-size: 1.8rem; color: #0F172A; }
        .header p { margin: 0; color: #64748B; font-size: 0.95rem; }
        .badges { display: flex; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
        .badge { background: #E0E7FF; color: #3730A3; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; }
        .badge-green { background: #DCFCE7; color: #166534; }
        .chat-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .chat-box { min-height: 250px; max-height: 420px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 16px; background: #FAFAFA; display: flex; flex-direction: column; gap: 12px; }
        .msg { padding: 10px 14px; border-radius: 8px; max-width: 80%; line-height: 1.5; font-size: 0.95rem; }
        .msg-user { background: #2563EB; color: #FFF; align-self: flex-end; }
        .msg-assistant { background: #E2E8F0; color: #0F172A; align-self: flex-start; }
        .input-bar { display: flex; gap: 8px; }
        .input-bar input { flex: 1; padding: 12px 16px; border: 1px solid var(--border); border-radius: 8px; font-size: 0.95rem; }
        .input-bar button { background: var(--primary); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; }
        .quick-chips { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
        .chip { background: #F1F5F9; border: 1px solid #CBD5E1; padding: 6px 12px; border-radius: 6px; font-size: 0.8rem; cursor: pointer; color: #334155; }
        .chip:hover { background: #E2E8F0; }
        .downloads { display: flex; gap: 12px; margin-top: 18px; }
        .btn-download { display: inline-flex; align-items: center; gap: 6px; padding: 10px 16px; border-radius: 8px; font-weight: 600; text-decoration: none; font-size: 0.85rem; border: 1px solid #CBD5E1; color: #334155; background: #FFF; }
        .btn-download:hover { background: #F1F5F9; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ ZentroMart AI Customer Support Agent</h1>
            <p>LLM & Generative AI Course Capstone | Consulting & Analytics Club, IIT Guwahati × Gnani.ai</p>
            <div class="badges">
                <span class="badge badge-green">✓ Active Vercel Cloud Serverless Function</span>
                <span class="badge">Reference Date: 20 October 2026</span>
                <span class="badge">Team: Team_VoiceOps_2026</span>
            </div>
            <div class="downloads">
                <a href="/report.pdf" class="btn-download" download>📄 Download Report PDF</a>
                <a href="/team_voiceops_2026_voiceops.zip" class="btn-download" download>📦 Download Submission ZIP</a>
            </div>
        </div>

        <div class="chat-card">
            <h3>💬 Grounded Customer Support Chat</h3>
            <div class="chat-box" id="chatBox">
                <div class="msg msg-assistant">Hello! I am ZentroMart's AI customer support assistant. How may I help you today? You can ask about order returns, refund timelines, or enter an Order ID.</div>
            </div>
            <form id="chatForm" onsubmit="sendQuery(event)">
                <div class="input-bar">
                    <input type="text" id="userInput" placeholder="Ask a question or enter order ID (e.g. 8978 2837)..." autocomplete="off" required>
                    <button type="submit" id="sendBtn">Send</button>
                </div>
            </form>
            <div class="quick-chips">
                <span style="font-size: 0.8rem; color: #64748B; align-self: center;">Quick Questions:</span>
                <button type="button" class="chip" onclick="ask('How fast are refunds to the ZentroMart Wallet?')">Wallet Refund Speed</button>
                <button type="button" class="chip" onclick="ask('Can I schedule a delivery at midnight?')">Midnight Delivery</button>
                <button type="button" class="chip" onclick="ask('Order 8978 2837. Last 4 digits 4533. Damaged TV.')">Damaged TV Claim</button>
                <button type="button" class="chip" onclick="ask('Maine Gulmohar Matte Lipstick khol liya hai, kya wapas hoga?')">Opened Lipstick (Hindi)</button>
            </div>
        </div>
    </div>

    <script>
        async function sendQuery(e) {
            if (e) e.preventDefault();
            const inp = document.getElementById('userInput');
            const txt = inp.value.trim();
            if (!txt) return;
            inp.value = '';

            const box = document.getElementById('chatBox');
            box.innerHTML += `<div class="msg msg-user">${escapeHtml(txt)}</div>`;
            box.scrollTop = box.scrollHeight;

            const btn = document.getElementById('sendBtn');
            btn.disabled = true;
            btn.innerText = 'Thinking...';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: txt})
                });
                const data = await res.json();
                box.innerHTML += `<div class="msg msg-assistant">${escapeHtml(data.reply || 'No response')}</div>`;
            } catch (err) {
                box.innerHTML += `<div class="msg msg-assistant" style="color: #DC2626;">Error communicating with agent: ${err.message}</div>`;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Send';
                box.scrollTop = box.scrollHeight;
            }
        }

        function ask(text) {
            document.getElementById('userInput').value = text;
            sendQuery();
        }

        function escapeHtml(t) {
            const div = document.createElement('div');
            div.textContent = t;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""

def app(environ, start_response):
    """WSGI callable exported for Vercel Serverless Function runtime."""
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET").upper()

    if path == "/" or path == "/index.html":
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [_VERCEL_HTML.encode("utf-8")]

    if path == "/api/chat" and method == "POST":
        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(content_length).decode("utf-8")
            data = json.loads(body)
            msg = data.get("message", "")
            agent = _get_vercel_agent()
            reply = agent.respond(msg)
            resp = json.dumps({"reply": reply, "evidence": getattr(agent, "last_retrieved_evidence", [])})
            start_response("200 OK", [("Content-Type", "application/json; charset=utf-8")])
            return [resp.encode("utf-8")]
        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "application/json")])
            return [json.dumps({"error": str(e)}).encode("utf-8")]

    if path == "/report.pdf":
        p = Path(__file__).resolve().parent / "report.pdf"
        if p.exists():
            start_response("200 OK", [("Content-Type", "application/pdf"), ("Content-Disposition", "inline; filename=report.pdf")])
            return [p.read_bytes()]

    if path == "/team_voiceops_2026_voiceops.zip":
        p = Path(__file__).resolve().parent / "team_voiceops_2026_voiceops.zip"
        if p.exists():
            start_response("200 OK", [("Content-Type", "application/zip"), ("Content-Disposition", "attachment; filename=team_voiceops_2026_voiceops.zip")])
            return [p.read_bytes()]

    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


# Vercel entrypoint aliases
handler = app
application = app
