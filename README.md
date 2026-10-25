# VoiceOps 2026: ZentroMart AI Customer Support Agent

> **LLM & Generative AI Course Capstone**  
> *Consulting & Analytics Club, IIT Guwahati × Gnani.ai*  
> **Team:** Team_VoiceOps_2026 | **Reference Date:** 20 October 2026

An enterprise-ready, voice-enabled customer support agent for **ZentroMart** (a fictional Indian e-commerce store selling groceries and electronics). Grounded in 21 company knowledge-base documents, the agent enforces strict document precedence, calculates time-sensitive business-day rules (skipping Sundays and warehouse holidays), executes caller identity verification before revealing order details, and communicates natively through **Gnani.ai's** speech-to-text (STT) and text-to-speech (TTS) models.

---

## 🌟 Key Capabilities & Highlights

1. **Grounded RAG with Document Precedence:**
   - Enforces the capstone hierarchy: `Current Policy (1) > Agent Guide (2) > Help (3) > FAQ (4) > Marketing (5)`.
   - Prevents policy trap failures (e.g., prioritizes `pol_refunds.md [REF-2]` instant wallet refund over `faq_general.md` outdated 5–7 day timeline).
   - Re-ranks superseded documents (e.g., `pol_returns_v4.md` overrides `pol_returns_v3_superseded.md`).
2. **Fixed Calendar & Business-Day Engine:**
   - Uses fixed reference date **20 October 2026** for all delay, refund, and deadline math.
   - Business days run Monday–Saturday, skipping Sundays and festival warehouse holidays in `data/holidays.csv` (15 Aug, 02 Oct, 19 Oct, 09 Nov 2026).
   - Return windows are counted in calendar days from delivery date.
3. **Two-Phase Caller Identity Verification:**
   - Withholds all private order information (status, amounts, items, address) until identity is verified via 4-digit registered phone suffix or 6-digit delivery pincode (`IDV-1`, `IDV-2`).
   - Enforces a maximum of one retry before locking down (`IDV-3`); never reads out secrets aloud (`IDV-4`).
4. **Human Handover & Escalation Protocol:**
   - Detects policy violations, safety risks (battery swelling, sparks), repeat inquiries (`contacts_last_7_days >= 2`), and high-value damage claims (> ₹10,000, `DWM-3`).
   - Generates compliance tickets via `create_ticket()` and confirms handovers only when `created=True`.
5. **Gnani.ai Voice Pipeline:**
   - Speech-to-Text via `GnaniSTTClient` (supporting 10 Indian languages and Hinglish `en-IN`).
   - Text-to-Speech via `GnaniTTSClient` with `timbre-v2.5` model and `Kaveri` voice.
   - Text normalization for the ear: spells order numbers (`8 9 7 8   2 8 3 7`), speaks dates with year (`27th October 2026`), and converts currency (`forty-nine rupees`).
   - Graceful recovery on empty or unclear audio clips.

---

## 📁 Repository Structure

```text
Capstone Project/
├── README.md               # Project overview, architecture, and run commands
├── requirements.txt        # Python dependency manifest
├── .env.example            # Environment variable template (no secrets)
├── .gitignore              # Git ignore rules
├── run.py                  # Official batch runner & interactive CLI
├── run.sh                  # Bash execution wrapper for Unix/macOS
├── app.py                  # Interactive Streamlit Web UI & Live Demo
├── generate_report.py      # Generates official 4-page report.pdf
├── package_submission.py   # One-click submission ZIP packager
├── report.pdf              # 4-page capstone report
├── src/                    # Modular source code
│   ├── config.py           # Constants, reference date (2026-10-20), paths
│   ├── calendar_utils.py   # Business days (Mon-Sat), holidays, calendar days
│   ├── kb_indexer.py       # YAML frontmatter parser, section chunker ([SECTION-ID])
│   ├── retriever.py        # Hybrid RAG retriever with precedence & entity linking
│   ├── tools_layer.py      # Caller verification state machine & ticket creator
│   ├── order_service.py    # Order lookup wrapper
│   ├── conversation.py     # Multi-turn context memory & session state
│   ├── agent.py            # Unified support agent (Baseline + RAG)
│   ├── voice.py            # Gnani STT & TTS integration + text normalizer
│   ├── audio_batch.py      # Batch audio folder processing engine
│   └── evaluation.py       # Automated benchmark suite (Step 1, 2, and 3)
├── tests/                  # Unit and integration test suite
│   └── test_agent_suite.py # 14 automated unit tests
├── sample_outputs/         # Sample audio outputs, transcripts, and evaluation logs
└── dataset/                # Extracted capstone dataset (CSVs, KB, practice)
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/zentromart-voiceops.git
cd zentromart-voiceops

# Create and activate a Python virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to provide your API keys:

```ini
# Gnani.ai API Key from app.gnani.ai/voice
GNANI_API_KEY=your_gnani_api_key_here
GNANI_TTS_MODEL=timbre-v2.5
GNANI_VOICE=Kaveri

# Primary LLM Provider: 'gemini', 'openai', or 'offline'
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

*(Note: If no API key is provided, the system seamlessly defaults to the built-in Deterministic Grounded Policy Engine, allowing full offline testing without crashes.)*

---

## 🎧 Official Run Command (For Evaluators & Organizers)

To batch process a directory of customer audio files, transcribe them with Gnani STT, run the RAG agent, and synthesize speech responses with Gnani TTS:

```bash
python run.py --audio-dir ./sample_outputs/input_audio --output-dir ./sample_outputs/batch_run
```

**Outputs generated in `--output-dir`:**
- `reply_<filename>.wav`: Spoken speech response synthesized with Gnani TTS.
- `batch_run_summary.json`: Complete machine-readable log containing input filename, transcript, reply text, audio path, processing latency, and retrieved `[SECTION-ID]` chunks.

---

## 💬 Interactive Modes

### 1. Terminal Interactive Support Chat

```bash
python run.py --interactive
```

Interact directly with the ZentroMart agent in multi-turn conversation mode with live verification and policy grounding.

### 2. Interactive Web Demo Dashboard

```bash
streamlit run app.py
```

Features:
- Live text and voice chat.
- Audio recording and playback.
- Real-time **Brain Inspector** displaying retrieved `[SECTION-ID]` chunks and document precedence.
- Real-time **Caller Verification State** and order lookup viewer.
- Live escalation ticket log.

---

## 📊 Benchmark Evaluation Suite

Run the automated evaluation across all 52 single questions and 23 conversations (96 customer turns):

```bash
python run.py --evaluate
```

### Measured Evaluation Results

| Evaluation Metric | Step 1: Baseline Prompt-Only | Step 2: Grounded RAG | Step 3: Multi-Turn Conversations |
| :--- | :---: | :---: | :---: |
| **Test Cases** | 52 questions | 52 questions | 23 conversations (96 turns) |
| **Decision Accuracy** | 90.38% | **100.0% (52/52)** | **80.21%** |
| **Must-Include Adherence** | 7.69% | **86.54%** | Passed required facts |
| **Must-Not-Include Compliance** | Failed (Hallucinated dates/fees) | **100.0% (Zero violations)** | Refused policy breaches |
| **Gold Evidence Recall** | 0.0% (No retrieval) | **65.96%** | Cites exact `[SECTION-ID]` |
| **Tool Execution Accuracy** | N/A | N/A | **100.0% (Zero Leaks)** |
| **Identity Verification Gate** | N/A | N/A | **100.0% Verified** |

Detailed evaluation results are saved in `sample_outputs/evaluation_summary.json` and documented in `report.pdf`.

---

## 🧪 Automated Unit Tests

Run the comprehensive unit test suite covering all 14 capstone requirements:

```bash
python -m unittest tests/test_agent_suite.py
```

*Results: 14 tests run in ~2.0s, 100% PASS.*

---

## 📦 Packaging Submission ZIP

To package the deliverables strictly according to Page 14 of the Capstone Brief:

```bash
python package_submission.py
```

Creates `team_voiceops_2026_voiceops.zip` containing:
- `README.md`
- `.env.example`
- `requirements.txt`
- `run.py` and `run.sh`
- `src/` modular source code
- `report.pdf`
- `sample_outputs/`
- *(Strictly excludes `.env`, secrets, `.venv/`, and dataset copies)*

---

## 📜 Capstone Compliance Checklist

- [x] Grounded strictly in `kb/` documents with `[SECTION-ID]` citation.
- [x] Document precedence: Policy (current) > Agent Guide > Help > FAQ > Marketing.
- [x] Superseded policy penalized (`pol_returns_v4.md` overrides `v3`).
- [x] Fixed date **20 October 2026** for all calculations.
- [x] Mon–Sat business days; skips Sundays and listed warehouse holidays.
- [x] Identity verification enforced before sharing order details (`IDV-1` to `IDV-4`).
- [x] Mock tools integrated (`lookup_order`, `create_ticket` checking `created=True`).
- [x] High-value damage (> ₹10,000) and safety hazards escalated (`ESC-1`, `DWM-3`).
- [x] Gnani.ai official SDK (`gnani-vachana`) STT & TTS with `timbre-v2.5` / `Kaveri`.
- [x] Text normalization for the ear (spaced order numbers, spoken currency, dates with year).
- [x] Zero hardcoding; works dynamically across unseen questions.
- [x] 4-page Capstone Report PDF generated (`report.pdf`).
