"""Batch Audio Processing Engine for ZentroMart VoiceOps.

Processes an input directory of customer audio files:
1. Audio in -> Gnani STT (REST endpoint)
2. Transcript -> ZentroMart Conversational Agent (RAG + Tools + Identity Verification)
3. Text reply -> Normalization for speech
4. Spoken reply -> Gnani TTS (timbre-v2.5, voice: Kaveri, output: reply_*.wav)
5. Saves machine-readable JSON logs with retrieved document sections ([SECTION-ID])
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.agent import ZentroMartAgent
from src.voice import VoicePipeline


def process_audio_batch(input_dir: str, output_dir: str, language_code: str = "en-IN",
                        tts_language: Optional[str] = None) -> Dict[str, Any]:
    """Process all audio files in input_dir and write text, audio, and metadata into output_dir.

    Supports cross-lingual listening and replying (e.g. listening in gu-IN and speaking in hi-IN).
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    agent = ZentroMartAgent(mode="rag")
    voice = VoicePipeline()
    tts_lang = tts_language or language_code

    audio_files = []
    for ext in ("*.wav", "*.mp3", "*.ogg", "*.flac", "*.m4a"):
        audio_files.extend(list(in_path.glob(ext)))
    audio_files.sort()

    print(f"Discovered {len(audio_files)} audio file(s) in {input_dir}")
    batch_records = []

    for idx, audio_file in enumerate(audio_files, start=1):
        file_stem = audio_file.stem
        start_time = time.time()
        print(f"[{idx}/{len(audio_files)}] Processing {audio_file.name}...")

        # 1. Transcribe audio with Gnani STT
        heard = voice.transcribe(str(audio_file), language_code=language_code)

        # 2. Agent response with RAG
        if not heard.strip():
            reply_text = "Sorry, I didn't catch that. Could you say it again?"
            retrieved_chunks = []
        else:
            reply_text = agent.respond(heard)
            retrieved_chunks = list(agent.last_retrieved_evidence)

        # 3. Synthesize reply with Gnani TTS
        out_wav_name = f"reply_{file_stem}.wav"
        out_wav_path = out_path / out_wav_name
        voice.synthesize(reply_text, str(out_wav_path), language=tts_lang)

        elapsed = round(time.time() - start_time, 3)

        record = {
            "index": idx,
            "input_file": audio_file.name,
            "transcript": heard,
            "reply_text": reply_text,
            "reply_audio_file": out_wav_name,
            "retrieved_evidence_sections": retrieved_chunks,
            "processing_time_sec": elapsed
        }
        batch_records.append(record)

    # Save summary report
    summary = {
        "total_files": len(audio_files),
        "processed_files": len(batch_records),
        "output_directory": str(out_path),
        "results": batch_records
    }
    summary_file = out_path / "batch_run_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBatch processing complete! Summary written to: {summary_file}")
    return summary
