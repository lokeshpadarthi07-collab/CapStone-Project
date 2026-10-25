"""ZentroMart VoiceOps 2026 - Main Batch Runner and Interactive CLI.

Official Run Command for Organizers & Evaluators:
    python run.py --audio-dir <folder_of_audio_files> --output-dir ./sample_outputs

Interactive Terminal Mode:
    python run.py --interactive

Benchmark Evaluation:
    python run.py --evaluate
"""
import sys
import argparse
from pathlib import Path
from src.audio_batch import process_audio_batch
from src.agent import ZentroMartAgent
from src.evaluation import run_full_capstone_evaluation


def interactive_cli():
    print("=" * 65)
    print("  ZentroMart AI Customer Support Agent (VoiceOps 2026)")
    print("  Type 'exit', 'quit', or 'reset' to manage your session.")
    print("=" * 65)

    agent = ZentroMartAgent(mode="rag")
    session_active = True

    while session_active:
        try:
            user_msg = input("\nCustomer > ").strip()
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit"):
                print("Ending support session. Goodbye!")
                break
            if user_msg.lower() == "reset":
                agent.reset_conversation()
                print("[Session reset. State cleared.]")
                continue

            reply = agent.respond(user_msg)
            print(f"Assistant > {reply}")
            if agent.last_retrieved_evidence:
                print(f"  [Retrieved Sections: {', '.join(agent.last_retrieved_evidence)}]")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="ZentroMart VoiceOps 2026 - Official Batch Audio & Conversation Runner"
    )
    parser.add_argument(
        "--audio-dir", "--input-dir", "-a",
        dest="audio_dir",
        type=str,
        default=None,
        help="Path to folder containing customer audio files (.wav, .mp3, etc.)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        type=str,
        default="./sample_outputs",
        help="Path to save speech responses, transcripts, and retrieval evidence"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en-IN",
        help="Input language code for Gnani STT (10 Indian languages, e.g. en-IN, hi-IN, gu-IN; default: en-IN)"
    )
    parser.add_argument(
        "--tts-language",
        type=str,
        default=None,
        help="Reply language code for Gnani TTS (e.g. hi-IN, hi-en for Hinglish). Defaults to --language."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive terminal support conversation mode"
    )
    parser.add_argument(
        "--evaluate", "-e",
        action="store_true",
        help="Execute automated evaluation across all 52 questions and 23 conversations"
    )

    args = parser.parse_args()

    if args.evaluate:
        run_full_capstone_evaluation(Path(args.output_dir))
        return

    if args.interactive or (not args.audio_dir and len(sys.argv) == 1):
        interactive_cli()
        return

    if args.audio_dir:
        audio_path = Path(args.audio_dir)
        if not audio_path.exists():
            print(f"Error: Specified audio directory does not exist: {audio_path}")
            sys.exit(1)
        process_audio_batch(
            input_dir=str(audio_path),
            output_dir=args.output_dir,
            language_code=args.language,
            tts_language=args.tts_language
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
