#!/usr/bin/env bash
# ZentroMart VoiceOps 2026 Batch Audio Runner
# Usage: ./run.sh --audio-dir <folder_path> --output-dir ./sample_outputs

set -e

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

$PYTHON_CMD run.py "$@"
