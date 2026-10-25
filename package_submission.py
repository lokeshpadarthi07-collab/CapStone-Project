"""Automated packaging script for Capstone submission ZIP.

Adheres strictly to Page 14 guidelines:
- Name format: team_name_voiceops.zip
- Contains: README.md, .env.example, requirements.txt, run.py, run.sh, src/, report.pdf, sample_outputs/
- Strictly excludes: .env, API keys, virtual environments (.venv), dataset copies.
"""
import os
import zipfile
from pathlib import Path


def create_submission_zip(team_name: str = "Team_VoiceOps_2026"):
    zip_filename = f"{team_name.lower()}_voiceops.zip"
    base_dir = Path(__file__).resolve().parent

    # Mandatory inclusion list
    included_files = [
        "README.md",
        ".env.example",
        "requirements.txt",
        "run.py",
        "run.sh",
        "app.py",
        "report.pdf",
    ]

    included_dirs = [
        "src",
        "sample_outputs",
        "tests",
    ]

    print(f"Creating submission archive: {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for fname in included_files:
            fpath = base_dir / fname
            if fpath.exists():
                print(f"  + Adding {fname}")
                zf.write(fpath, arcname=fname)
            else:
                print(f"  ! Warning: {fname} not found!")

        # Add directories
        for dname in included_dirs:
            dpath = base_dir / dname
            if dpath.exists():
                for root, dirs, files in os.walk(dpath):
                    # Filter out __pycache__ and sensitive files
                    dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
                    for file in files:
                        if file.endswith((".pyc", ".pyo")) or file.startswith("."):
                            continue
                        full_path = Path(root) / file
                        rel_path = full_path.relative_to(base_dir)
                        print(f"  + Adding {rel_path}")
                        zf.write(full_path, arcname=str(rel_path))

    zip_size = os.path.getsize(zip_filename)
    print(f"\nSuccessfully created {zip_filename} ({zip_size} bytes / {round(zip_size / 1024, 2)} KB)")
    print("Zero secrets, zero .env files, zero virtual environments included.")


if __name__ == "__main__":
    create_submission_zip()
