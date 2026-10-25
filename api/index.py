import sys
from pathlib import Path

# Ensure root directory is available on sys.path for serverless environment
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from app import app, handler, application
