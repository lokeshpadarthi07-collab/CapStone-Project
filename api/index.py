import sys
import traceback
from pathlib import Path

# Ensure root directory is available on sys.path for serverless environment
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from app import app, handler, application
except Exception as e:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    err_tb = traceback.format_exc()
    print("STARTUP ERROR:", err_tb)
    app = FastAPI(title="ZentroMart Error")

    @app.get("/{path:path}", response_class=HTMLResponse)
    def catch_all(path: str = ""):
        return HTMLResponse(
            content=f"<h2>Application Startup Error</h2><pre>{err_tb}</pre>",
            status_code=500
        )

    handler = app
    application = app
