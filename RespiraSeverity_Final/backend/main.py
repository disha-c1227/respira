"""
RespiraSeverity Backend Entrypoint
Exposes the modular FastAPI application defined in backend.app.main.
"""
from backend.app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
