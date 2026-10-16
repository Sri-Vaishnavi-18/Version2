from __future__ import annotations

import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from contract_lens.data import LABELS
from contract_lens.inference import (
    LABEL_DETAILS,
    MAX_TEXT_CHARACTERS,
    ClausePredictor,
    analyze_text,
)
from contract_lens.sources import extract_pdf_text, fetch_policy_url

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("CONTRACT_LENS_ARTIFACTS", ROOT / "artifacts"))
FRONTEND_DIST = ROOT / "pactio-frontend" / "pactio-frontend" / "dist"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MODEL_NAME = "bilstm_attention"

app = FastAPI(
    title="Pactio Policy Analysis API",
    description="Evidence-linked policy topic analysis using the trained BiLSTM + Attention model.",
    version="1.0.0",
)
allowed_origins = [
    origin.strip()
    for origin in os.environ.get(
        "PACTIO_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    url: str | None = None


@lru_cache(maxsize=1)
def get_predictor() -> ClausePredictor:
    return ClausePredictor(ARTIFACTS, MODEL_NAME)


def _analyze(text: str, document: str) -> dict[str, Any]:
    if len(text) > MAX_TEXT_CHARACTERS:
        raise HTTPException(
            status_code=413,
            detail=f"Policy text exceeds the {MAX_TEXT_CHARACTERS:,}-character limit.",
        )
    try:
        result = analyze_text(text, get_predictor())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    grouped = {
        label: {
            "key": label,
            "label": LABEL_DETAILS[label]["name"],
            "description": LABEL_DETAILS[label]["plain_english"],
            "count": 0,
            "clauses": [],
        }
        for label in LABELS
    }
    for finding in result["findings"]:
        group = grouped[finding["label"]]
        group["count"] += 1
        group["clauses"].append(
            {
                "text": finding["evidence"],
                "evidence": finding["evidence"],
                "explanation": finding["plain_english"],
                "review_prompt": finding["review_prompt"],
                "score": finding["score"],
                "key_terms": finding["key_terms"],
            }
        )

    return {
        "model": MODEL_NAME,
        "document": document,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "clause_count": result["clause_count"],
        "topics": list(grouped.values()),
        "summary": result["summary"],
        "key_terms": result["key_terms"],
        "disclaimer": (
            "Research prototype; not legal advice or a compliance decision. "
            "Model scores are not calibrated certainty."
        ),
    }


@app.get("/api/health")
def health() -> dict[str, bool | str]:
    try:
        get_predictor()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_ready": True,
    }


@app.post("/api/analyze")
def analyze_policy(request: AnalysisRequest) -> dict[str, Any]:
    has_text = request.text is not None
    has_url = request.url is not None
    if has_text == has_url:
        raise HTTPException(
            status_code=422, detail="Provide exactly one of 'text' or 'url'."
        )
    if has_text:
        return _analyze(request.text or "", "pasted-policy")

    url = (request.url or "").strip()
    if not url:
        raise HTTPException(status_code=422, detail="Policy URL cannot be empty.")
    try:
        text = fetch_policy_url(url)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return _analyze(text, url)


@app.post("/api/analyze-file")
def analyze_file(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Upload a PDF or TXT file.")

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413, detail="Uploaded files must be 20 MB or smaller."
        )
    if suffix == ".pdf":
        try:
            text = extract_pdf_text(content)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    else:
        try:
            text = content.decode("utf-8-sig").strip()
        except UnicodeDecodeError as error:
            raise HTTPException(
                status_code=400, detail="TXT uploads must use UTF-8 encoding."
            ) from error
    return _analyze(text, filename)


if FRONTEND_DIST.is_dir() and (FRONTEND_DIST / "index.html").is_file():
    app.mount(
        "/",
        StaticFiles(directory=FRONTEND_DIST, html=True),
        name="pactio-frontend",
    )
else:

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "name": "Pactio",
            "message": "Build the Vite frontend to serve the web application here.",
            "api_docs": "/docs",
        }
