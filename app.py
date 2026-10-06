"""
app.py
------
API service: PDF văn bản -> OCR API -> LLM -> danh sách nhiệm vụ (JSON).

Chạy:
    uvicorn app:app --host 0.0.0.0 --port 8080
Docs (Swagger): http://<host>:8080/docs
"""

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from utils.llm import MODEL
from utils.ocr_client import extract_text, ocr_document
from utils.tasks import extract_tasks

app = FastAPI(title="Trích xuất nhiệm vụ", version="1.0.0")


class Task(BaseModel):
    ten_nhiem_vu: str = Field(description="Tên nhiệm vụ")
    yeu_cau: str = Field(description="Yêu cầu cần hoàn thành")
    don_vi_thuc_hien: str | None = Field(description="Đơn vị thực hiện (mỗi nhiệm vụ đúng 1 đơn vị)")
    don_vi_phoi_hop: list[str] = Field(description="Các đơn vị phối hợp")
    thoi_han: str | None = Field(description="Thời gian cần hoàn thành; null nếu văn bản không nêu")
    can_cu: str | None = Field(description="Vị trí trong văn bản")


class TextRequest(BaseModel):
    text: str = Field(description="Nội dung văn bản (đã OCR)")
    model: str = MODEL


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:  # lỗi OCR/LLM upstream
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/api/v1/tasks", response_model=list[Task])
async def tasks_from_pdf(file: UploadFile = File(..., description="File PDF văn bản")):
    """Upload PDF → OCR → trích xuất danh sách nhiệm vụ."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng.")

    # Giữ tên file gốc khi gửi sang OCR API.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, os.path.basename(file.filename))
        with open(path, "wb") as f:
            f.write(content)
        ocr_result = await run_in_threadpool(_run, ocr_document, path)

    text = extract_text(ocr_result)
    return await run_in_threadpool(_run, extract_tasks, text)


@app.post("/api/v1/tasks/text", response_model=list[Task])
def tasks_from_text(req: TextRequest):
    """Trích xuất danh sách nhiệm vụ từ văn bản đã có sẵn (bỏ qua bước OCR)."""
    return _run(extract_tasks, req.text, model=req.model)
