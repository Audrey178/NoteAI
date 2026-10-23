

import asyncio
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.docx_reader import extract_docx_text
from utils.llm import MODEL
from utils.ocr_client import extract_text, ocr_document
from utils.pdf_text import read_pdf_text_layer
from utils.tasks import extract_tasks
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Trích xuất nhiệm vụ",
    version="1.0.0",
    description="Trích xuất danh sách nhiệm vụ từ văn bản hành chính (PDF / .docx). "
                "Tài liệu đầy đủ: README.md, mục API service.",
)

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Task(BaseModel):
    task_name: str = Field(description="Tên nhiệm vụ")
    requirement: str = Field(description="Yêu cầu cần hoàn thành")
    executing_unit: str | None = Field(description="Đơn vị thực hiện (mỗi nhiệm vụ đúng 1 đơn vị)")
    coordinating_units: list[str] = Field(description="Các đơn vị phối hợp")
    deadline: str | None = Field(description="Thời gian cần hoàn thành; null nếu văn bản không nêu")
    reference: str | None = Field(description="Vị trí trong văn bản")


class TextRequest(BaseModel):
    text: str = Field(description="Nội dung văn bản (đã OCR)")
    model: str = MODEL


# Số văn bản xử lý đồng thời; mặc định 1 vì llama.cpp chạy CPU với --parallel 1.
# Đếm trong 1 process nên API phải chạy 1 worker uvicorn (xem Dockerfile).
_jobs = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_JOBS", "1")))


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:  # lỗi OCR/LLM upstream
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/health", summary="Kiểm tra trạng thái API")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/v1/tasks", response_model=list[Task], summary="Trích xuất nhiệm vụ từ file PDF / .docx")
async def tasks_from_document(file: UploadFile = File(..., description="File văn bản PDF hoặc Word (.docx)")):
    """
    Upload một văn bản hành chính, trả về mảng nhiệm vụ. Chi tiết: README.md, mục API service.
    """
    filename = (file.filename or "").lower()
    if filename.endswith(".doc"):
        raise HTTPException(status_code=415, detail="Định dạng .doc cũ chưa hỗ trợ, hãy lưu lại thành .docx.")
    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF hoặc Word (.docx).")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng.")

    # Từ chối ngay thay vì xếp hàng: request chờ sau một request khác (~30 phút trên CPU) dễ bị client hoặc
    # proxy ngắt giữa chừng, trong khi server vẫn phải chạy hết hàng đợi.
    if _jobs.locked():
        raise HTTPException(status_code=503, headers={"Retry-After": "600"},
                            detail="Server đang xử lý văn bản khác, hãy thử lại sau.")
    async with _jobs:
        return await _extract_tasks_from_file(file.filename, content)


async def _extract_tasks_from_file(filename: str, content: bytes) -> list[dict]:
    if filename.lower().endswith(".docx"):
        try:
            text = await run_in_threadpool(extract_docx_text, content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        # Giữ tên file gốc khi gửi sang OCR API.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, os.path.basename(filename))
            with open(path, "wb") as f:
                f.write(content)
            # PDF có sẵn lớp text thì không cần OCR
            ocr_result = await run_in_threadpool(read_pdf_text_layer, path)
            if ocr_result is None:
                ocr_result = await run_in_threadpool(_run, ocr_document, path)
        text = extract_text(ocr_result)

    if not text:
        raise HTTPException(status_code=400, detail="Không trích xuất được nội dung văn bản.")
    return await run_in_threadpool(_run, extract_tasks, text)


