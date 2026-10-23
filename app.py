

import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.docx_reader import extract_docx_text
from utils.job_queue import Job, JobQueue, QueueFull
from utils.llm import MODEL
from utils.ocr_client import extract_text, ocr_document
from utils.pdf_text import read_pdf_text_layer
from utils.tasks import extract_tasks
from dotenv import load_dotenv

load_dotenv()

# Số văn bản xử lý đồng thời; mặc định 1 vì llama.cpp chạy CPU với --parallel 1.
# Hàng đợi nằm trong 1 process nên API phải chạy 1 worker uvicorn (xem Dockerfile).
jobs = JobQueue(
    workers=int(os.getenv("MAX_CONCURRENT_JOBS", "1")),
    max_queued=int(os.getenv("MAX_QUEUED_JOBS", "20")),
    ttl=float(os.getenv("JOB_TTL_SECONDS", "86400")),
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    jobs.start()
    yield
    await jobs.stop()


app = FastAPI(
    lifespan=lifespan,
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


class JobError(BaseModel):
    status_code: int = Field(description="Mã lỗi tương ứng nếu gọi đồng bộ (400 / 502 / 500)")
    detail: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    filename: str
    position: int | None = Field(description="Số job đang chờ phía trước (0 = chạy kế tiếp); null khi không còn chờ")
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: list[Task] | None = Field(description="Danh sách nhiệm vụ khi status = done")
    error: JobError | None = Field(description="Lý do lỗi khi status = failed")


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:  # lỗi OCR/LLM upstream
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/health", summary="Kiểm tra trạng thái API")
def health():
    return {"status": "ok", "model": MODEL, "busy": jobs.is_busy(), "queued_jobs": jobs.queued_count()}


async def _read_upload(file: UploadFile) -> bytes:
    filename = (file.filename or "").lower()
    if filename.endswith(".doc"):
        raise HTTPException(status_code=415, detail="Định dạng .doc cũ chưa hỗ trợ, hãy lưu lại thành .docx.")
    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF hoặc Word (.docx).")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng.")
    return content


def _submit(filename: str, content: bytes) -> Job:
    try:
        return jobs.submit(filename, lambda: _extract_tasks_from_file(filename, content))
    except QueueFull:
        raise HTTPException(status_code=503, headers={"Retry-After": "600"},
                            detail="Hàng đợi đã đầy, hãy thử lại sau.") from None


def _ts(t: float | None) -> datetime | None:
    return datetime.fromtimestamp(t, timezone.utc) if t else None


def _job_status(job: Job) -> JobStatus:
    return JobStatus(
        job_id=job.id, status=job.status, filename=job.filename, position=jobs.position(job),
        created_at=_ts(job.created_at), started_at=_ts(job.started_at), finished_at=_ts(job.finished_at),
        result=job.result, error=job.error,
    )


@app.post("/v1/jobs", response_model=JobStatus, status_code=202,
          summary="Gửi file PDF / .docx vào hàng đợi, trả job_id ngay")
async def create_job(response: Response,
                     file: UploadFile = File(..., description="File văn bản PDF hoặc Word (.docx)")):
    """
    Nhận file và xếp hàng xử lý; lấy kết quả qua `GET /v1/jobs/{job_id}`. Chi tiết: README.md, mục API service.
    """
    job = _submit(file.filename, await _read_upload(file))
    response.headers["Location"] = f"/v1/jobs/{job.id}"
    return _job_status(job)


@app.get("/v1/jobs/{job_id}", response_model=JobStatus, summary="Trạng thái / kết quả của job")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job (sai id, hoặc đã hết hạn lưu).")
    return _job_status(job)


@app.post("/v1/tasks", response_model=list[Task], summary="Trích xuất nhiệm vụ từ file PDF / .docx (đồng bộ)")
async def tasks_from_document(file: UploadFile = File(..., description="File văn bản PDF hoặc Word (.docx)")):
    """
    Upload một văn bản hành chính, chờ và trả về mảng nhiệm vụ. Văn bản dài nên dùng `POST /v1/jobs`.
    Chi tiết: README.md, mục API service.
    """
    content = await _read_upload(file)
    # Không xếp hàng ở chế độ đồng bộ: chờ sau job khác (~30 phút/job trên CPU) dễ bị client hoặc proxy ngắt.
    if jobs.is_busy():
        raise HTTPException(status_code=503, headers={"Retry-After": "600"},
                            detail="Server đang xử lý văn bản khác, hãy dùng POST /v1/jobs hoặc thử lại sau.")
    job = _submit(file.filename, content)
    await job.done.wait()
    if job.error:
        raise HTTPException(status_code=job.error["status_code"], detail=job.error["detail"])
    return job.result


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


