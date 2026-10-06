

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
    task_name: str = Field(description="Tên nhiệm vụ")
    requirement: str = Field(description="Yêu cầu cần hoàn thành")
    executing_unit: str | None = Field(description="Đơn vị thực hiện (mỗi nhiệm vụ đúng 1 đơn vị)")
    coordinating_units: list[str] = Field(description="Các đơn vị phối hợp")
    deadline: str | None = Field(description="Thời gian cần hoàn thành; null nếu văn bản không nêu")
    reference: str | None = Field(description="Vị trí trong văn bản")


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
    """
    Upload PDF → OCR → trích xuất danh sách nhiệm vụ.

    **Input:** multipart/form-data, trường `file` là file PDF văn bản hành chính.

    **Output:** mảng JSON, mỗi phần tử là một nhiệm vụ:

    - `task_name` (string): tên ngắn gọn của nhiệm vụ (động từ + đối tượng),
      vd "Thẩm định hồ sơ điều chỉnh cục bộ quy hoạch chung".
    - `requirement` (string): yêu cầu cần hoàn thành — làm gì, về vấn đề gì, cho đối tượng nào,
      sản phẩm/kết quả đầu ra; bám sát câu chữ văn bản.
    - `executing_unit` (string | null): đơn vị thực hiện/chủ trì. Mỗi nhiệm vụ có đúng 1 đơn vị;
      null nếu văn bản không xác định được.
    - `coordinating_units` (string[]): các đơn vị phối hợp; [] nếu không có.
      Khi văn bản nêu chung chung ("các cơ quan liên quan"...), đơn vị được suy luận từ knowledge base.
    - `deadline` (string | null): thời gian cần hoàn thành, giữ nguyên cách ghi trong văn bản
      (vd "Trong tháng 6 năm 2020"); null nếu văn bản không nêu.
    - `reference` (string | null): vị trí trong văn bản, vd "Mục III, khoản 10".

    **Lỗi:** 415 không phải PDF, 400 file rỗng, 502 lỗi OCR API / LLM (xem `detail`).
    """
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


