

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.docx_reader import extract_docx_text
from utils.llm import MODEL
from utils.ocr_client import extract_text, ocr_document
from utils.tasks import extract_tasks
from dotenv import load_dotenv

load_dotenv()

API_DESCRIPTION = """
Đọc văn bản hành chính (PDF scan hoặc Word) và trích xuất **danh sách nhiệm vụ được giao**:
đơn vị nào thực hiện, đơn vị nào phối hợp, yêu cầu gì, thời hạn nào, căn cứ ở đâu trong văn bản.

## Luồng xử lý

```
PDF  ──► OCR ──┐
               ├──► văn bản ──► LLM + knowledge base ──► list[Task]
.docx ─────────┘   (đọc trực tiếp, không qua OCR)
```

- **OCR**: dịch vụ cấu hình qua `OCR_ENDPOINT` (self-host: PaddleOCR-VL-1.6).
- **LLM**: endpoint OpenAI-compatible cấu hình qua `OPENAI_BASE_URL` (self-host: llama.cpp).
- **Knowledge base** (`knowledge_base.json`): phạm vi phụ trách của các cơ quan, dùng để suy luận
  đích danh đơn vị khi văn bản chỉ nêu chung chung ("các cơ quan liên quan"...).

## Thời gian xử lý

Một request được xử lý **đồng bộ** (trả kết quả khi xong), thường mất vài phút:

| Đầu vào | Thời gian tham khảo (văn bản 8 trang) |
|---|---|
| `.docx` | ~1.5–3 phút (chỉ LLM) |
| PDF | ~3–8 phút (OCR + LLM) |

LLM xử lý **1 request mỗi lần**, request đến sau phải xếp hàng → client / reverse proxy cần đặt
timeout đủ lớn (khuyến nghị ≥ 15 phút).

## Lưu ý về kết quả

Kết quả do LLM sinh ra, nên được **người rà soát lại** trước khi sử dụng chính thức:
có thể sót nhiệm vụ, gán nhầm vai trò phối hợp, hoặc ghi tên đơn vị chưa chuẩn hóa.
"""

TAGS_METADATA = [
    {"name": "Nhiệm vụ", "description": "Trích xuất nhiệm vụ từ văn bản hành chính."},
    {"name": "Hệ thống", "description": "Kiểm tra trạng thái dịch vụ."},
]

app = FastAPI(
    title="Trích xuất nhiệm vụ",
    version="1.0.0",
    description=API_DESCRIPTION,
    openapi_tags=TAGS_METADATA,
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
    """Một nhiệm vụ được giao trong văn bản."""

    task_name: str = Field(
        description="Tên ngắn gọn của nhiệm vụ (động từ + đối tượng)",
        examples=["Thẩm định hồ sơ điều chỉnh cục bộ quy hoạch chung xây dựng"],
    )
    requirement: str = Field(
        description="Yêu cầu cần hoàn thành: làm gì, về vấn đề gì, sản phẩm đầu ra; bám sát câu chữ văn bản",
        examples=["Tổ chức thẩm định hồ sơ điều chỉnh cục bộ quy hoạch chung xây dựng Thành phố Hồ Chí Minh, "
                  "trình Thủ tướng Chính phủ xem xét, quyết định."],
    )
    executing_unit: str | None = Field(
        description="Đơn vị thực hiện / chủ trì. Mỗi nhiệm vụ đúng 1 đơn vị (văn bản giao cho nhiều đơn vị "
                    "cùng thực hiện thì tách thành nhiều nhiệm vụ); null nếu không xác định được",
        examples=["Bộ Xây dựng"],
    )
    coordinating_units: list[str] = Field(
        description="Các đơn vị phối hợp; [] nếu không có. Cụm chung chung (\"các cơ quan liên quan\"...) "
                    "được suy luận thành đơn vị cụ thể qua knowledge base, không suy luận được thì bị bỏ",
        examples=[["Ủy ban nhân dân Thành phố Hồ Chí Minh"]],
    )
    deadline: str | None = Field(
        description="Thời hạn, giữ nguyên cách ghi trong văn bản; null nếu văn bản không nêu",
        examples=["Trong thời gian 10 ngày làm việc sau khi nhận được Hồ sơ"],
    )
    reference: str | None = Field(
        description="Vị trí trong văn bản (mục, khoản)",
        examples=["Mục III, khoản 10"],
    )


class Health(BaseModel):
    status: str = Field(description="`ok` khi dịch vụ sẵn sàng", examples=["ok"])
    model: str = Field(description="Tên model LLM đang dùng", examples=["local-llm"])


class ErrorResponse(BaseModel):
    detail: str = Field(description="Mô tả lỗi")


def _error(description: str, example: str) -> dict:
    return {
        "model": ErrorResponse,
        "description": description,
        "content": {"application/json": {"example": {"detail": example}}},
    }


class TextRequest(BaseModel):
    text: str = Field(description="Nội dung văn bản (đã OCR)")
    model: str = MODEL


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:  # lỗi OCR/LLM upstream
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get(
    "/health",
    response_model=Health,
    tags=["Hệ thống"],
    summary="Kiểm tra trạng thái",
    description="Trả `status: ok` khi API đang chạy. Không kiểm tra OCR / LLM phía sau.",
)
def health():
    return {"status": "ok", "model": MODEL}


@app.post(
    "/v1/tasks",
    response_model=list[Task],
    tags=["Nhiệm vụ"],
    summary="Trích xuất nhiệm vụ từ file văn bản",
    responses={
        400: _error("File rỗng, file .docx hỏng, hoặc không trích xuất được nội dung", "File rỗng."),
        415: _error("Định dạng không hỗ trợ (chỉ nhận .pdf, .docx)", "Chỉ hỗ trợ file PDF hoặc Word (.docx)."),
        502: _error("Lỗi từ dịch vụ OCR hoặc LLM phía sau", "Không đọc được JSON từ LLM: ..."),
    },
)
async def tasks_from_document(file: UploadFile = File(..., description="File văn bản PDF (kể cả bản scan) hoặc Word (.docx)")):
    """
    Upload một văn bản hành chính, trả về **mảng nhiệm vụ** (`list[Task]`).

    **Đầu vào** — `multipart/form-data`, trường `file`:

    | Định dạng | Cách lấy nội dung |
    |---|---|
    | `.pdf` | Gửi sang dịch vụ OCR (hỗ trợ PDF scan) |
    | `.docx` | Đọc trực tiếp đoạn văn + bảng, không qua OCR (nhanh hơn) |
    | `.doc` | Không hỗ trợ — lưu lại thành `.docx` |

    **Quy tắc bóc tách**

    - A thực hiện, B phối hợp → **1 nhiệm vụ**: `executing_unit = A`, `coordinating_units = [B]`.
    - A và B cùng thực hiện → **2 nhiệm vụ**, mỗi đơn vị một nhiệm vụ.
    - Cụm chung chung ("các cơ quan liên quan", "các Bộ, ngành"...) được suy luận thành đơn vị cụ thể
      qua knowledge base; không suy luận được thì bị bỏ khỏi danh sách.

    **Ví dụ**

    ```bash
    curl -m 1200 -F "file=@Thông-báo-192-TB-VPCP.pdf" http://localhost:8080/v1/tasks
    ```

    Request xử lý đồng bộ, mất vài phút (PDF 8 trang: ~3–8 phút) — đặt timeout phía client ≥ 15 phút.
    """
    filename = (file.filename or "").lower()
    if filename.endswith(".doc"):
        raise HTTPException(status_code=415, detail="Định dạng .doc cũ chưa hỗ trợ, hãy lưu lại thành .docx.")
    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ file PDF hoặc Word (.docx).")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng.")

    if filename.endswith(".docx"):
        try:
            text = await run_in_threadpool(extract_docx_text, content)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    else:
        # Giữ tên file gốc khi gửi sang OCR API.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, os.path.basename(file.filename))
            with open(path, "wb") as f:
                f.write(content)
            ocr_result = await run_in_threadpool(_run, ocr_document, path)
        text = extract_text(ocr_result)

    if not text:
        raise HTTPException(status_code=400, detail="Không trích xuất được nội dung văn bản.")
    return await run_in_threadpool(_run, extract_tasks, text)


