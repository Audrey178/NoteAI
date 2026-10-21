"""
ocr_service/app.py
------------------
OCR self-host bằng PaddleOCR-VL-1.6, thay cho OCR API bên ngoài.

- Phân tích layout (PP-DocLayout) chạy CPU trong container này.
- Nhận dạng từng vùng bằng VLM PaddleOCR-VL-1.6 (0.9B) chạy trên llama.cpp server (container ocr-vl, GPU).
  Model OCR dòng chữ truyền thống của PaddleOCR (PP-OCRv5/v6) không có chữ tiếng Việt có dấu nên không dùng.

Giữ nguyên giao diện mà utils/ocr_client.py đang gọi:
    POST /v1/ocr/documents  (multipart: file + các form field rasterize/enable_correction/... bị bỏ qua)
    -> JSON có trường `content` = toàn bộ văn bản (Markdown) theo thứ tự đọc.
"""

import os
import re
import tempfile
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

VL_SERVER_URL = os.getenv("OCR_VL_SERVER_URL", "http://ocr-vl:8000/v1")
DEVICE = os.getenv("OCR_DEVICE", "cpu")

ALLOWED_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# Ảnh/con dấu được xuất thành thẻ <img>/<div> trong Markdown — vô ích với LLM
HTML_IMAGE_RE = re.compile(r"<div[^>]*>\s*<img[^>]*>\s*</div>|<img[^>]*>")

API_DESCRIPTION = """
OCR self-host cho NoteAI, thay OCR API bên ngoài. Dùng **PaddleOCR-VL-1.6**:

1. Phân tích layout trang (PP-DocLayout, chạy CPU trong container này).
2. Nhận dạng chữ từng vùng bằng VLM PaddleOCR-VL-1.6 0.9B chạy trên llama.cpp (`OCR_VL_SERVER_URL`).

Hỗ trợ tiếng Việt có dấu, PDF scan nhiều trang, bảng (xuất Markdown).
Xử lý tuần tự từng request; PDF 8 trang ~5 phút khi VLM chạy CPU 8 luồng.
"""

app = FastAPI(title="NoteAI OCR (PaddleOCR-VL)", version="1.0.0", description=API_DESCRIPTION)
_pipeline = None
_lock = threading.Lock()  # pipeline Paddle không thread-safe: xử lý tuần tự


def _build_pipeline():
    from paddleocr import PaddleOCRVL

    return PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url=VL_SERVER_URL,
        device=DEVICE,
        # Paddle 3.3 + oneDNN lỗi (ConvertPirAttribute2RuntimeAttribute) -> tắt
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


def _ocr(path: str) -> tuple[str, int]:
    with _lock:
        results = list(_pipeline.predict(path))
        merged = _pipeline.concatenate_markdown_pages([r.markdown for r in results])
    text = merged if isinstance(merged, str) else merged["markdown_texts"]
    text = HTML_IMAGE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), len(results)


@app.on_event("startup")
def load_pipeline():
    global _pipeline
    _pipeline = _build_pipeline()


@app.get("/health", summary="Kiểm tra trạng thái",
         description="`status` là `ok` khi pipeline đã nạp xong model, `loading` khi đang khởi động.")
def health():
    return {"status": "ok" if _pipeline is not None else "loading", "vl_server": VL_SERVER_URL}


@app.post("/v1/ocr/documents", summary="OCR một file PDF / ảnh")
async def ocr_document(file: UploadFile = File(..., description="File PDF hoặc ảnh (png, jpg, bmp, tif)")):
    """
    Trả JSON, trong đó **`content`** là toàn bộ văn bản theo thứ tự đọc (Markdown, đã bỏ thẻ ảnh/con dấu).

    Giao diện tương thích với OCR API cũ mà `utils/ocr_client.py` gọi: các form field khác
    (`rasterize`, `enable_correction`, `process_table`, `use_celery`, `use_cache`) được chấp nhận nhưng bỏ qua.

    ```json
    {
      "schema_version": "1.0.0",
      "document": {"filename": "a.pdf", "page_count": 8},
      "content": "THÔNG BÁO\\n\\nKết luận của Thủ tướng Chính phủ ...",
      "processing": {"ocr_engine": "paddleocr-vl-1.6", "duration_ms": 315535}
    }
    ```

    **Lỗi:** 415 định dạng không hỗ trợ, 400 file rỗng, 500 lỗi OCR (xem `detail`).
    """
    filename = (file.filename or "").lower()
    suffix = os.path.splitext(filename)[1]
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Chỉ hỗ trợ: {', '.join(ALLOWED_SUFFIXES)}")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng.")

    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(content)
        tmp.flush()
        start = time.time()
        try:
            text, page_count = await run_in_threadpool(_ocr, tmp.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OCR lỗi: {e}") from e

    return {
        "schema_version": "1.0.0",
        "document": {"filename": file.filename, "page_count": page_count},
        "content": text,
        "processing": {
            "ocr_engine": "paddleocr-vl-1.6",
            "duration_ms": int((time.time() - start) * 1000),
        },
    }
