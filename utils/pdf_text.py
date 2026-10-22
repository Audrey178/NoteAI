"""
pdf_text.py
-----------
Đọc lớp text có sẵn của PDF (PDF xuất từ Word/phần mềm) — nhanh hơn OCR hàng trăm lần,
rất đáng kể khi chạy CPU. PDF scan (không có lớp text) hoặc text lỗi font thì trả None để gọi OCR.
"""

import os
import unicodedata

import pymupdf

MIN_PAGE_CHARS = 50  # trang ít chữ hơn -> coi là trang scan
# Tỉ lệ tối thiểu chữ cái tiếng Việt có dấu ngoài Latin-1 (ư, ơ, đ, ệ, ...).
# Text lỗi font cũ (TCVN3/VNI) chỉ ra ký tự Latin-1 nên tỉ lệ ~0.
MIN_VIETNAMESE_RATIO = 0.02


def _looks_vietnamese(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    extended = sum(1 for c in letters if "Ā" <= c <= "ỿ")
    return extended / len(letters) >= MIN_VIETNAMESE_RATIO


def read_pdf_text_layer(file_path: str) -> dict | None:
    """Trả về kết quả cùng dạng JSON OCR (`document`, `content`), hoặc None nếu phải OCR."""
    try:
        with pymupdf.open(file_path) as doc:
            pages = [unicodedata.normalize("NFC", page.get_text()).strip() for page in doc]
    except Exception:  # PDF hỏng/mã hóa -> để OCR xử lý
        return None

    if not pages or any(len(p) < MIN_PAGE_CHARS for p in pages):
        return None
    content = "\n\n".join(pages)
    if not _looks_vietnamese(content):
        return None

    return {
        "schema_version": "1.0.0",
        "document": {"filename": os.path.basename(file_path), "page_count": len(pages)},
        "content": content,
        "processing": {"ocr_engine": "pdf-text-layer"},
    }
