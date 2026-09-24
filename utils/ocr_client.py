"""
ocr_client.py
-------------
Gọi OCR API để chuyển file PDF/ảnh thành văn bản.
ENDPOINT cấu hình trong file .env (OCR_ENDPOINT).
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OCR_ENDPOINT = os.getenv("OCR_ENDPOINT")
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "900"))  # PDF 7 trang mất ~85s


def ocr_document(
    file_path: str,
    rasterize: bool = True,
    enable_correction: bool = True,
    process_table: bool = True,
    use_celery: bool = True,
    use_cache: bool = True,
    timeout: int = OCR_TIMEOUT,
) -> dict:
    """Upload file lên OCR API, trả về JSON Canonical OCR (schema 1.0.0)."""
    if not OCR_ENDPOINT:
        raise RuntimeError("Chưa cấu hình OCR_ENDPOINT trong file .env.")

    data = {
        "rasterize": str(rasterize).lower(),
        "enable_correction": str(enable_correction).lower(),
        "process_table": str(process_table).lower(),
        "use_celery": str(use_celery).lower(),
        "use_cache": str(use_cache).lower(),
    }

    try:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            response = requests.post(OCR_ENDPOINT, files=files, data=data, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Không kết nối được tới OCR API: {OCR_ENDPOINT}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("OCR API phản hồi quá lâu (timeout). Hãy tăng OCR_TIMEOUT.") from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"OCR API lỗi {response.status_code}: {response.text[:500]}") from e

    return response.json()


def load_ocr_json(json_path: str) -> dict:
    """Đọc lại kết quả OCR đã lưu (vd response_*.json) — tiện test không cần gọi API."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def extract_text(ocr_result: dict) -> str:
    """
    Lấy văn bản từ kết quả OCR.

    `content` là toàn bộ text theo thứ tự đọc, đã loại header/footer lặp lại
    (vd "Tài liệu này thuộc sử dụng của Viettel...") nên dùng trực tiếp để tóm tắt.
    """
    return (ocr_result.get("content") or "").strip()
