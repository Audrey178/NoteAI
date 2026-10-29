"""
pdf_pipeline.py
---------------
Flow: PDF văn bản -> OCR API (ocr_client.py) -> trích xuất giao việc bằng LLM + knowledge base (assignments.py).

Chạy:
    python pdf_pipeline.py "Hệ thống văn phòng Điện tử VOFFICE.pdf"
    python pdf_pipeline.py --ocr-json response_1790213221450.json   # dùng kết quả OCR có sẵn
"""

import argparse
import json
import os

from utils.assignments import extract_assignments
from utils.llm import MODEL
from utils.ocr_client import extract_text, load_ocr_json, ocr_document

RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def process_pdf(pdf_path: str | None = None, ocr_json: str | None = None,
                model: str = MODEL) -> dict:
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    if ocr_json:
        print(f"[1/2] Đọc kết quả OCR từ {ocr_json}...")
        ocr_result = load_ocr_json(ocr_json)
    else:
        print(f"[1/2] Đang OCR {pdf_path}...")
        ocr_result = ocr_document(pdf_path)

    doc = ocr_result.get("document", {})
    name = os.path.splitext(doc.get("filename") or os.path.basename(pdf_path or ocr_json))[0]
    text = extract_text(ocr_result)

    if not ocr_json:
        with open(os.path.join(RESULTS_FOLDER, f"{name}_ocr.json"), "w", encoding="utf-8") as f:
            json.dump(ocr_result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS_FOLDER, f"{name}_text.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[2/2] Đang trích xuất giao việc {doc.get('page_count', '?')} trang ({len(text)} ký tự) bằng {model}...")
    assignments = extract_assignments(text, model=model)

    assignments_path = os.path.join(RESULTS_FOLDER, f"{name}_giaoviec.md")
    with open(assignments_path, "w", encoding="utf-8") as f:
        f.write(assignments)
    print(f"Đã lưu giao việc: {assignments_path}")

    return {"text": text, "assignments": assignments, "assignments_path": assignments_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất giao việc từ văn bản PDF qua OCR API + LLM")
    parser.add_argument("pdf", nargs="?", help="Đường dẫn file PDF")
    parser.add_argument("--ocr-json", help="Dùng file JSON OCR có sẵn thay vì gọi API")
    parser.add_argument("--model", default=MODEL, help="Tên model LLM")
    args = parser.parse_args()

    if not args.pdf and not args.ocr_json:
        parser.error("Cần truyền file PDF hoặc --ocr-json.")

    result = process_pdf(args.pdf, args.ocr_json, args.model)
    print("\n" + result["assignments"])
