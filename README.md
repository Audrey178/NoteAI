# NoteAI — Tóm tắt văn bản PDF bằng OCR + LLM

Pipeline chuyển văn bản hành chính dạng PDF thành bản tóm tắt Markdown có cấu trúc:

```
PDF ──► OCR API ──► văn bản thô ──► LLM (OpenAI-compatible) ──► tóm tắt .md
```

Bản tóm tắt gồm các phần: **Thông tin văn bản**, **Trích yếu**, **Nội dung chính**,
**Kết luận** (kèm căn cứ), **Mục tiêu / Định hướng** và bảng **Giao việc**
(đơn vị/cá nhân – nhiệm vụ – thời hạn). Prompt yêu cầu model chỉ dựa vào nội dung
văn bản, không suy diễn hay dùng kiến thức bên ngoài.

## Cấu trúc thư mục

```
.
├── pdf_pipeline.py      # Entry point: OCR -> tóm tắt, lưu kết quả vào results/
├── utils/
│   ├── ocr_client.py    # Gọi OCR API, đọc JSON OCR có sẵn, trích text
│   └── summarize.py     # Prompt tóm tắt + gọi endpoint /chat/completions
├── requirements.txt
├── .env.example         # Mẫu cấu hình
└── results/             # Kết quả đầu ra (bị .gitignore)
```

## Cài đặt

Yêu cầu Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Sau đó điền các biến trong `.env`:

| Biến              | Ý nghĩa                                              | Mặc định                      |
|-------------------|------------------------------------------------------|-------------------------------|
| `OCR_ENDPOINT`    | URL OCR API (`.../v1/ocr/documents`)                 | — (bắt buộc khi OCR file PDF) |
| `OCR_TIMEOUT`     | Thời gian chờ OCR tối đa (giây)                      | `900`                         |
| `OPENAI_BASE_URL` | Base URL của server LLM tương thích OpenAI           | `http://localhost:8000/v1`    |
| `MODEL_NAME`      | Tên model trên server                                | `google/gemma-4-26B-A4B-it`   |
| `OPENAI_API_KEY`  | API key (để trống nếu server không yêu cầu)          | `EMPTY`                       |

Server LLM có thể là bất kỳ endpoint nào hỗ trợ `/chat/completions` theo chuẩn OpenAI
(vLLM, SGLang, Ollama, ...).

## Sử dụng

Tóm tắt một file PDF (gọi OCR API rồi LLM):

```bash
python pdf_pipeline.py "Thông-báo-192-TB-VPCP.pdf"
```

Dùng lại kết quả OCR đã lưu, bỏ qua bước gọi OCR API (tiện khi thử prompt/model):

```bash
python pdf_pipeline.py --ocr-json "results/Thông-báo-192-TB-VPCP_ocr.json"
```

Chọn model khác với giá trị trong `.env`:

```bash
python pdf_pipeline.py file.pdf --model <tên-model>
```

## Kết quả

Với file đầu vào `<tên>.pdf`, pipeline ghi vào thư mục `results/`:

| File                  | Nội dung                                                  |
|-----------------------|-----------------------------------------------------------|
| `<tên>_ocr.json`      | JSON trả về từ OCR API (chỉ tạo khi gọi OCR)              |
| `<tên>_text.txt`      | Văn bản đã trích xuất, dùng làm đầu vào cho LLM           |
| `<tên>_summary.md`    | Bản tóm tắt Markdown                                      |

Bản tóm tắt cũng được in ra terminal khi chạy xong.

## Dùng như thư viện

```python
from pdf_pipeline import process_pdf

result = process_pdf("file.pdf")
print(result["summary"])        # nội dung tóm tắt
print(result["summary_path"])   # đường dẫn file .md
```

## Ghi chú

- OCR một PDF khoảng 7 trang mất ~85 giây; với văn bản dài hãy tăng `OCR_TIMEOUT`.
- Văn bản trích từ OCR có thể còn lỗi chính tả nhỏ — prompt đã lưu ý model về điều này.
- Muốn thay đổi cấu trúc bản tóm tắt, sửa `DOCUMENT_SUMMARY_PROMPT` trong `utils/summarize.py`.
