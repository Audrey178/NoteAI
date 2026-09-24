# NoteAI — Trích xuất giao việc từ văn bản PDF bằng OCR + LLM

Pipeline đọc văn bản hành chính dạng PDF và xác định **đích danh đơn vị nào phải làm nhiệm vụ gì**:

```
PDF ──► OCR API ──► văn bản thô ──► LLM + knowledge base ──► giao việc .md
```

Kết quả gồm:

- **Rà soát chủ thể chung chung** — các chỗ văn bản chỉ nêu "các cơ quan liên quan", "các cơ quan tham mưu"...
- **A. Nhiệm vụ trực tiếp** — đơn vị xác định được từ chính văn bản (nêu đích danh, hoặc suy ra chắc chắn từ context).
- **B. Nhiệm vụ gián tiếp** — với từng chủ thể chung chung, suy luận từ context văn bản + `knowledge_base.json`
  (phạm vi / từ khóa / đối tượng của từng cơ quan) để chỉ ra đích danh cơ quan, kèm lý do và mức tin cậy.
- **C. Chưa xác định được đơn vị** — chủ thể chung chung mà không có cơ quan nào trong KB phù hợp.

Gồm 2 lượt gọi LLM: lượt 1 trích nhiệm vụ trực tiếp + rà soát chủ thể chung chung (chỉ dựa vào văn bản);
lượt 2 dùng KB để xác định đơn vị cho các chủ thể chung chung.

## Cấu trúc thư mục

```
.
├── pdf_pipeline.py      # Entry point: OCR -> giao việc, lưu kết quả vào results/
├── knowledge_base.json  # Phạm vi phụ trách của các cơ quan (dùng để suy luận nhiệm vụ gián tiếp)
├── utils/
│   ├── ocr_client.py    # Gọi OCR API, đọc JSON OCR có sẵn, trích text
│   ├── assignments.py   # Prompt + 2 lượt trích xuất giao việc
│   ├── knowledge_base.py# Đọc KB, chuyển sang dạng đưa vào prompt
│   └── llm.py           # Gọi endpoint /chat/completions
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
| `LLM_TEMPERATURE` | Temperature khi gọi LLM                              | `0`                           |
| `KB_PATH`         | Đường dẫn knowledge base                             | `knowledge_base.json`         |

Server LLM có thể là bất kỳ endpoint nào hỗ trợ `/chat/completions` theo chuẩn OpenAI
(vLLM, SGLang, Ollama, ...).

## Sử dụng

Trích xuất giao việc từ một file PDF (gọi OCR API rồi LLM):

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
| `<tên>_giaoviec.md`   | Bảng giao việc Markdown                                   |

Kết quả cũng được in ra terminal khi chạy xong.

## Dùng như thư viện

```python
from pdf_pipeline import process_pdf

result = process_pdf("file.pdf")
print(result["assignments"])        # nội dung giao việc
print(result["assignments_path"])   # đường dẫn file .md
```

## Ghi chú

- OCR một PDF khoảng 7 trang mất ~85 giây; với văn bản dài hãy tăng `OCR_TIMEOUT`.
- Văn bản trích từ OCR có thể còn lỗi chính tả nhỏ — prompt đã lưu ý model về điều này.
- Muốn thay đổi cách trích xuất, sửa `EXTRACT_PROMPT` / `RESOLVE_PROMPT` trong `utils/assignments.py`.
- Chất lượng nhiệm vụ gián tiếp phụ thuộc vào độ phủ của `knowledge_base.json`: cơ quan không có trong KB sẽ không được gợi ý.
