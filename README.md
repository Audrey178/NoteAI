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
├── pdf_pipeline.py      # Entry point CLI: OCR -> giao việc, lưu kết quả vào results/
├── app.py               # API service (FastAPI): PDF -> danh sách nhiệm vụ JSON
├── Dockerfile           # Đóng gói API service
├── docker-compose.yml   # API + LLM self-host (llama.cpp) + OCR self-host (PaddleOCR-VL)
├── ocr_service/         # OCR self-host: PaddleOCR-VL-1.6, giữ giao diện /v1/ocr/documents
├── knowledge_base.json  # Phạm vi phụ trách của các cơ quan (dùng để suy luận nhiệm vụ gián tiếp)
├── utils/
│   ├── ocr_client.py    # Gọi OCR API, đọc JSON OCR có sẵn, trích text
│   ├── assignments.py   # Prompt + 2 lượt trích xuất giao việc
│   ├── tasks.py         # Prompt + chuẩn hóa/tách nhiệm vụ dạng JSON (dùng cho API)
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

## API service

Trả về **danh sách nhiệm vụ** dạng JSON. Mỗi nhiệm vụ gồm:

| Trường | Ý nghĩa |
|---|---|
| `task_name` | Tên nhiệm vụ |
| `requirement` | Yêu cầu cần hoàn thành |
| `executing_unit` | Đơn vị thực hiện (đúng 1 đơn vị) |
| `coordinating_units` | Danh sách đơn vị phối hợp |
| `deadline` | Thời gian cần hoàn thành (`null` nếu văn bản không nêu) |
| `reference` | Vị trí trong văn bản |

Quy tắc bóc tách:

- Đơn vị A thực hiện, đơn vị B phối hợp → **01 nhiệm vụ** (A thực hiện, B phối hợp).
- Đơn vị A và đơn vị B cùng thực hiện → **02 nhiệm vụ**, mỗi đơn vị thực hiện một nhiệm vụ
  (việc tách được làm trong code, `utils/tasks.py::normalize_tasks`).
- Cụm chung chung ("các cơ quan liên quan"...) được suy luận thành đơn vị cụ thể qua `knowledge_base.json`;
  cụm không suy luận được sẽ bị loại khỏi danh sách đơn vị.

### Chạy

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Hoặc Docker, gồm cả LLM và OCR self-host (xem các mục dưới):

```bash
docker compose up -d
```

### LLM self-host (GPU 8GB VRAM)

`docker-compose.yml` chạy kèm container `llm`: **llama.cpp server**, toàn bộ model nằm trên GPU
nên không đòi hỏi nhiều RAM.

- Yêu cầu: GPU NVIDIA ≥ 8GB VRAM + `nvidia-container-toolkit`.
- Lần chạy đầu tự tải weight về `LLM_MODELS_DIR` (mặc định `./models`), các lần sau dùng lại;
  `summarizer` chờ `llm` healthy mới khởi động.
- Context mặc định 32k token, KV cache lượng tử hóa `q8_0`, xử lý 1 request mỗi lần (các request khác xếp hàng).

Chọn model qua `LLM_HF_REPO` / `LLM_HF_FILE` trong `.env`:

| Model | `LLM_HF_REPO` / `LLM_HF_FILE` | Weight | Ghi chú |
|---|---|---|---|
| Qwen3.5-9B Q4_K_M (mặc định) | `unsloth/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf` | 5.7GB | Vừa 8GB thoải mái với context 32k |
| Gemma-4-12B QAT Q4_0 | `google/gemma-4-12B-it-qat-q4_0-gguf` / `gemma-4-12b-it-qat-q4_0.gguf` | 7.0GB | Cùng họ với model gốc nên prompt hợp hơn, nhưng rất sát 8GB — có thể phải giảm `LLM_CTX_SIZE` |
| Gemma-4-E4B QAT Q4_0 | `google/gemma-4-E4B-it-qat-q4_0-gguf` / `gemma-4-E4B_q4_0-it.gguf` | 5.2GB | Nhẹ, nhưng khả năng suy luận đơn vị yếu hơn |

Bị lỗi hết VRAM (OOM) thì giảm `LLM_CTX_SIZE` (vd `24576`) hoặc chọn model nhỏ hơn.

### OCR self-host (PaddleOCR-VL-1.6)

Thay OCR API bên ngoài bằng 2 container:

- `ocr-vl`: llama.cpp server chạy VLM **PaddleOCR-VL-1.6** (0.9B, `PaddlePaddle/PaddleOCR-VL-1.6-GGUF`) nhận dạng chữ.
- `ocr`: pipeline PaddleOCR-VL (`ocr_service/`) — phân tích layout trên CPU, gửi từng vùng sang `ocr-vl`;
  giữ nguyên giao diện `POST /v1/ocr/documents` -> `{"content": ...}` nên `utils/ocr_client.py` không đổi.

Model OCR dòng chữ truyền thống của PaddleOCR (PP-OCRv5 latin / PP-OCRv6) **không có chữ tiếng Việt có dấu**
(`Bộ` -> `B`, `Kết luận` -> `Kt lun`) nên không dùng.

Chất lượng trên `Thông-báo-192-TB-VPCP.pdf` (8 trang), so với OCR API cũ (Surya + sửa chính tả bằng Gemma 26B):
giống 99.2% theo ký tự, đủ 23 khoản, giữ đúng tên các Bộ/UBND; lỗi còn lại chủ yếu là sai dấu lẻ tẻ ("Phố" thay "Phó").

**OCR chạy CPU, GPU dành cho LLM**: LLM đã chiếm ~6.3GB, thêm VLM trên GPU (~2.8GB) sẽ vượt 8GB.
`ocr-vl` dùng image llama.cpp bản CPU, không xin GPU. Số luồng: `OCR_VL_THREADS` (mặc định 8).

Đo trên PDF 8 trang:

| VLM chạy trên | 8 trang | VRAM |
|---|---|---|
| CPU 8 luồng (mặc định) | ~291s | 0 |
| GPU (image `server-cuda` + reservation GPU, `-ngl 99`) | ~69s | ~2.8GB — chỉ dùng khi GPU ≥ 10GB |

Swagger UI: `http://<host>:8080/docs`

### Endpoint

| Method | Path | Input | Output |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "ok", "model": ...}` |
| `POST` | `/v1/tasks` | multipart `file` (PDF hoặc `.docx`) | `list[Task]` |

```bash
curl -F "file=@Thông-báo-192-TB-VPCP.pdf" http://localhost:8080/v1/tasks
```

```json
[
  {
    "task_name": "Thẩm định hồ sơ điều chỉnh cục bộ quy hoạch chung xây dựng Thành phố",
    "requirement": "Tổ chức thẩm định hồ sơ điều chỉnh cục bộ quy hoạch chung xây dựng Thành phố Hồ Chí Minh ...",
    "executing_unit": "Bộ Xây dựng",
    "coordinating_units": ["Ủy ban nhân dân Thành phố Hồ Chí Minh"],
    "deadline": "Trong thời gian 10 ngày làm việc sau khi nhận được Hồ sơ",
    "reference": "Mục III, khoản 10, 11"
  }
]
```

Mã lỗi: `415` không phải PDF, `400` file rỗng, `502` lỗi OCR API / LLM (kèm `detail`).
Một request PDF gồm OCR (~85s cho 7 trang) + LLM nên cần đặt timeout ở reverse proxy/client đủ lớn (vd ≥ 15 phút).

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
