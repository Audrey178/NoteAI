# NoteAI — Trích xuất giao việc từ văn bản PDF bằng OCR + LLM

Pipeline đọc văn bản hành chính dạng PDF và xác định **đích danh đơn vị nào phải làm nhiệm vụ gì**:

```
PDF ──► OCR ──► văn bản thô ──► LLM + knowledge base ──► giao việc .md / JSON
```

OCR và LLM đều có thể self-host bằng `docker compose` trên máy có GPU 8GB VRAM
(OCR: PaddleOCR-VL-1.6 chạy CPU; LLM: Qwen3.5-9B trên llama.cpp chạy GPU) — xem [Docker compose](#docker-compose-self-host).

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

Sau đó chỉnh `.env`. Có 2 cách chạy, dùng các biến khác nhau:

| Cách chạy                                                 | OCR / LLM lấy ở đâu                                                                            | Biến cần quan tâm                                   |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **`docker compose`** (self-host, khuyến nghị)             | Container `ocr` / `llm` trong cùng compose — compose tự đặt `OCR_ENDPOINT` / `OPENAI_BASE_URL` | Bảng 2 (thư mục model, model LLM, context...)       |
| **Chạy trực tiếp** (`pdf_pipeline.py`, `uvicorn app:app`) | Dịch vụ bất kỳ có URL truy cập được                                                            | Bảng 1, đặc biệt `OCR_ENDPOINT` / `OPENAI_BASE_URL` |

**Bảng 1 — biến của app** (đọc trong `utils/`, `app.py`):

| Biến                                                       | Ý nghĩa                                                      | Mặc định trong code                                      |
| ---------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- |
| `OCR_ENDPOINT`                                             | URL OCR API (`.../v1/ocr/documents`)                         | — (bắt buộc khi OCR file PDF)                            |
| `OCR_TIMEOUT`                                              | Thời gian chờ OCR tối đa (giây)                              | `900`                                                    |
| `OPENAI_BASE_URL`                                          | Base URL của server LLM tương thích OpenAI                   | `http://localhost:8000/v1`                               |
| `MODEL_NAME`                                               | Tên model trên server (compose: cũng là alias của llama.cpp) | `google/gemma-4-26B-A4B-it`, `.env.example`: `local-llm` |
| `OPENAI_API_KEY`                                           | API key (để trống nếu server không yêu cầu)                  | `EMPTY`                                                  |
| `LLM_TEMPERATURE`                                          | Temperature khi gọi LLM                                      | `0`                                                      |
| `LLM_MAX_RETRIES` / `LLM_BACKOFF_BASE` / `LLM_BACKOFF_MAX` | Retry khi lỗi kết nối / 429 / 5xx (số lần, giây)             | `3` / `2` / `30`                                         |
| `KB_PATH`                                                  | Đường dẫn knowledge base                                     | `knowledge_base.json`                                    |
| `CORS_ORIGINS`                                             | Origin được phép gọi API, cách nhau bởi dấu phẩy             | `*`                                                      |

Server LLM có thể là bất kỳ endpoint nào hỗ trợ `/chat/completions` theo chuẩn OpenAI
(llama.cpp, vLLM, SGLang, Ollama, ...).

**Bảng 2 — biến của `docker compose`** (self-host):

| Biến                          | Ý nghĩa                                            | Mặc định                                             |
| ----------------------------- | -------------------------------------------------- | ---------------------------------------------------- |
| `LLM_MODELS_DIR`              | Thư mục lưu model GGUF (LLM + PaddleOCR-VL)        | `./models`                                           |
| `OCR_MODELS_DIR`              | Thư mục lưu model layout của PaddleOCR             | `./models/paddlex`                                   |
| `LLM_HF_REPO` / `LLM_HF_FILE` | Model LLM trên Hugging Face                        | `unsloth/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf` |
| `LLM_CTX_SIZE`                | Context của LLM (token)                            | `32768`                                              |
| `LLM_REASONING_BUDGET`        | Giới hạn token suy nghĩ (-1 không giới hạn, 0 tắt) | `8192`                                               |
| `OCR_VL_THREADS`              | Số luồng CPU cho VLM nhận dạng chữ                 | `8`                                                  |
| `API_PORT`                    | Cổng host của API                                  | `8080`                                               |

Ví dụ `.env` cho server / workspace có ổ riêng cho đồ nặng (vd `~/scratch` trên workspace Coder) —
để model ở đó cho khỏi đầy quota home:

```bash
OCR_TIMEOUT=900
MODEL_NAME=local-llm
LLM_TEMPERATURE=0

LLM_MODELS_DIR=/home/jovyan/scratch/<user>/models
OCR_MODELS_DIR=/home/jovyan/scratch/<user>/paddlex

LLM_CTX_SIZE=32768
LLM_REASONING_BUDGET=8192
OCR_VL_THREADS=8
API_PORT=8080
```

`.env` nằm trong `.gitignore` — không commit file này (có thể chứa API key).

## Sử dụng

CLI `pdf_pipeline.py` gọi thẳng `OCR_ENDPOINT` và `OPENAI_BASE_URL` trong `.env`, nên cần 2 dịch vụ này
truy cập được từ máy chạy lệnh. Stack `docker compose` **không mở cổng** `ocr` / `llm` ra host — khi dùng
compose, gọi qua API (`POST /v1/tasks`), hoặc dùng `--ocr-json` để bỏ qua OCR.

Trích xuất giao việc từ một file PDF (gọi OCR rồi LLM):

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

| File                | Nội dung                                        |
| ------------------- | ----------------------------------------------- |
| `<tên>_ocr.json`    | JSON trả về từ OCR API (chỉ tạo khi gọi OCR)    |
| `<tên>_text.txt`    | Văn bản đã trích xuất, dùng làm đầu vào cho LLM |
| `<tên>_giaoviec.md` | Bảng giao việc Markdown                         |

Kết quả cũng được in ra terminal khi chạy xong.

## API service

API (FastAPI) nhận file văn bản hành chính và trả về **danh sách nhiệm vụ** dạng JSON:
đơn vị nào thực hiện, đơn vị nào phối hợp, yêu cầu gì, thời hạn nào, căn cứ ở đâu trong văn bản.

```
PDF  ──► OCR ──┐
               ├──► văn bản ──► LLM + knowledge base ──► list[Task]
.docx ─────────┘   (đọc trực tiếp, không qua OCR)
```

### Chạy

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Hoặc Docker, gồm cả LLM và OCR self-host (xem mục [Docker compose](#docker-compose-self-host)).

Swagger UI: `http://<host>:8080/docs`

### Endpoint

| Method | Path        | Input                               | Output                                         |
| ------ | ----------- | ----------------------------------- | ---------------------------------------------- |
| `GET`  | `/health`   | —                                   | `{"status": "ok", "model": "<tên model LLM>"}` |
| `POST` | `/v1/tasks` | multipart `file` (PDF hoặc `.docx`) | `list[Task]`                                   |

`/health` chỉ cho biết API đang chạy, không kiểm tra OCR / LLM phía sau.

### `POST /v1/tasks`

Đầu vào: `multipart/form-data`, trường `file`.

| Định dạng | Cách lấy nội dung                                        |
| --------- | -------------------------------------------------------- |
| `.pdf`    | Gửi sang dịch vụ OCR (`OCR_ENDPOINT`), hỗ trợ PDF scan   |
| `.docx`   | Đọc trực tiếp đoạn văn + bảng, không qua OCR (nhanh hơn) |
| `.doc`    | Không hỗ trợ — lưu lại thành `.docx`                     |

```bash
curl -m 1200 -F "file=@Thông-báo-192-TB-VPCP.pdf" http://localhost:8080/v1/tasks
```

Đầu ra: mảng JSON, mỗi phần tử là một `Task`:

| Trường               | Kiểu           | Ý nghĩa                                                                                |
| -------------------- | -------------- | -------------------------------------------------------------------------------------- |
| `task_name`          | string         | Tên ngắn gọn của nhiệm vụ (động từ + đối tượng)                                        |
| `requirement`        | string         | Yêu cầu cần hoàn thành: làm gì, về vấn đề gì, sản phẩm đầu ra; bám sát câu chữ văn bản |
| `executing_unit`     | string \| null | Đơn vị thực hiện / chủ trì — đúng 1 đơn vị; `null` nếu không xác định được             |
| `coordinating_units` | string[]       | Các đơn vị phối hợp; `[]` nếu không có                                                 |
| `deadline`           | string \| null | Thời hạn, giữ nguyên cách ghi trong văn bản; `null` nếu văn bản không nêu              |
| `reference`          | string \| null | Vị trí trong văn bản (mục, khoản)                                                      |

```json
[
  {
    "task_name": "Thẩm định hồ sơ điều chỉnh cục bộ quy hoạch chung xây dựng",
    "requirement": "Tổ chức thẩm định hồ sơ điều chỉnh cục bộ quy hoạch chung xây dựng Thành phố Hồ Chí Minh, trình Thủ tướng Chính phủ xem xét, quyết định.",
    "executing_unit": "Bộ Xây dựng",
    "coordinating_units": [],
    "deadline": "Trong thời gian 10 ngày làm việc sau khi nhận được Hồ sơ",
    "reference": "Mục III, khoản 10"
  }
]
```

Quy tắc bóc tách:

- Đơn vị A thực hiện, đơn vị B phối hợp → **01 nhiệm vụ** (`executing_unit = A`, `coordinating_units = [B]`).
- Đơn vị A và đơn vị B cùng thực hiện → **02 nhiệm vụ**, mỗi đơn vị thực hiện một nhiệm vụ
  (việc tách được làm trong code, `utils/tasks.py::normalize_tasks`).
- Cụm chung chung ("các cơ quan liên quan", "các Bộ, ngành"...) được suy luận thành đơn vị cụ thể qua
  `knowledge_base.json`; cụm không suy luận được sẽ bị loại khỏi danh sách đơn vị.

Mã lỗi (body dạng `{"detail": "..."}`):

| Mã    | Khi nào                                                      | Ví dụ `detail`                           |
| ----- | ------------------------------------------------------------ | ---------------------------------------- |
| `400` | File rỗng, `.docx` hỏng, hoặc không trích xuất được nội dung | `File rỗng.`                             |
| `415` | Không phải `.pdf` / `.docx`                                  | `Chỉ hỗ trợ file PDF hoặc Word (.docx).` |
| `422` | Thiếu trường `file`                                          | —                                        |
| `502` | Lỗi từ dịch vụ OCR hoặc LLM phía sau                         | `Không đọc được JSON từ LLM: ...`        |

Thời gian xử lý: request chạy **đồng bộ**, trả kết quả khi xong.

| Đầu vào (văn bản 8 trang) | Thời gian tham khảo                      |
| ------------------------- | ---------------------------------------- |
| `.docx`                   | ~1.5–3 phút (chỉ LLM)                    |
| PDF                       | ~3–8 phút (OCR self-host chạy CPU + LLM) |

LLM xử lý 1 request mỗi lần, request đến sau phải xếp hàng → đặt timeout ở client / reverse proxy đủ lớn
(khuyến nghị ≥ 15 phút).

Kết quả do LLM sinh ra nên cần **người rà soát** trước khi dùng chính thức: có thể sót nhiệm vụ,
gán nhầm vai trò phối hợp, hoặc tên đơn vị chưa chuẩn hóa.

### OCR service (`ocr_service/`)

Dịch vụ OCR self-host, dùng nội bộ trong `docker compose` (không publish cổng ra ngoài):

| Method | Path                | Input                                           | Output                                            |
| ------ | ------------------- | ----------------------------------------------- | ------------------------------------------------- |
| `GET`  | `/health`           | —                                               | `{"status": "ok" \| "loading", "vl_server": ...}` |
| `POST` | `/v1/ocr/documents` | multipart `file` (PDF hoặc ảnh png/jpg/bmp/tif) | JSON, trường `content` là văn bản Markdown        |

```json
{
  "schema_version": "1.0.0",
  "document": { "filename": "a.pdf", "page_count": 8 },
  "content": "THÔNG BÁO\n\nKết luận của Thủ tướng Chính phủ ...",
  "processing": { "ocr_engine": "paddleocr-vl-1.6", "duration_ms": 315535 }
}
```

Giao diện tương thích với OCR API cũ mà `utils/ocr_client.py` gọi: các form field `rasterize`,
`enable_correction`, `process_table`, `use_celery`, `use_cache` được chấp nhận nhưng bỏ qua; ảnh / con dấu
trong Markdown được lọc bỏ. Mã lỗi: `415` định dạng không hỗ trợ, `400` file rỗng, `500` lỗi OCR.

## Docker compose (self-host)

```
                 ┌──────────────────────────── docker compose ────────────────────────────┐
PDF / .docx ──►  │ summarizer :8080 ──► ocr (layout, CPU) ──► ocr-vl (PaddleOCR-VL, CPU)   │
                 │        │                                                               │
                 │        └──────────► llm (Qwen3.5-9B, llama.cpp, GPU ~6.3GB)            │
                 └────────────────────────────────────────────────────────────────────────┘
```

| Service      | Container            | Vai trò                                             | Chạy trên |
| ------------ | -------------------- | --------------------------------------------------- | --------- |
| `summarizer` | `meeting-summarizer` | API `/v1/tasks` (cổng `API_PORT`, mặc định 8080)    | CPU       |
| `llm`        | `noteai-llm`         | LLM trích xuất nhiệm vụ                             | GPU       |
| `ocr`        | `noteai-ocr`         | Phân tích layout PaddleOCR, API `/v1/ocr/documents` | CPU       |
| `ocr-vl`     | `noteai-ocr-vl`      | VLM nhận dạng chữ PaddleOCR-VL-1.6                  | CPU       |

Yêu cầu: GPU NVIDIA ≥ 8GB VRAM + `nvidia-container-toolkit`, RAM trống ≥ 8GB (OCR ~3.5GB).

```bash
cp .env.example .env
# Đặt thư mục lưu model (nên để ở ổ dành cho đồ nặng, vd ~/scratch trên workspace Coder):
#   LLM_MODELS_DIR=/path/to/scratch/models
#   OCR_MODELS_DIR=/path/to/scratch/paddlex
docker compose up -d --build
```

Lần đầu tự tải model (~7.5GB: Qwen3.5-9B 5.7GB + PaddleOCR-VL 1.8GB + model layout); các lần sau dùng lại
từ `LLM_MODELS_DIR` / `OCR_MODELS_DIR`. `summarizer` chỉ khởi động sau khi `llm` và `ocr` healthy.

Thời gian trên `Thông-báo-192-TB-VPCP.pdf` (8 trang): OCR ~5 phút (CPU 8 luồng) + LLM ~2.5–3 phút.

### LLM self-host

`docker-compose.yml` chạy kèm container `llm`: **llama.cpp server**, toàn bộ model nằm trên GPU
nên không đòi hỏi nhiều RAM.

- Yêu cầu: GPU NVIDIA ≥ 8GB VRAM + `nvidia-container-toolkit`.
- Lần chạy đầu tự tải weight về `LLM_MODELS_DIR` (mặc định `./models`), các lần sau dùng lại;
  `summarizer` chờ `llm` healthy mới khởi động.
- Context mặc định 32k token, KV cache lượng tử hóa `q8_0`, xử lý 1 request mỗi lần (các request khác xếp hàng).

Chọn model qua `LLM_HF_REPO` / `LLM_HF_FILE` trong `.env`:

| Model                        | `LLM_HF_REPO` / `LLM_HF_FILE`                                          | Weight | Ghi chú                                                                                       |
| ---------------------------- | ---------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------- |
| Qwen3.5-9B Q4_K_M (mặc định) | `unsloth/Qwen3.5-9B-GGUF` / `Qwen3.5-9B-Q4_K_M.gguf`                   | 5.7GB  | Vừa 8GB thoải mái với context 32k                                                             |
| Gemma-4-12B QAT Q4_0         | `google/gemma-4-12B-it-qat-q4_0-gguf` / `gemma-4-12b-it-qat-q4_0.gguf` | 7.0GB  | Cùng họ với model gốc nên prompt hợp hơn, nhưng rất sát 8GB — có thể phải giảm `LLM_CTX_SIZE` |
| Gemma-4-E4B QAT Q4_0         | `google/gemma-4-E4B-it-qat-q4_0-gguf` / `gemma-4-E4B_q4_0-it.gguf`     | 5.2GB  | Nhẹ, nhưng khả năng suy luận đơn vị yếu hơn                                                   |

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

| VLM chạy trên                                          | 8 trang | VRAM                             |
| ------------------------------------------------------ | ------- | -------------------------------- |
| CPU 8 luồng (mặc định)                                 | ~291s   | 0                                |
| GPU (image `server-cuda` + reservation GPU, `-ngl 99`) | ~69s    | ~2.8GB — chỉ dùng khi GPU ≥ 10GB |

### Vận hành & kiểm tra

Nếu Docker daemon không ở socket mặc định (vd server dùng chung `/run/docker-sock/docker.sock`):
`export DOCKER_HOST=unix:///run/docker-sock/docker.sock`.

```bash
# Trạng thái: cả 4 container Up, 3 container noteai-* (healthy)
docker compose ps
docker inspect -f '{{.Name}} restarts={{.RestartCount}} oom={{.State.OOMKilled}}' \
  noteai-llm noteai-ocr-vl noteai-ocr meeting-summarizer
curl -s localhost:8080/health                 # {"status":"ok","model":"local-llm"}

# Log
docker compose logs -f --tail 50
docker logs -f meeting-summarizer             # request vào API, lỗi 4xx/5xx
docker logs noteai-llm 2>&1 | grep "eval time"   # tốc độ mỗi lần gọi LLM
docker logs noteai-ocr 2>&1 | grep POST       # các lần gọi OCR

# Tài nguyên
nvidia-smi                                    # noteai-llm ~6.3GB VRAM
docker stats --no-stream                      # CPU / RAM từng container

# Test thật
time curl -s -m 1700 -F "file=@Thông-báo-192-TB-VPCP.pdf" localhost:8080/v1/tasks | python3 -m json.tool --no-ensure-ascii | head -40

# Cập nhật / khởi động lại / dừng
git pull && docker compose up -d --build
docker compose restart ocr-vl
docker compose down                           # model trong LLM_MODELS_DIR / OCR_MODELS_DIR vẫn giữ nguyên
```

Mở Swagger UI của server từ máy cá nhân: `ssh -L 8080:localhost:8080 <server>` rồi vào `http://localhost:8080/docs`.

## Ghi chú

- OCR một PDF 8 trang mất ~5 phút với OCR self-host chạy CPU; với văn bản dài hãy tăng `OCR_TIMEOUT` (mặc định 900s).
- Văn bản trích từ OCR có thể còn lỗi chính tả nhỏ — prompt đã lưu ý model về điều này.
- Muốn thay đổi cách trích xuất, sửa `EXTRACT_PROMPT` / `RESOLVE_PROMPT` trong `utils/assignments.py`.
- Chất lượng nhiệm vụ gián tiếp phụ thuộc vào độ phủ của `knowledge_base.json`: cơ quan không có trong KB sẽ không được gợi ý.
