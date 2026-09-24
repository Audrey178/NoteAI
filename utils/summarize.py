import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1") 
MODEL = os.getenv("MODEL_NAME", "google/gemma-4-26B-A4B-it")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY") 


DOCUMENT_SUMMARY_PROMPT = """\
Bạn là trợ lý văn phòng. Dựa hoàn toàn vào nội dung văn bản hành chính dưới đây
(được trích xuất bằng OCR nên có thể còn một số lỗi chính tả nhỏ),
hãy tóm tắt súc tích bằng Markdown, gồm đúng các phần sau:

## 1. Thông tin văn bản

Loại văn bản, số/ký hiệu, cơ quan ban hành, ngày ban hành, người ký (nếu có).

## 2. Trích yếu

Văn bản về việc gì (1-2 câu).

## 3. Nội dung chính

Gạch đầu dòng theo từng nội dung/mục quan trọng của văn bản.

## 4. Kết luận

Liệt kê các nội dung được chốt, quyết định, thống nhất, yêu cầu hoặc chỉ đạo trong văn bản.

Áp dụng các rule sau:

* Nếu văn bản **nói trực tiếp** một kết luận thì trích xuất trực tiếp, không suy diễn thêm.
* Nếu một kết luận **không được viết thành một câu hoàn chỉnh**, nhưng có thể xác định chắc chắn bằng cách liên kết các thông tin/context đã có trong chính văn bản, được phép tổng hợp thành một kết luận ngắn gọn.
* Không được tạo kết luận mới chỉ vì nó "hợp lý" hoặc thường xảy ra trong thực tế.
* Không sử dụng kiến thức bên ngoài văn bản.
* Không biến việc paraphrase một câu đã có thành một kết luận mới.
* Nếu không đủ căn cứ thì không đưa vào kết luận.
* Với mỗi kết luận cần có căn cứ trong văn bản để kiểm chứng; có thể ghi ngắn gọn `(Căn cứ: ...)`.

## 5. Mục tiêu / Định hướng

Chỉ liệt kê **mục tiêu hoặc định hướng cấp cao** mà văn bản đặt ra,
tức là trạng thái hoặc hướng phát triển mong muốn trong tương lai.

Chỉ đưa vào mục này khi nội dung mang tính:

- định hướng phát triển;
- mục tiêu tổng thể;
- trạng thái mong muốn ở cấp chiến lược;
- định hướng dài hạn.

**Không đưa vào mục này:**

- nhiệm vụ cụ thể của một đơn vị/cá nhân;
- hành động phải thực hiện;
- thời hạn phải hoàn thành;
- chỉ tiêu gắn trực tiếp với một nhiệm vụ cụ thể;
- kết luận/quyết định đã được chốt.

Ví dụ:

> "Phát triển Thành phố trở thành trung tâm tài chính khu vực và quốc tế."

→ Mục tiêu / Định hướng.

Trong khi:

> "Giao Bộ X hoàn thành Đề án trung tâm tài chính trong Quý IV."

→ Giao việc, không đưa vào Mục tiêu.

Nếu văn bản không có mục tiêu/định hướng cấp cao rõ ràng:
`Không nêu rõ.`

## 6. Giao việc

| Đơn vị/cá nhân | Nhiệm vụ | Thời hạn |
|---|---|---|

- Ghi các nhiệm vụ mà văn bản xác định **ai phải thực hiện hành động gì**.
- Chủ thể và hành động được nêu rõ trong văn bản; trích xuất trực tiếp, không ghi `(suy ra)`.
- Chủ thể hoặc hành động không nằm trong một câu giao việc hoàn chỉnh nhưng có thể xác định chắc chắn bằng cách liên kết các câu/context liên quan trong chính văn bản.
- Chỉ được suy ra khi căn cứ trong văn bản đủ rõ.
- Chỉ được suy ra **chủ thể hoặc liên kết giữa các thông tin đã có**, không tự tạo hành động mới.
- Không biến mục tiêu, chỉ tiêu hoặc kết quả mong muốn thành nhiệm vụ nếu văn bản không thể hiện hành động cần thực hiện.
- Không gom nội dung khái quát hoặc nhiều nhiệm vụ khác nhau thành một nhiệm vụ chung.
- Khi nhiều chủ thể thực hiện các hành động khác nhau, tách thành các nhiệm vụ tương ứng.
- Không gán toàn bộ chuỗi phối hợp cho một chủ thể; mỗi hành động phải được gắn với chủ thể có căn cứ trong văn bản.
- Không suy ra kiến thức bên ngoài văn bản.
- Mỗi nhiệm vụ độc lập là một dòng; các bước liên tiếp của cùng một nhiệm vụ có thể giữ trong cùng một dòng.
- Nếu không đủ căn cứ xác định **chủ thể + hành động**, không đưa vào bảng.

### Thời hạn

- Ghi đúng thời hạn được nêu trong văn bản.
- Không nêu → `Không nêu`.
- Không tự suy ra hoặc bổ sung thời hạn.

## Quy tắc phân loại

- **Kết luận:** Đã chốt, quyết định, thống nhất, chấp thuận hoặc yêu cầu gì?
- **Mục tiêu / Định hướng:** Muốn đạt trạng thái, chỉ tiêu hoặc định hướng gì?
- **Giao việc:** Ai phải thực hiện hành động cụ thể nào?
- **Nội dung chính:** Văn bản đề cập vấn đề gì?

### Nguyên tắc ưu tiên

- Có **ai + hành động cụ thể** → Giao việc.
- Có **hành động cụ thể nhưng chủ thể bị lược bỏ**, và chủ thể xác định chắc chắn từ context → Giao việc `(suy ra)`.
- Chỉ có **kết quả/trạng thái mong muốn ở cấp tổng thể** → Mục tiêu / Định hướng.
- Là **quyết định/chấp thuận/thống nhất/yêu cầu được chốt** → Kết luận.
- Không biến mục tiêu thành giao việc nếu văn bản không xác định ai phải thực hiện.
- Không biến giao việc thành mục tiêu chỉ vì nhiệm vụ hướng tới một kết quả.
- Không tạo giao việc mới chỉ từ quan hệ nghiệp vụ hoặc kiến thức bên ngoài.

Dưới bảng ghi:

**Hiệu lực:** thời điểm văn bản có hiệu lực và văn bản bị thay thế (nếu có).

## Văn bản:

## {document}

Chỉ trả về bản tóm tắt, không thêm lời dẫn hoặc giải thích khác.

"""


def query_llm(prompt: str, model: str = MODEL, timeout: int = 600) -> str:
    """
    Gửi 1 prompt tới endpoint OpenAI-compatible (/chat/completions) và lấy kết quả.

    Args:
        prompt: nội dung prompt.
        model: tên model trên server (vd "google/gemma-4-26B-A4B-it").
        timeout: thời gian chờ tối đa (giây) — văn bản dài cần lâu hơn.

    Returns:
        Văn bản phản hồi từ model.
    """
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Không kết nối được tới LLM endpoint: {BASE_URL}") from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError("LLM phản hồi quá lâu (timeout). Hãy tăng timeout.") from e
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"LLM endpoint lỗi {response.status_code}: {response.text[:500]}") from e

    data = response.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def summarize_document(document: str, model: str = MODEL) -> str:
    """Tóm tắt văn bản (vd văn bản hành chính trích từ PDF qua OCR)."""
    if not document.strip():
        return "(Không có nội dung văn bản để tóm tắt.)"

    prompt = DOCUMENT_SUMMARY_PROMPT.format(document=document)
    return query_llm(prompt, model=model)
