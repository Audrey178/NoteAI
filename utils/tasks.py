"""
Trích xuất danh sách nhiệm vụ có cấu trúc (JSON) từ văn bản hành chính, dùng cho API.

Mỗi nhiệm vụ gồm: tên nhiệm vụ, yêu cầu cần hoàn thành, đơn vị thực hiện, đơn vị phối hợp, thời hạn.

Quy tắc bóc tách:
- "A thực hiện, B phối hợp" → 01 nhiệm vụ: A thực hiện, B phối hợp.
- "A và B cùng thực hiện"   → 02 nhiệm vụ, mỗi đơn vị thực hiện một nhiệm vụ.
  (LLM trả `don_vi_thuc_hien` dạng danh sách; việc tách được làm chắc chắn trong code.)
"""

import json
import re

from utils.knowledge_base import load_knowledge_base, render_knowledge_base
from utils.llm import MODEL, query_llm

TASK_PROMPT = """\
Bạn là chuyên viên văn phòng chuyên theo dõi giao việc. Dựa vào văn bản hành chính dưới đây
(được trích xuất bằng OCR nên có thể còn một số lỗi chính tả nhỏ) và **Tham chiếu chức năng đơn vị**,
hãy liệt kê **tất cả nhiệm vụ** được giao trong văn bản.

# 1. Cách xác định nhiệm vụ

- Nhiệm vụ là nội dung văn bản xác định **phải thực hiện một hành động cụ thể**
  (chủ trì, tham mưu, xây dựng, tổ chức, hướng dẫn, kiểm tra, báo cáo, nghiên cứu, trình, thẩm định...).
- Văn bản quy định nhiệm vụ, quyền hạn của một cơ quan:
  mỗi khoản có hành động cụ thể là nhiệm vụ; chủ thể bị lược bỏ trong khoản đó chính là cơ quan đó.
- Nội dung chỉ là **quyền** ("được quyền...", "được ưu tiên...") thì không phải nhiệm vụ.
- Không biến mục tiêu, chỉ tiêu hoặc kết quả mong muốn thành nhiệm vụ nếu văn bản không thể hiện hành động.
- Không tự tạo hành động mới ngoài văn bản. Không bỏ sót điều, khoản nào có hành động cụ thể.

# 2. Quy tắc bóc tách nhiệm vụ

- "Đơn vị A chủ trì/thực hiện, (phối hợp với) Đơn vị B" → **01 nhiệm vụ**:
  `don_vi_thuc_hien` = ["A"], `don_vi_phoi_hop` = ["B"]. **Không** tạo nhiệm vụ riêng cho B.
- Chỉ đưa đơn vị vào `don_vi_phoi_hop` khi văn bản nêu vai trò **phối hợp** với nhiệm vụ đó.
  Đơn vị có **hành động riêng** (vd A lập hồ sơ, B thẩm định; A gửi, B có ý kiến) → mỗi hành động là một nhiệm vụ riêng.
- "Đơn vị A và Đơn vị B (cùng) thực hiện ..." → `don_vi_thuc_hien` = ["A", "B"] (hệ thống sẽ tự tách
  thành 02 nhiệm vụ, mỗi đơn vị một nhiệm vụ).
- Các nhiệm vụ khác nhau (khác hành động / khác đối tượng / khác thời hạn) → mỗi nhiệm vụ một phần tử,
  không gom chung.

# 3. Xác định đơn vị khi văn bản nêu chung chung

Áp dụng cho cả `don_vi_thuc_hien` và `don_vi_phoi_hop`
(vd "chủ trì, phối hợp với các cơ quan liên quan" → phải suy luận đích danh các đơn vị phối hợp).
Khi văn bản ghi chung chung ("các cơ quan liên quan", "các bộ, ngành liên quan", "cơ quan chức năng",
"các cơ quan tham mưu"...), suy luận đích danh đơn vị:
1. Xác định lĩnh vực, đối tượng, phạm vi của nhiệm vụ.
2. Tra **Tham chiếu**: cơ quan nào có Phạm vi / Từ khóa / Đối tượng khớp.
   Văn bản đang làm việc trong khối nào thì chọn cơ quan trong khối đó, KHÔNG LẤY CƠ QUAN CẤP CAO HƠN.
3. Kiểm tra với context: cơ quan có thuộc đúng nhóm văn bản nhắc tới không
   ("các cơ quan tham mưu của Trung ương Đảng" chỉ gồm khối Đảng; "các bộ, ngành" chỉ gồm khối Nhà nước).
   Cơ quan được văn bản nhắc tới ở chỗ khác là căn cứ mạnh hơn.
4. Chọn **tất cả** cơ quan qua được bước 2 và 3, chép đúng tên trong Tham chiếu.
   **Không** chép cụm chung chung vào danh sách đơn vị.
   Không cơ quan nào phù hợp → bỏ cụm đó khỏi danh sách (không bịa).

# 4. Định dạng trả về

Chỉ trả về **một mảng JSON** (không markdown, không lời dẫn), mỗi phần tử có đúng các khóa:

[
  {{
    "ten_nhiem_vu": "Tên ngắn gọn của nhiệm vụ (động từ + đối tượng), tối đa ~20 từ",
    "yeu_cau": "Yêu cầu cần hoàn thành: làm gì, về vấn đề gì, cho đối tượng nào, sản phẩm/kết quả đầu ra; bám sát câu chữ văn bản",
    "don_vi_thuc_hien": ["Tên đơn vị thực hiện/chủ trì"],
    "don_vi_phoi_hop": ["Tên đơn vị phối hợp"],
    "thoi_han": "Thời hạn đúng như văn bản nêu, hoặc null nếu không nêu",
    "can_cu": "Vị trí trong văn bản, vd 'Mục III, khoản 13'"
  }}
]

- `don_vi_phoi_hop` không có → [].
- Ngày văn bản có hiệu lực không phải thời hạn thực hiện.
- Không có nhiệm vụ nào → [].

# 5. Tham chiếu chức năng đơn vị

Đây **không** phải nội dung văn bản; chỉ dùng để suy luận đơn vị khi văn bản nêu chung chung.

{knowledge_base}

# 6. Văn bản

{document}
"""


def _parse_json_array(raw: str) -> list:
    """Lấy mảng JSON từ phản hồi LLM (chịu được ```json fences hoặc lời dẫn thừa)."""
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    if isinstance(data, dict):  # vd {"nhiem_vu": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), [])
    if not isinstance(data, list):
        raise ValueError("Phản hồi LLM không phải mảng JSON.")
    return data


# Cụm chung chung LLM đôi khi vẫn chép vào danh sách đơn vị ("các Bộ, ngành liên quan"...).
_GENERIC_UNIT = re.compile(r"^(các|những|cơ quan chức năng)\b|liên quan$", re.IGNORECASE)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    units = (str(v).strip() for v in value if v and str(v).strip())
    return list(dict.fromkeys(u for u in units if not _GENERIC_UNIT.search(u)))


def _as_text(value) -> str | None:
    text = str(value).strip() if value is not None else ""
    return None if text.lower() in ("", "null", "none", "không nêu") else text


def normalize_tasks(items: list) -> list[dict]:
    """Chuẩn hóa và tách nhiệm vụ: mỗi đơn vị thực hiện là một nhiệm vụ riêng.

    LLM trả khóa tiếng Việt (theo prompt); output của API dùng khóa tiếng Anh.
    """
    tasks = []
    for item in items:
        if not isinstance(item, dict):
            continue
        executors = _as_list(item.get("don_vi_thuc_hien"))
        coordinators = _as_list(item.get("don_vi_phoi_hop"))
        for executor in executors or [None]:
            tasks.append({
                "task_name": _as_text(item.get("ten_nhiem_vu")) or "",
                "requirement": _as_text(item.get("yeu_cau")) or "",
                "executing_unit": executor,
                "coordinating_units": [c for c in coordinators if c != executor],
                "deadline": _as_text(item.get("thoi_han")),
                "reference": _as_text(item.get("can_cu")),
            })
    return tasks


def extract_tasks(document: str, model: str = MODEL, retries: int = 1) -> list[dict]:
    """Trích xuất danh sách nhiệm vụ có cấu trúc từ văn bản."""
    if not document.strip():
        return []

    prompt = TASK_PROMPT.format(
        knowledge_base=render_knowledge_base(load_knowledge_base()),
        document=document,
    )
    for attempt in range(retries + 1):
        raw = query_llm(prompt, model=model)
        try:
            return normalize_tasks(_parse_json_array(raw))
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == retries:
                raise RuntimeError(f"Không đọc được JSON từ LLM: {raw[:500]}") from e
