import json
import os
from functools import lru_cache

KB_PATH = os.getenv(
    "KB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge_base.json"),
)


@lru_cache(maxsize=1)
def load_knowledge_base(path: str = KB_PATH) -> tuple[dict, ...]:
    with open(path, encoding="utf-8") as f:
        return tuple(json.load(f))


def render_knowledge_base(kb: tuple[dict, ...]) -> str:
    """Chuyển KB sang dạng liệt kê gọn (bỏ `n`) để đưa vào prompt."""
    lines = []
    for entry in kb:
        objects = list(dict.fromkeys(entry.get("objects", [])))  # bỏ trùng, giữ thứ tự
        lines.append(
            f"- **{entry['authority']}** — Phạm vi: {entry.get('context', '')}"
            f" | Từ khóa: {', '.join(entry.get('keywords', []))}"
            f" | Đối tượng: {', '.join(objects)}"
        )
    return "\n".join(lines)
