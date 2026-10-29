"""
docx_reader.py
--------------
Đọc văn bản từ file Word (.docx) — không cần qua OCR vì text đã có sẵn.
Giữ đúng thứ tự đọc của đoạn văn và bảng trong thân văn bản.
"""

from io import BytesIO

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            text = " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
            # Ô gộp (merged) được python-docx trả lặp lại → bỏ trùng liền kề.
            if not cells or cells[-1] != text:
                cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_docx_text(content: bytes) -> str:
    """Trả về toàn bộ text của file .docx (đoạn văn + bảng) theo thứ tự xuất hiện."""
    try:
        doc = Document(BytesIO(content))
    except Exception as e:  # file hỏng / không phải docx
        raise ValueError("File .docx không hợp lệ hoặc bị hỏng.") from e

    blocks = []
    for item in doc.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
        else:
            text = _table_text(item)
        if text:
            blocks.append(text)
    return "\n".join(blocks)
