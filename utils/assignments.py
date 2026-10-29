"""
Trích xuất giao việc từ văn bản hành chính: xác định đích danh đơn vị nào làm nhiệm vụ gì.

- Nhiệm vụ trực tiếp: đơn vị thực hiện xác định được từ chính văn bản.
- Nhiệm vụ gián tiếp: văn bản chỉ nêu chung chung ("các cơ quan liên quan"...),
  phải suy luận từ context văn bản + knowledge base để chỉ ra đích danh đơn vị.
"""

from utils.knowledge_base import load_knowledge_base, render_knowledge_base
from utils.llm import MODEL, query_llm

ASSIGNMENT_PROMPT = """\
Bạn là chuyên viên văn phòng chuyên theo dõi giao việc. Dựa vào văn bản hành chính dưới đây
(được trích xuất bằng OCR nên có thể còn một số lỗi chính tả nhỏ) và **Tham chiếu chức năng đơn vị**,
hãy xác định **đích danh đơn vị nào phải làm nhiệm vụ gì**.

# 1. Cách xác định nhiệm vụ

- Nhiệm vụ là nội dung văn bản xác định **phải thực hiện một hành động cụ thể**
  (chủ trì, phối hợp, tham mưu, xây dựng, tổ chức, hướng dẫn, kiểm tra, báo cáo, thi hành...).
- Văn bản quy định nhiệm vụ, quyền hạn của một cơ quan: 
  mỗi khoản có hành động cụ thể là nhiệm vụ; chủ thể bị lược bỏ trong khoản đó chính là cơ quan A.
- Nội dung chỉ là **quyền** ("được quyền...", "được ưu tiên...") thì không phải nhiệm vụ;
  nhưng trách nhiệm đi kèm của cơ quan khác ("các cơ quan... có trách nhiệm...") là nhiệm vụ.
- Không biến mục tiêu, chỉ tiêu hoặc kết quả mong muốn thành nhiệm vụ nếu văn bản không thể hiện hành động.
- Không tự tạo hành động mới ngoài văn bản. Mỗi nhiệm vụ độc lập là một dòng; không gom nhiều nhiệm vụ khác nhau.
- Không bỏ sót điều, khoản nào có hành động cụ thể.

# 2. Nhiệm vụ trực tiếp

Đơn vị thực hiện **xác định được từ chính văn bản**:
- được nêu đích danh trong câu (vd "Bộ Quốc phòng chủ trì rà soát chế độ xuất ngũ..."), hoặc
- bị lược bỏ nhưng xác định chắc chắn từ context văn bản (vd khoản thuộc quy định nhiệm vụ của cơ quan A).

Câu dạng "X chủ trì, phối hợp với Y làm Z" → X — Chủ trì — Z.
Nếu Y được nêu đích danh (vd "phối hợp với Bộ Tài chính", "giữa bốn Văn phòng: ...") → thêm dòng Y — Phối hợp — Z.
Văn bản liệt kê nhiều đơn vị cùng thực hiện → ghi **đầy đủ** danh sách, không thu hẹp.

Trong bảng nhiệm vụ trực tiếp:
- cột Đơn vị thực hiện **không** chứa cụm chung chung ("các cơ quan tham mưu", "các cơ quan liên quan"...);
- cột Nhiệm vụ **không** chứa phần "phối hợp với các cơ quan liên quan", "điều phối các cơ quan chức năng"...
  (phần đó thuộc nhiệm vụ gián tiếp).

# 3. Nhiệm vụ gián tiếp (phải suy luận đích danh đơn vị)

Là khi đơn vị chỉ được nêu chung chung hoặc không nêu, vd:
"phối hợp với các cơ quan liên quan", "các cơ quan tham mưu", "cơ quan chức năng", "các bộ, ngành liên quan",
Các đơn vị PHẢI ĐƯỢC PHÂN CÔNG LÀM MỘT NHIỆM VỤ CỤ THỂ .

Với từng trường hợp, suy luận theo từng bước:
1. **Xác định việc cần làm là gì**: lĩnh vực, đối tượng, phạm vi của nhiệm vụ.
2. **Tra Tham chiếu**: cơ quan nào có Phạm vi / Từ khóa / Đối tượng khớp với việc đó.
   Chú ý các lưu ý phân biệt cơ quan trong Phạm vi của văn bản (Văn bản đang làm việc trong khối nào thì cơ quan trong khối đó chịu trách nhiệm KHÔNG LẤY CƠ QUAN CẤP CAO HƠN).
3. **Kiểm tra với context văn bản**: cơ quan đó có thuộc đúng nhóm mà văn bản nhắc tới không
   (vd "các cơ quan tham mưu của Trung ương Đảng" chỉ gồm cơ quan khối Đảng, không gồm các Bộ;
   "các bộ, ngành" chỉ gồm cơ quan khối Nhà nước), có phù hợp với cơ quan ban hành, đối tượng áp dụng,
   nơi nhận của văn bản không. Cơ quan được văn bản nhắc tới ở chỗ khác là căn cứ mạnh hơn.
4. Chọn các cơ quan **qua được cả bước 2 và 3**.

Bắt buộc:
- Đơn vị thực hiện **phải là tên một cơ quan có trong Tham chiếu**, chép đúng tên;
  **không được** chép lại cụm chung chung làm đơn vị.
- Cụm chỉ **một nhóm** (vd: "các cơ quan tham mưu của Trung ương Đảng", "các bộ, ngành") → xét **tất cả** cơ quan
  trong Tham chiếu thuộc nhóm đó, chọn các cơ quan **có phạm vi liên quan đến nhiệm vụ**, mỗi cơ quan một dòng.
  **Không cần** tìm ra một cơ quan duy nhất: cụm chỉ nhiều cơ quan thì ghi nhiều cơ quan;
  "cụm từ chỉ chung nhiều cơ quan" **không phải** lý do để ghi Chưa xác định.
- "Các cơ quan liên quan" không phải lý do để bỏ qua: lĩnh vực được xác định từ **chính nội dung nhiệm vụ**.
- Không cơ quan nào qua được bước 2 và 3 → ghi vào mục Chưa xác định được đơn vị, nêu đã xét cơ quan nào và vì sao loại.

Mức tin cậy: `Cao` — khớp rõ cả Tham chiếu và context; `Trung bình` — khớp Tham chiếu nhưng context chưa khẳng định.

## Ví dụ

Câu: "Văn phòng Chính phủ chủ trì, phối hợp với các bộ, ngành liên quan xây dựng cơ chế hỗ trợ hộ kinh doanh
chuyển sang sử dụng hóa đơn điện tử" (Điều 1, khoản 3)

- Nhiệm vụ trực tiếp: Văn phòng Chính phủ — Chủ trì — Xây dựng cơ chế hỗ trợ hộ kinh doanh chuyển sang sử dụng hóa đơn điện tử
- Nhiệm vụ gián tiếp: Bộ Tài Chính — Phối hợp — Xây dựng cơ chế hỗ trợ hộ kinh doanh chuyển sang sử dụng hóa đơn điện tử —
  "các bộ, ngành liên quan" — Tham chiếu: phạm vi thuế, hóa đơn điện tử, đối tượng hộ kinh doanh;
  context: "bộ, ngành" là khối Nhà nước, Bộ Tài Chính thuộc nhóm này — Cao
- Không chọn Ủy ban Kinh tế và Tài chính dù có từ khóa "hóa đơn điện tử", vì đó là cơ quan của Quốc hội,
  không thuộc nhóm "bộ, ngành".

# 4. Định dạng trả về

Trả về **đúng 4 mục theo thứ tự sau** (làm phần suy luận trước, bảng nhiệm vụ trực tiếp sau cùng):

## Nhiệm vụ gián tiếp

| STT | Đơn vị thực hiện | Vai trò | Nhiệm vụ | Cụm từ trong văn bản | Lý do (Tham chiếu + context) | Tin cậy | Căn cứ |
|---|---|---|---|---|---|---|---|

Mỗi cơ quan ở dòng "Kết luận" phía trên là một dòng trong bảng này.

## Chưa xác định được đơn vị

| STT | Cụm từ trong văn bản | Nhiệm vụ | Lý do | Căn cứ |
|---|---|---|---|---|

## Nhiệm vụ trực tiếp

| STT | Đơn vị thực hiện | Vai trò | Nhiệm vụ | Thời hạn | Căn cứ |
|---|---|---|---|---|---|

Quy ước cột:
- **Vai trò:** Vai trò trong nhiệm vụ đấy.
- **Nhiệm vụ:** ghi cụ thể làm gì, về vấn đề gì, cho đối tượng nào; bám sát câu chữ văn bản.
- **Thời hạn:** đúng thời hạn thực hiện nêu trong văn bản; không nêu → `Không nêu`.
  Ngày văn bản có hiệu lực không phải thời hạn thực hiện.
- **Căn cứ:** vị trí trong văn bản (vd `Điều 2, khoản 13`).
- Mục nào không có dòng nào → ghi `Không có`. Giữ đúng thứ tự cột như tiêu đề bảng.

Chỉ trả về 4 mục trên, không thêm lời dẫn hoặc giải thích khác.

# 5. Tham chiếu chức năng đơn vị

Đây **không** phải nội dung văn bản; chỉ dùng để suy luận đơn vị cho nhiệm vụ gián tiếp.

{knowledge_base}

# 6. Văn bản

{document}
"""


def extract_assignments(document: str, model: str = MODEL) -> str:
    """Trích xuất giao việc (trực tiếp + gián tiếp suy luận bằng KB), trả về Markdown."""
    if not document.strip():
        return "(Không có nội dung văn bản.)"

    prompt = ASSIGNMENT_PROMPT.format(
        knowledge_base=render_knowledge_base(load_knowledge_base()),
        document=document,
    )
    return query_llm(prompt, model=model)
