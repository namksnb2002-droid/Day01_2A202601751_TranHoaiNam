# K3 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 9h00–13h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> *Khi temperature tăng từ 0.0 đến 1.5, độ ngẫu nhiên và tính sáng tạo của phản hồi tăng dần. Ở mức 0.0–0.5, các câu trả lời rất nhất quán, tập trung và mang tính sự thật cao. Khi lên mức 1.0–1.5, văn phong trở nên đa dạng, phong phú hơn nhưng dễ xuất hiện tình trạng lan man hoặc thông tin thiếu chính xác.*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> *Tôi sẽ đặt temperature trong khoảng 0.0 đến 0.2 (hoặc chọn 0.0). Lý do vì chatbot hỗ trợ khách hàng cần ưu tiên tuyệt đối tính chính xác, ổn định và nhất quán; giá trị thấp giúp hạn chế tối đa việc AI tự bịa ra thông tin (hallucination) về chính sách, giá cả hay dịch vụ.*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> *1. Tính toán ước tính chi phí:Tổng lượng token đầu ra/ngày: $10.000 \text{ người} \times 3 \text{ lượt} \times 350 \text{ token} = 10.500.000 \text{ token}$ ($10,5$ triệu tokens/ngày).Chi phí GPT-4o-mini (đơn giá output $\approx \$0.60 / 1\text{M tokens}$):$$\text{Chi phí/ngày} = 10.5 \times \$0.60 = \$6.30 \quad (\approx \$189 / \text{tháng})$$Chi phí GPT-4o (đơn giá output $\approx \$10.00 / 1\text{M tokens}$):$$\text{Chi phí/ngày} = 10.5 \times \$10.00 = \$105.00 \quad (\approx \$3.150 / \text{tháng})$$Kết luận: GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần (hoặc lên tới 25 lần tùy theo biểu giá output thực tế). Với quy mô $30.000$ lượt gọi/ngày, việc dùng GPT-4o sẽ tiêu tốn thêm hàng ngàn USD mỗi tháng.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> *Phản hồi từ persona "giáo viên tiểu học" rất ngắn gọn, sử dụng từ ngữ mộc mạc và hình ảnh ví dụ gần gũi (như cuốn sổ ghi chép chung của lớp) giúp trẻ em dễ hiểu. Ngược lại, persona "chuyên gia tài chính" cho câu trả lời chi tiết hơn, tràn ngập các thuật ngữ chuyên ngành (như sổ cái phân tán, mã hóa bất đối xứng, cơ chế đồng thuận) với văn phong trang trọng. Điều này cho thấy system prompt đóng vai trò định hình bối cảnh (contextual framing), giúp điều hướng toàn bộ từ vựng, độ sâu kiến thức và giọng văn của model mà không cần thay đổi nội dung câu hỏi gốc.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> *1. Thử nghiệm so sánh trên đoạn văn mẫu (~100 từ tiếng Việt):Đoạn văn thử nghiệm (100 từ): "Trí tuệ nhân tạo đang phát triển với tốc độ chưa từng có, thay đổi cách chúng ta làm việc và giao tiếp hàng ngày. Việc hiểu rõ cách thức vận hành của các mô hình ngôn ngữ lớn giúp các lập trình viên tối ưu hóa chi phí và hiệu năng khi xây dựng sản phẩm. Trong thực tế, việc quản lý token đóng vai trò vô cùng quan trọng đối với ngân sách của dự án."*
    Số từ thực tế: $100$ từ.
    Ước lượng theo công thức Part 1 (số từ / 0.75): $100 / 0.75 \approx \mathbf{133}$ tokens.
    Đếm thực tế bằng tiktoken (cl100k_base): $\mathbf{212}$ tokens.
    Chênh lệch: Số token đếm bằng tiktoken cao hơn ước lượng công thức khoảng $59.4\%$ (chênh lệch rất lớn).
    2. Nguyên nhân tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài:
    Bảng từ vựng (Vocabulary) ưu tiên tiếng Anh: Bộ mã hóa BPE (Byte-Pair Encoding) của OpenAI được huấn luyện chủ yếu trên dữ liệu tiếng Anh. Các từ tiếng Anh phổ biến thường được mã hóa thành 1 token duy nhất.
    Xử lý ký tự có dấu (Diacritics): Tiếng Việt chứa nhiều nguyên âm có dấu (như á, à, ả, ã, ạ, ê, ơ, ư...). Bộ tokenizer thường phải tách các từ/âm tiết tiếng Việt thành nhiều mảnh sub-word hoặc chuỗi byte nhỏ lẻ, khiến $1$ từ tiếng Việt trung bình tốn từ $1.5$ đến $2.5$ tokens.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> *Streaming quan trọng nhất trong các ứng dụng tương tác trực tiếp với con người (như Chatbot, trợ lý giao tiếp thời gian thực, công cụ hỗ trợ viết lách) nơi câu trả lời dài, giúp giảm thiểu thời gian chờ đợi cảm nhận (perceived latency) bằng cách hiển thị từng token ngay khi chúng xuất hiện. Ngược lại, non-streaming lại phù hợp hơn cho các tác vụ chạy ngầm giữa hệ thống với hệ thống (như trích xuất dữ liệu tự động, gọi hàm/function calling, xuất dữ liệu cấu trúc JSON hay xử lý hàng loạt), nơi chương trình cần nhận đầy đủ và nguyên vẹn toàn bộ dữ liệu phản hồi trước khi thực thi các bước logic tiếp theo.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> *Exponential backoff giúp hệ thống bị quá tải có thêm khoảng nghỉ tăng dần ($0.1s, 0.2s, 0.4s...$) để giải phóng tài nguyên và bình phục dần dần thay vì liên tục bị dồn nén yêu cầu. Nếu hàng nghìn client cùng sử dụng delay cố định giống nhau (ví dụ 1 giây), tất cả sẽ đồng loạt gửi lại request đúng sau mỗi 1 giây, tạo ra các "làn sóng" truy cập khổng lồ trùng thời điểm (hiện tượng Thundering Herd hay Retry Storm). Sự dồn dập lặp đi lặp lại này sẽ khiến server tiếp tục sập liên tục và không bao giờ có cơ hội hồi phục.*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> *Persona đã chọn: Trợ lý lập trình viên Python cao cấp (Senior Python Mentor).*

    System prompt:
    "Bạn là một trợ lý lập trình viên Python giàu kinh nghiệm. Hãy trả lời ngắn gọn, tập trung trực tiếp vào bản chất vấn đề, cung cấp ví dụ code minh họa tối giản và luôn phản hồi bằng tiếng Việt."

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> *Hạn chế lớn nhất của trợ lý: Cắt cứng bộ nhớ ngữ cảnh ở 3 lượt hội thoại gần nhất (history[-6:]) và chỉ lưu trong RAM. Điều này khiến trợ lý hoàn toàn "quên" các thông tin, yêu cầu hoặc ràng buộc quan trọng mà người dùng đã thiết lập ở đầu cuộc trò chuyện nếu cuộc hội thoại kéo dài quá 3 lượt.*

    Đề xuất cải thiện: Triển khai Cơ chế tóm tắt ngữ cảnh tự động (Conversation Summarization).

    Mô tả cách triển khai: Khi lịch sử hội thoại vượt quá 3 lượt (hoặc chạm ngưỡng token quy định), thay vì xóa bỏ hoàn toàn các tin nhắn cũ, hệ thống sẽ gọi một model nhẹ (gpt-4o-mini) để tóm tắt các lượt hội thoại trước đó thành một đoạn văn ngắn (summary). Sau đó, chèn đoạn summary này vào đầu danh sách history dưới dạng tin nhắn context. Cách làm này giúp giữ lại các thông tin cốt lõi dài hạn mà không làm bùng nổ số lượng token gửi kèm.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/` và zip theo hướng dẫn README
