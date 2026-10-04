# Reflection — Lab 19

**Tên:** VuHieuThien — 2A202602867
**Cohort:** A20-K4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**Kết quả trên 50 golden queries:**

| Loại query | Số | BM25 | Vector | Hybrid |
|---|---|---|---|---|
| exact | 15 | 96.7% | 88.7% | 96.7% |
| paraphrase | 15 | 33.3% | 24.0% | 32.0% |
| mixed | 20 | 97.0% | 98.5% | **100%** |

**Exact query:** BM25 thắng. Câu hỏi chứa từ kỹ thuật verbatim trong docs. Keyword signal đủ mạnh. Hybrid không thắng vì BM25 đã đủ tốt.

**Paraphrase query:** Cả ba mode đều yếu (<35%). Người dùng Việt dùng từ khác nhưng không chứa từ gốc. Vector giảm do model `bge-small-en-v1.5` training trên tiếng Anh.

**Mixed query:** Hybrid thắng rõ — 100% vs 97% BM25, 98.5% Vector. Câu hỏi thật có cả từ exact và ý tưởng paraphrased. BM25 bắt exact words. Vector bắt semantic. RRF hợp nhất cả hai.

**Khi không dùng hybrid:**

1. **Dùng pure BM25** khi query chứa từ kỹ thuật đặc thù (product code, API name, error code). BM25 nhanh hơn (P50=2.1ms vs 16.4ms hybrid). Vector không thêm giá trị.

2. **Dùng pure vector** khi muốn tìm kiếm semantic không dựa từ khóa (tìm tài liệu "giống nghĩa" chứ không phải "chứa từ"). Nhưng với model English-trained trên dữ liệu Việt, pure vector rủi ro cao. Nên dùng hybrid như fallback.

---

## Điều ngạc nhiên nhất khi làm lab này

Sự chênh lệch giữa single-shot và agentic retrieval trên NB6 — single-shot chỉ lấy được 8% một nửa câu hỏi ghép. RAG một-lượt trả lời "đúng một nửa" cho compound queries mà không ai phát hiện.

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _<tên đồng đội nếu có>_
