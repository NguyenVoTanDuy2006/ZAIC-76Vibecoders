# Semantic score cho benchmark ASR

Benchmark dùng BERTScore F1 với `vinai/phobert-base` làm bộ mã hóa tiếng Việt. BERTScore so khớp embedding có ngữ cảnh của các token trong bản chép và câu chuẩn; điểm cao hơn nghĩa là nội dung gần nhau hơn. Điểm được tính cho từng mẫu, sau đó lấy trung bình số học trên các mẫu có bản chép. Bảng báo cáo ghi cả câu gốc (`raw_bertscore_f1`) và câu đã chuẩn hóa (`normalized_bertscore_f1`). Điểm chưa được rescale và không dùng IDF; không coi đây là tỷ lệ phần trăm chính xác.

PhoBERT cần văn bản đã tách từ. Pipeline dùng PyVi để tách từ trước khi gọi BERTScore. Tài liệu PhoBERT khuyên dùng RDRSegmenter của VnCoreNLP vì đó là bộ tách từ dùng khi huấn luyện; PyVi được chọn để pipeline Kaggle gọn và không cần tải thêm tài nguyên Java. Đây là một nguồn sai khác cần lưu ý khi diễn giải điểm.

PhoBERT là lựa chọn đã chốt cho phép đo này. BamiBERT là encoder tiếng Việt mới hơn và theo công bố của tác giả đạt kết quả cao hơn PhoBERT trên nhiều benchmark NLP; kết quả đó không tự chứng minh BamiBERT cho BERTScore tương quan tốt hơn với đánh giá con người trên tập ASR này. Nếu đổi encoder, điểm của hai lần chạy không còn so sánh trực tiếp được.

Số tham số ASR được đếm từ `model.parameters()` ngay sau khi mỗi GPU worker tải checkpoint. Đây là tổng tham số đang có trong model ASR, không cộng processor, bộ nhớ tạm hoặc PhoBERT. PhoBERT có số tham số riêng trong `semantic_model_parameter_count`. Tên “0.6B” và “1.7B” là quy mô công bố làm tròn; `parameter_count` là con số đo thực tế của checkpoint được chạy.

Nguồn: [PhoBERT](https://huggingface.co/vinai/phobert-base), [hướng dẫn tách từ của PhoBERT](https://huggingface.co/vinai/phobert-base-v2), [BamiBERT](https://arxiv.org/abs/2607.02259), [BERTScore](https://github.com/Tiiiger/bert_score).
