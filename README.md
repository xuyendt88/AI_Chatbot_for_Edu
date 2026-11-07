# 🎓 PathEdu - AI Chatbot Tư vấn Tuyển sinh & Hướng nghiệp

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![Google Gemini API](https://img.shields.io/badge/Google%20Gemini-API-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI_Chatbot_for_Edu** là giải pháp chatbot thông minh được thiết kế đặc biệt dành cho học sinh THPT (đặc biệt là học sinh lớp 12). Hệ thống kết hợp giữa **Machine Learning (ML)** dự đoán điểm thi, **RAG / Dữ liệu tuyển sinh thông minh** và **AI Agent** nhằm hỗ trợ tra cứu kết quả học tập, định hướng ngành nghề và chọn trường Đại học/Cao đẳng một cách tự nhiên, chính xác.

---

## 📌 Tính năng chính

- **Tra cứu kết quả học tập:** Tra cứu điểm trung bình các môn, xếp hạng học lực và điểm thi thử.
- **Dự đoán điểm thi thử (ML Predictor):** Sử dụng các mô hình Machine Learning để dự đoán điểm thi dựa trên kết quả học tập lớp 12.
- **Tư vấn Ngành học & Trường Đại học:** Khớp nối điểm dự đoán/thi thử với điểm chuẩn, học phí và tiêu chuẩn xét tuyển của các trường.
- **Giao diện Chatbot thông minh:** Tích hợp AI Agent để giải đáp thắc mắc tự nhiên theo ngôn ngữ giao tiếp hàng ngày.

---

## 📁 Cấu trúc thư mục & Dự án

```text
AI_Chatbot_for_Edu/
├── .github/              # Cấu hình GitHub Actions & Workflows CI/CD
├── data/                 # Thư mục chứa các tệp dữ liệu CSV đã làm sạch
│   ├── 01_Danh_sach_va_Thanh_tich_hoc_sinh_clean.csv
│   ├── 02_Ket_qua_va_Xep_hang_Hoc_tap_clean.csv
│   ├── 03_Nganh_hoc_va_Huong_nghiep_clean.csv
│   ├── 04_Tuyen_sinh_va_Hoc_phi_clean.csv
│   └── 05_Ket_qua_thi_thu_clean.csv
├── data_cleaning.py      # Script kiểm tra, xử lý giá trị khuyết và làm sạch dữ liệu
├── ml_model.py           # Script huấn luyện, đánh giá và lưu mô hình ML dự đoán điểm
├── tools.py              # Các hàm bổ trợ (Function Calling): tra cứu điểm, lọc trường, gợi ý ngành
├── agent.py              # Cấu hình AI Agent / LLM xử lý hội thoại và điều phối công cụ
├── app.py                # Giao diện ứng dụng người dùng (Streamlit)
├── requirements.txt      # Danh sách các thư viện Python phụ thuộc
├── my_work_log           # Nhật ký tiến độ công việc và ghi chú phát triển
└── README.md             # Tài liệu hướng dẫn dự án

```

## 🏗️ Sơ đồ Hoạt động & Kiến trúc Vận hành Hệ thống

Sơ đồ dưới đây minh họa chi tiết luồng xử lý tương tác giữa người dùng, giao diện Streamlit, các dịch vụ back-end, mô hình ML và Google Gemini API:

```text
streamlit run app.py

👤 [ Người dùng / Học sinh (User) ]
        │
        │ 1. Nhập câu hỏi tư vấn / tra cứu (Prompt) trên giao diện Web
        ▼
 🌐 [ app.py (Streamlit UI - Giao diện người dùng) ]
        │
        │ 2. Lưu câu hỏi vào st.session_state.messages (Duy trì lịch sử trò chuyện)
        │ 3. Gọi hàm điều phối AI Agent / LLM tương ứng
        ▼
 ⚙️ [ agent.py / tools.py (Back-end Logic & Function Calling) ]
        │
        ├─► Bước A: Tải & tiền xử lý tập dữ liệu tuyển sinh / học tập (CSV) bằng Pandas (df)
        │
        ├─► Bước B: Kích hoạt mô hình Machine Learning (ml_model.py)
        │     └─ Dự đoán điểm thi thử và đánh giá khả năng trúng tuyển của học sinh
        │
        ├─► Bước C: Truy xuất ngữ cảnh thông minh (Smart Context Retrieval)
        │     ├─ Chuẩn hóa tiếng Việt & quét từ khóa chuyên ngành, trường học, điểm chuẩn
        │     ├─ Lọc dữ liệu thống kê tổng quan (điểm trung bình, học lực, phổ điểm)
        │     └─ Đóng gói dữ liệu (data_insight) kèm System Instruction (Chống ảo giác / bịa đặt)
        │
        ├─► Bước D: Kết nối Google Gemini API (Client & Chat Session)
        │     ├─ Đính kèm lịch sử trò chuyện cũ (history_formatted)
        │     └─ Gửi prompt kèm ngữ cảnh dữ liệu nội bộ lên Google Gemini để tổng hợp câu trả lời
        │
        └─► Bước E: Nhận phản hồi (response.text) từ Gemini API và trả về cho app.py
        │
        ▼
 🌐 [ app.py (Streamlit UI) ]
        │
        │ 4. Hiển thị câu trả lời trực quan lên khung chat qua st.markdown()
        │ 5. Lưu phản hồi vào st.session_state.messages để duy trì mạch hội thoại liên tục
        ▼
 👤 [ Người dùng nhận được kết quả tư vấn tuyển sinh chính xác, cá nhân hóa ]
