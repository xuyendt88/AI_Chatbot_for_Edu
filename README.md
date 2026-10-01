# AI_Chatbot_for_Edu

# 🎓 AI Chatbot for Edu — Hệ thống AI Tư vấn Tuyển sinh & Định hướng Hướng nghiệp

**AI_Chatbot_for_Edu** là giải pháp chatbot thông minh hỗ trợ học sinh THPT (đặc biệt là học sinh lớp 12) trong việc tra cứu điểm số, dự đoán kết quả thi thử bằng Machine Learning và tư vấn chọn ngành, chọn trường đại học/cao đẳng phù hợp.

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
├── app.py                # Giao diện ứng dụng người dùng (Streamlit / Gradio / FastAPI)
├── requirements.txt      # Danh sách các thư viện Python phụ thuộc
├── my_work_log           # Nhật ký tiến độ công việc và ghi chú phát triển
└── README.md             # Tài liệu hướng dẫn dự án
