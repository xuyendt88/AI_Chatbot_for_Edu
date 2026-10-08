"""PathEdu - ứng dụng hoàn chỉnh (giao diện + agent)."""

import uuid
import streamlit as st
from agent import build_agent, get_final_text

#  1. Cấu hình trang 
st.set_page_config(page_title="PathEdu", page_icon="🎓")
st.title("🎓 PathEdu")
st.caption("Trợ lý tra cứu học tập và tư vấn hướng nghiệp — bản thử nghiệm học tập")

#  3.sổ ghi nhớ 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    try:
        st.session_state.agent = build_agent()
    except Exception as exc:
        st.error("Không khởi tạo được trợ lý. Kiểm tra API key, tên model và thư viện.")
        st.exception(exc)
        st.stop()

#  4. Thanh bên 
with st.sidebar:
    st.subheader("Giới thiệu")
    st.write("PathEdu giúp em xem kết quả học tập, gợi ý ngành và trường phù hợp.")
    st.caption(f"Mã cuộc trò chuyện: {st.session_state.thread_id[:8]}")

    if st.button("🔄 Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pop("last_error", None)
        st.rerun()

    st.divider()
    st.markdown(
        "**Phạm vi an toàn**\n"
        "- Dữ liệu học sinh là giả lập\n"
        "- Kết quả dự đoán chỉ để tham khảo\n"
        "- Trợ lý không quyết định thay em\n"
        "- Học sinh không sửa được dữ liệu"
    )

# ---- 5. Vẽ lại lịch sử chat (kèm Tool trace của từng câu trả lời) ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# ---- 6. Nút câu hỏi mẫu ----
CAU_HOI_MAU = [
    "Cho em xem kết quả học tập của HS1023",
    "Em giỏi Toán, Lý thì nên học ngành gì?",
    "Ngành CNTT có những trường nào?",
]
cau_hoi_chon = None
if not st.session_state.messages:
    st.write("Em có thể bắt đầu bằng một câu hỏi mẫu:")
    for cau in CAU_HOI_MAU:
        if st.button(cau, use_container_width=True):
            cau_hoi_chon = cau

# ---- 7. Ô nhập tin nhắn ----
prompt = st.chat_input("Em muốn hỏi gì về ngành học, trường đại học?") or cau_hoi_chon

# ---- 8. Xử lý tin nhắn mới ----
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # trace = []
    with st.chat_message("assistant"):
        try:
            with st.spinner("PathEdu đang tra cứu..."):
                result = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
            answer = get_final_text(result)
            
            st.session_state.pop("last_error", None)
        except Exception as exc:
            answer = "Xin lỗi, mình chưa xử lý được câu hỏi này. Em thử lại hoặc hỏi theo cách khác nhé."
            st.session_state.last_error = str(exc)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "trace": trace})

# ---- 9. Khung lỗi ----
if "last_error" in st.session_state:
    with st.expander("⚠️ Chi tiết lỗi (dành cho nhóm phát triển)"):
        st.code(st.session_state.last_error)
