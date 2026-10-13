"""PathEdu - ứng dụng hoàn chỉnh (giao diện + agent)"""

import base64
import json
import os
import uuid

import streamlit as st
from agent import build_agent, get_final_result
from tools import get_academic_ranking, get_student_info  

# ---- 0. Cấu hình giao diện (đổi ở đây là đủ) ----
TEN_APP = "PathEdu – La bàn chọn ngành cho học sinh THPT"
HINH_NEN = "assets/hinh_nen.png"   # đổi ảnh khác: chép ảnh vào thư mục assets
DO_MO_NEN = 0.55                   # 0 = ảnh rõ nét, 1 = ẩn hẳn ảnh
AVATAR = {"user": "🧑‍🎓", "assistant": "🧭"}


def dat_hinh_nen(duong_dan: str, do_mo: float) -> None:
    """Chèn ảnh làm nền mờ phía sau toàn bộ app. Không có ảnh thì bỏ qua, không làm app lỗi."""
    if not os.path.exists(duong_dan):
        return
    with open(duong_dan, "rb") as f:
        anh_b64 = base64.b64encode(f.read()).decode()
    lop_phu = f"rgba(247,245,239,{do_mo})"
    st.markdown(
        f"""<style>
        .stApp {{
            background-image: linear-gradient({lop_phu}, {lop_phu}),
                              url("data:image/png;base64,{anh_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        [data-testid="stBottom"] > div {{ background: transparent; }}
        </style>""",
        unsafe_allow_html=True,
    )


#  1. Cấu hình trang
st.set_page_config(page_title="PathEdu – La bàn chọn ngành", page_icon="🧭")
dat_hinh_nen(HINH_NEN, DO_MO_NEN)
st.title(f"🧭 {TEN_APP}")
st.caption("Xem học lực · Dự đoán điểm thi · Gợi ý ngành & trường phù hợp — dành cho học sinh và phụ huynh")

#  3. Sổ ghi nhớ
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

cau_hoi_chon = None  # câu hỏi đến từ nút bấm (câu mẫu hoặc ô tra cứu nhanh)

#  4. Thanh bên
with st.sidebar:
    st.subheader("Giới thiệu")
    st.write("PathEdu giúp em xem kết quả học tập, gợi ý ngành và trường phù hợp.")
    st.caption(f"Mã cuộc trò chuyện: {st.session_state.thread_id[:8]}")

    if st.button("🔄 Cuộc trò chuyện mới", width="stretch", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pop("last_error", None)
        st.toast("Đã bắt đầu cuộc trò chuyện mới", icon="✨")
        st.rerun()

    st.divider()

    # ---- 4b. Tra cứu nhanh mã học sinh ----
    st.subheader("🔍 Tra cứu nhanh")
    ma_hs = st.text_input("Mã học sinh", placeholder="VD: HS1186").strip().upper()
    if ma_hs:
        ho_so = json.loads(get_student_info(ma_hs))
        if "error" in ho_so:
            st.warning(ho_so["error"])
        else:
            xep_hang = json.loads(get_academic_ranking(ma_hs))
            st.success(f'**{ho_so["ho_va_ten"]}** — Lớp {ho_so["lop"]} · Khối {ho_so["khoi"]}')
            if "error" not in xep_hang:
                st.metric("Điểm trung bình chung", xep_hang["diem_trung_binh_chung"],
                          help=xep_hang["so_sanh_voi_lop"])
                st.caption(xep_hang["so_sanh_voi_toan_khoi"])
            with st.expander("📋 Bảng điểm 10 môn"):
                st.dataframe(
                    {"Môn": list(ho_so["bang_diem"]), "Điểm": list(ho_so["bang_diem"].values())},
                    hide_index=True, width="stretch",
                )
            if st.button("💬 Nhờ PathEdu tư vấn cho em này", width="stretch"):
                cau_hoi_chon = f"Phân tích kết quả học tập của {ma_hs} và gợi ý ngành, trường phù hợp"

    st.divider()
    st.markdown(
        "**📌 Tính năng chính**\n"
        "- 📊 **Tra cứu học tập:** điểm các môn, xếp hạng, điểm thi thử\n"
        "- 🤖 **Dự đoán điểm thi:** mô hình ML dựa trên học bạ lớp 12\n"
        "- 🏫 **Tư vấn ngành & trường:** so điểm với điểm chuẩn, học phí\n"
        "- 💬 **Hỏi đáp tự nhiên:** trò chuyện như nhắn tin với AI"
    )

# ---- 5. Vẽ lại lịch sử chat ----
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATAR[message["role"]]):
        st.markdown(message["content"])

# ---- 6. Nút câu hỏi mẫu ----
CAU_HOI_MAU = [
    "📊 Cho em xem kết quả học tập của HS1023",
    "🧩 Em giỏi Toán, Lý thì nên học ngành gì?",
    "🏫 Ngành Công nghệ thông tin có những trường nào?",
]
if not st.session_state.messages:
    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        st.markdown(
            "Chào em! 👋 Mình là **PathEdu**. Em có thể hỏi về điểm số, ngành học hoặc trường đại học. "
            "Chưa biết hỏi gì thì bấm thử một câu bên dưới nhé:"
        )
    for cau in CAU_HOI_MAU:
        if st.button(cau, width="stretch"):
            cau_hoi_chon = cau.split(" ", 1)[1]  # bỏ biểu tượng ở đầu câu

# ---- 7. Ô nhập tin nhắn ----
prompt = st.chat_input("Em muốn hỏi gì về ngành học, trường đại học?") or cau_hoi_chon

# ---- 8. Xử lý tin nhắn mới ----
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        try:
            with st.spinner("🧭 PathEdu đang tra cứu dữ liệu..."):
                result = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
            answer = get_final_result(result)

            st.session_state.pop("last_error", None)
        except Exception as exc:
            answer = "Xin lỗi, mình chưa xử lý được câu hỏi này. Em thử lại hoặc hỏi theo cách khác nhé."
            st.session_state.last_error = str(exc)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ---- 9. Khung lỗi ----
if "last_error" in st.session_state:
    with st.expander("⚠️ Chi tiết lỗi (dành cho nhóm phát triển)"):
        st.code(st.session_state.last_error)

st.caption("PathEdu · Dự án học tập FPT Jetking · Dữ liệu giả lập — kết quả dự đoán chỉ mang tính tham khảo")
