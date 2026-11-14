"""PathEdu - ứng dụng hoàn chỉnh (giao diện + agent). Bản nâng cấp giao diện sáng sủa, hiện đại."""

import base64
import json
import os
import uuid

import streamlit as st
from agent import build_agent, get_final_result
from tools import get_academic_ranking, get_student_info  

# ---- 0. Cấu hình trang & Giao diện sáng sủa ----
TEN_APP = "PathEdu – Trợ lý học tập cho học sinh THPT"
HINH_NEN = "assets/hinh_nen.png"   
DO_MO_NEN = 0.92                   # Tăng độ mờ nền để giao diện sáng và nổi bật nội dung chính
AVATAR = {"user": "🧑‍🎓", "assistant": "🧭"}

st.set_page_config(
    page_title="PathEdu – Trợ lý học tập",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def dat_hinh_nen(duong_dan: str, do_mo: float) -> None:
    """Chèn CSS tùy chỉnh giúp giao diện sáng sủa, bo tròn mềm mại và hiện đại hơn."""
    bg_style = ""
    if os.path.exists(duong_dan):
        with open(duong_dan, "rb") as f:        # rb chế độ đọc nhị phân
            anh_b64 = base64.b64encode(f.read()).decode()
        lop_phu = f"rgba(255,255,255,{do_mo})"
        bg_style = f"""
        .stApp {{
            background-image: linear-gradient({lop_phu}, {lop_phu}),
                              url("data:image/png;base64,{anh_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg_style = ".stApp { background-color: #F8FAFC; }"

    st.markdown(
        f"""<style>
        {bg_style}
        /* Tinh chỉnh khung chat input */
        [data-testid="stBottom"] > div {{ background: transparent; }}
        [data-testid="stChatInput"] > div {{
            border: 2px solid #0D9488; 
            border-radius: 16px; 
            background: #FFFFFF;
            box-shadow: 0 4px 20px rgba(13,148,136,0.12);
        }}
        [data-testid="stChatInput"] textarea {{ 
            font-size: 1.05rem; 
            color: #1E293B; 
            caret-color: #0D9488; 
        }}
        [data-testid="stChatInput"] textarea::placeholder {{ 
            color: #94A3B8; 
            opacity: 1; 
        }}
        [data-testid="stChatInputSubmitButton"] {{ 
            background: #0D9488; 
            color: #FFFFFF; 
            border-radius: 10px;
        }}
        /* Bo tròn và làm nổi bật các khối container / expander */
        .streamlit-expanderHeader {{
            background-color: #F1F5F9;
            border-radius: 10px;
        }}
        </style>""",
        unsafe_allow_html=True,
    )

dat_hinh_nen(HINH_NEN, DO_MO_NEN)

# Tiêu đề chính tinh gọn, sáng sủa
st.title(f"🧭 {TEN_APP}")
st.caption("✨ Xem học lực · Dự đoán điểm thi · Gợi ý ngành & trường phù hợp — dành cho học sinh và phụ huynh")
st.markdown("---")

# ---- 1. Quản lý Session State ----
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

cau_hoi_chon = None  

# ---- 2. Thanh bên (Sidebar) hiện đại, gọn gàng ----
with st.sidebar:
    st.subheader("💡 Giới thiệu")
    st.markdown("PathEdu giúp em tra cứu kết quả học tập, định hướng ngành nghề và chọn trường đại học phù hợp.")
    st.caption(f"Mã phiên: `{st.session_state.thread_id[:8]}`")

    if st.button("🔄 Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pop("last_error", None)
        st.toast("Đã bắt đầu cuộc trò chuyện mới!", icon="✨")
        st.rerun()

    st.markdown("---")

    # Tra cứu nhanh mã học sinh
    st.subheader("🔍 Tra cứu học sinh")
    ma_hs = st.text_input("Nhập mã học sinh", placeholder="VD: HS1186").strip().upper()
    if ma_hs:
        ho_so = json.loads(get_student_info(ma_hs))
        if "error" in ho_so:
            st.warning(ho_so["error"])
        else:
            xep_hang = json.loads(get_academic_ranking(ma_hs))
            st.success(f"**{ho_so['ho_va_ten']}**\n\nLớp {ho_so['lop']} · Khối {ho_so['khoi']}")
            if "error" not in xep_hang:
                st.metric("Điểm trung bình chung", xep_hang["diem_trung_binh_chung"],
                          help=xep_hang["so_sanh_voi_lop"])
                st.caption(xep_hang["so_sanh_voi_toan_khoi"])
            with st.expander("📋 Xem bảng điểm 10 môn"):
                st.dataframe(
                    {"Môn": list(ho_so["bang_diem"]), "Điểm": list(ho_so["bang_diem"].values())},
                    hide_index=True, use_container_width=True,
                )
            if st.button("💬 Tư vấn cho học sinh này", use_container_width=True):
                cau_hoi_chon = f"Phân tích kết quả học tập của {ma_hs} và gợi ý ngành, trường phù hợp"

    st.markdown("---")
    st.markdown(
        "**📌 Tính năng nổi bật**\n"
        "- 📊 **Tra cứu học tập:** Điểm số & xếp hạng\n"
        "- 🤖 **Dự đoán điểm thi:** Mô hình ML học bạ\n"
        "- 🏫 **Tư vấn hướng nghiệp:** So sánh điểm chuẩn\n"
        "- 💬 **Hỏi đáp tự nhiên:** Trò chuyện thông minh"
    )

# ---- 3. Hiển thị lịch sử trò chuyện ----
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATAR[message["role"]]):
        st.markdown(message["content"])

# ---- 4. Khối câu hỏi gợi ý thân thiện ----
CAU_HOI_MAU = [
    "📊 Cho em xem kết quả học tập của HS1023",
    "🧩 Em giỏi Toán, Lý thì nên học ngành gì?",
    "🏫 Ngành Công nghệ thông tin có những trường nào?",
]

if not st.session_state.messages:
    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        st.markdown(
            "Chào em! 👋 Mình là **PathEdu** – Trợ lý học tập thông minh. "
            "Em có thể hỏi về điểm số, ngành học hoặc chọn trường đại học bên dưới nhé:"
        )
    khu_cau_mau = st.container()                     
else:
    khu_cau_mau = st.expander("💡 Gợi ý câu hỏi nhanh")    

with khu_cau_mau:
    cols = st.columns(len(CAU_HOI_MAU))
    for i, cau in enumerate(CAU_HOI_MAU):
        with cols[i]:
            if st.button(cau, use_container_width=True):
                cau_hoi_chon = cau.split(" ", 1)[1]  

# ---- 5. Xử lý ô nhập tin nhắn ----
prompt = st.chat_input("✍️ Em muốn hỏi gì về ngành học, trường đại học?") or cau_hoi_chon

# ---- 6. Tiến trình tương tác AI ----
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATAR["assistant"]):
        try:
            with st.spinner("🧭 PathEdu đang phân tích dữ liệu..."):
                result = st.session_state.agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                )
            answer = get_final_result(result)
            st.session_state.pop("last_error", None)
        except Exception as exc:
            answer = "Xin lỗi em, hiện tại hệ thống đang bận. Em thử lại hoặc đặt câu hỏi khác nhé!"
            st.session_state.last_error = str(exc)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

# ---- 7. Xử lý hiển thị lỗi gọn gàng (nếu có) ----
if "last_error" in st.session_state:
    with st.expander("⚠️ Chi tiết lỗi kỹ thuật"):
        st.code(st.session_state.last_error)

st.markdown("---")
st.caption("PathEdu · Dự án học tập FPT Jetking · Dữ liệu giả lập mang tính chất tham khảo học tập.")
