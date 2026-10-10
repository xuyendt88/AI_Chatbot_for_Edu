"""
LangChain & LangGraph Agent cho PathEdu Assistant.
Sử dụng mô hình Gemini 3.6 và kiến trúc ReAct Agent chuẩn công nghiệp.
"""

import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool

# Nhập các hàm nghiệp vụ từ tools.py
from tools import (
    get_student_info as raw_get_student_info,
    get_academic_ranking as raw_get_academic_ranking,
    get_exam_work_history as raw_get_exam_work_history,
    predict_score_and_recommend as raw_predict_score_and_recommend,
    recommend_universities as raw_recommend_universities,
    get_major_guidance as raw_get_major_guidance,
)

# ---------------------------------------------------------------------------
# 1. BỌC CÁC CÔNG CỤ THEO CHUẨN LANGCHAIN TOOLS
# ---------------------------------------------------------------------------
@tool
def get_student_info(ma_hoc_sinh: str) -> str:
    """Tra cứu thông tin cá nhân và bảng điểm học tập các môn học theo mã học sinh (ví dụ: 'HS1023')."""
    return str(raw_get_student_info(ma_hoc_sinh=ma_hoc_sinh))

@tool
def get_academic_ranking(ma_hoc_sinh: str) -> str:
    """Tra cứu xếp hạng học lực và điểm trung bình của học sinh theo mã học sinh."""
    return str(raw_get_academic_ranking(ma_hoc_sinh=ma_hoc_sinh))

@tool
def get_exam_work_history(ma_hoc_sinh: str) -> str:
    """Xem lịch sử làm bài thi thử và điểm các môn của học sinh."""
    return str(raw_get_exam_work_history(ma_hoc_sinh=ma_hoc_sinh))

@tool
def predict_score_and_recommend(ma_hoc_sinh: str, khoi_thi: str) -> str:
    """Dự đoán điểm thi tốt nghiệp dựa trên Machine Learning và đề xuất tổ hợp/khối thi (ví dụ: 'A00', 'D01')."""
    return str(raw_predict_score_and_recommend(ma_hoc_sinh=ma_hoc_sinh, khoi_thi=khoi_thi))

@tool
def recommend_universities(score: float, khoi_thi: str, major_name: str = "") -> str:
    """Gợi ý danh sách các trường đại học phù hợp với điểm xét tuyển, khối thi và ngành học."""
    return str(raw_recommend_universities(score=score, khoi_thi=khoi_thi, major_name=major_name))

@tool
def get_major_guidance(interest_keywords: str) -> str:
    """Tư vấn định hướng nghề nghiệp, giải đáp tố chất và cơ hội việc làm dựa trên sở thích, kỹ năng."""
    return str(raw_get_major_guidance(interest_keywords=interest_keywords))

ALL_TOOLS = [
    get_student_info,
    get_academic_ranking,
    get_exam_work_history,
    predict_score_and_recommend,
    recommend_universities,
    get_major_guidance,
]

# ---------------------------------------------------------------------------
# 2. SYSTEM PROMPT (CHUẨN HÓA TIẾNG VIỆT & CHỐNG ẢO GIÁC)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Bạn là PathEdu Assistant - Chuyên gia tư vấn hướng nghiệp, tuyển sinh đại học và tra cứu học tập.

Quy tắc ứng xử và bảo đảm chất lượng:
1. Luôn luôn trả lời 100% bằng Tiếng Việt chuẩn mực, ân cần, mang tính giáo dục và động viên học sinh.
2. Tuyệt đối không sử dụng tiếng Nga, tiếng Trung hay bất kỳ ngôn ngữ nào khác.
3. BẮT BUỘC sử dụng công cụ (tools) khi người dùng hỏi về điểm số, xếp hạng, kết quả thi hoặc thông tin trường học.
4. Tuyệt đối KHÔNG tự suy đoán hoặc bịa đặt (hallucinate) dữ liệu học sinh khi công cụ chưa trả về kết quả.
5. Nếu người dùng cần tra cứu điểm hoặc dự đoán kết quả nhưng chưa cung cấp mã học sinh, hãy lịch sự hỏi mã học sinh (ví dụ: HS1023).
6. Tận dụng ngữ cảnh các lượt chat trước để trả lời liền mạch các câu hỏi tiếp theo.
7. Đưa ra lời khuyên khách quan, phân tích điểm mạnh - điểm yếu và khuyến khích học sinh cải thiện môn học còn yếu.
"""

# ---------------------------------------------------------------------------
# 3. HÀM LẤY API KEY LINH HOẠT VÀ AN TOÀN
# ---------------------------------------------------------------------------
def _resolve_api_key():
    # Ưu tiên lấy từ Streamlit Secrets nếu đang chạy web
    try:
        import streamlit as st
        for key in ["GEMINI_API_KEY", "APP_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"]:
            if key in st.secrets and st.secrets[key]:
                return st.secrets[key]
    except Exception:
        pass

    # Nếu không có, tìm trong biến môi trường
    for key in ["GEMINI_API_KEY", "APP_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(key)
        if val:
            return val
    return None

# ---------------------------------------------------------------------------
# 4. HÀM KHỞI TẠO AGENT (THEO MẪU AGENT 1)
# ---------------------------------------------------------------------------
def build_agent():
    api_key = _resolve_api_key()
    if not api_key:
        raise RuntimeError(
            "Chưa cấu hình API Key. Vui lòng thêm APP_API_KEY hoặc GEMINI_API_KEY "
            "vào file .streamlit/secrets.toml hoặc biến môi trường."
        )

    # Khởi tạo Chat Model
    # Cách A: Dùng trực tiếp Google Gemini API nếu key bắt đầu bằng AIzaSy
    if api_key.startswith("AIzaSy"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0,  # Giữ ở mức 0 để đảm bảo tính chính xác tuyệt đối của điểm số
        )
    # Cách B: Dùng qua cổng OpenRouter
    else:
        from langchain_openrouter import ChatOpenRouter
        model = ChatOpenRouter(
            model="google/gemini-3.6-flash",
            api_key=api_key,
            temperature=0,
        )

    # Tương thích linh hoạt giữa langchain.agents và langgraph.prebuilt
    try:
        from langchain.agents import create_agent
        return create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(
            model=model,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

# ---------------------------------------------------------------------------
# 5. HÀM TRÍCH XUẤT NỘI DUNG PHẢN HỒI (KHỚP VỚI APP.PY)
# ---------------------------------------------------------------------------
def get_final_result(result: dict) -> str:
    """Rút trích câu trả lời cuối cùng của trợ lý từ kết quả trả về của Agent."""
    if not isinstance(result, dict):
        return str(result)

    messages = result.get("messages", [])
    if not messages:
        return "Rất tiếc, hệ thống chưa nhận được phản hồi từ AI."

    last_msg = messages[-1]
    content = getattr(last_msg, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if text_blocks:
            return "\n".join(text_blocks)

    return str(content)

# Alias để tương thích cả 2 tên gọi (get_final_result trong app.py và get_final_text trong mẫu 1)
get_final_text = get_final_result
