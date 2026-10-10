import os
import streamlit as st
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# Nhập các hàm gốc từ tools.py
import tools as t
# Bọc các hàm thành công cụ chuẩn LangChain với mô tả chi tiết
@tool
def get_student_info(ma_hoc_sinh: str) -> str:
    """Tra cứu thông tin cá nhân và bảng điểm học tập các môn học theo mã học sinh (ví dụ: 'HS1023')."""
    return str(t.get_student_info(ma_hoc_sinh))

@tool
def get_academic_ranking(ma_hoc_sinh: str) -> str:
    """Tra cứu xếp hạng học lực và điểm trung bình của học sinh."""
    return str(t.get_academic_ranking(ma_hoc_sinh))

@tool
def get_exam_work_history(ma_hoc_sinh: str) -> str:
    """Xem lịch sử làm bài thi thử và điểm các môn của học sinh."""
    return str(t.get_exam_work_history(ma_hoc_sinh))

@tool
def predict_score_and_recommend(ma_hoc_sinh: str, khoi_thi: str) -> str:
    """Dự đoán điểm thi tốt nghiệp THPT dựa trên Machine Learning và đề xuất tổ hợp môn."""
    return str(t.predict_score_and_recommend(ma_hoc_sinh, khoi_thi))

@tool
def recommend_universities(score: float, khoi_thi: str, major_name: str = "") -> str:
    """Gợi ý danh sách các trường đại học phù hợp với điểm xét tuyển, khối thi và tên ngành học."""
    return str(t.recommend_universities(score, khoi_thi, major_name))

@tool
def get_major_guidance(interest_keywords: str) -> str:
    """Tư vấn định hướng học tập, ngành nghề và cơ hội việc làm dựa trên sở thích, kỹ năng."""
    return str(t.get_major_guidance(interest_keywords))

ALL_TOOLS = [
    get_student_info,
    get_academic_ranking,
    get_exam_work_history,
    predict_score_and_recommend,
    recommend_universities,
    get_major_guidance,
]

SYSTEM_PROMPT = """Bạn là anh chị Chuyên gia Tư vấn Tuyển sinh PathEdu.
1. Luôn trả lời hoàn toàn bằng Tiếng Việt lịch sự, ân cần và động viên.
2. Tự động gọi công cụ khi cần tra cứu điểm, dự đoán kết quả hoặc tìm trường.
3. Không tự bịa đặt điểm số khi chưa tra cứu thành công."""

def _get_api_key():
    for k in ["GEMINI_API_KEY", "APP_API_KEY", "GOOGLE_API_KEY"]:
        if k in st.secrets and st.secrets[k]:
            return str(st.secrets[k]).strip()
        if os.getenv(k):
            return str(os.getenv(k)).strip()
    return None

def build_agent():
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Chưa cấu hình API Key trong .streamlit/secrets.toml.")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0,
    )

    return create_react_agent(
        model=model,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )

def get_final_result(result: dict) -> str:
    messages = result.get("messages", [])
    if not messages:
        return "Chưa nhận được phản hồi từ trợ lý."
    
    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join([b.get("text", "") for b in content if isinstance(b, dict) and "text" in b])
    return str(content)

get_final_text = get_final_result
