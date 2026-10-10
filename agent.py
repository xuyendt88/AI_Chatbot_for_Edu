import os
import streamlit as st
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

# Nhập các hàm gốc từ tools.py
import tools as t

# Bọc các hàm thành công cụ chuẩn LangChain với mô tả chi tiết, rõ ràng để LLM nhận diện ngay
@tool
def get_student_info(ma_hoc_sinh: str) -> str:
    """Tra cứu thông tin cá nhân, bảng điểm 10 môn học và chứng chỉ của học sinh theo mã (VD: HS1023)."""
    try:
        return str(t.get_student_info(ma_hoc_sinh))
    except Exception as e:
        return f'{{"error": "Lỗi thực thi tool: {str(e)}"}}'

@tool
def get_academic_ranking(ma_hoc_sinh: str) -> str:
    """Tra cứu xếp hạng học lực, điểm trung bình chung và so sánh với lớp/khối của học sinh theo mã."""
    try:
        return str(t.get_academic_ranking(ma_hoc_sinh))
    except Exception as e:
        return f'{{"error": "Lỗi thực thi tool: {str(e)}"}}'

@tool
def get_exam_work_history(ma_hoc_sinh: str) -> str:
    """Xem lịch sử làm bài thi thử và điểm các môn thi thử của học sinh theo mã."""
    try:
        return str(t.get_exam_work_history(ma_hoc_sinh))
    except Exception as e:
        return f'{{"error": "Lỗi thực thi tool: {str(e)}"}}'

@tool
def predict_score_and_recommend(ma_hoc_sinh: str, khoi_thi: str) -> str:
    """Dự đoán điểm thi THPT bằng Machine Learning và đề xuất tổng điểm 3 môn theo khối thi (A00, D01...)."""
    try:
        return str(t.predict_score_and_recommend(ma_hoc_sinh, khoi_thi))
    except Exception as e:
        return f'{{"error": "Lỗi thực thi tool: {str(e)}"}}'

@tool
def recommend_universities(tong_diem_3_mon_du_doan: float, khoi_thi: str) -> str:
    """Gợi ý danh sách trường đại học, học phí và điểm chuẩn phù hợp với tổng điểm dự đoán và khối thi."""
    try:
        return str(t.recommend_universities(khoi_thi, tong_diem_3_mon_du_doan))
    except Exception as e:
        return f'{{"error": "Lỗi thực thi tool: {str(e)}"}}'

@tool
def get_major_guidance(ten_nganh_hoac_tukhoa: str) -> str:
    """Tư vấn định hướng ngành nghề, mô tả ngành, tổ hợp môn xét tuyển và cơ hội việc làm theo từ khóa."""
    try:
        return str(t.get_major_guidance(ten_nganh_hoac_tukhoa))
    except Exception as e:
        return f'{{"error": "Lỗi thực thi tool: {str(e)}"}}'

ALL_TOOLS = [
    get_student_info,
    get_academic_ranking,
    get_exam_work_history,
    predict_score_and_recommend,
    recommend_universities,
    get_major_guidance,
]

SYSTEM_PROMPT = """Bạn là anh chị Chuyên gia Tư vấn Tuyển sinh PathEdu.
1. Luôn trả lời hoàn toàn bằng Tiếng Việt ân cần, dịu dàng, lịch sự, hài hước và trung thực.
2. Tự động gọi công cụ khi cần tra cứu điểm, dự đoán kết quả hoặc tìm trường.
3. Nếu người dùng hỏi chung chung không có mã học sinh, hãy lịch sự yêu cầu cung cấp mã học sinh (VD: HS1001).
4. Không tự bịa đặt điểm số hoặc thông tin khi chưa tra cứu thành công qua công cụ."""

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

    # Sử dụng gemini-1.5-flash để tối ưu tốc độ phản hồi cực nhanh và ổn định với tool calling
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
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
