"""
PathEdu AI Agent - Sử dụng Google Gemini API chính thức (gemini-3.6-flash)
Tự động kích hoạt Function Calling để tra cứu học tập, điểm số và tư vấn ngành.
"""

import os
import streamlit as st
import google.generativeai as genai

# Nhập các hàm nghiệp vụ từ tools.py
from tools import (
    get_student_info,
    get_academic_ranking,
    get_exam_work_history,
    predict_score_and_recommend,
    recommend_universities,
    get_major_guidance,
)

# ---------------------------------------------------------------------------
# 1. SYSTEM PROMPT (CHUẨN HÓA TIẾNG VIỆT & CHỐNG BỊA ĐẶT DỮ LIỆU)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Bạn là Chuyên gia Tư vấn Hướng nghiệp và Tuyển sinh Đại học PathEdu, tận tụy và nhiệt tình.
Bạn BẮT BUỘC phải luôn trả lời hoàn toàn bằng Tiếng Việt chuẩn mực.
Tuyệt đối không sử dụng tiếng Nga, tiếng Trung hay bất kỳ ngôn ngữ nào khác trong câu trả lời.

Nhiệm vụ chính:
- Cung cấp thông tin chính xác về học tập, điểm chuẩn, cơ hội việc làm và định hướng phù hợp cho học sinh.
- Tra cứu thông tin điểm số học tập và điểm thi thử của học sinh khi có mã học sinh (VD: HS1023, HS1181).
- Dự đoán điểm thi THPT Quốc gia và đề xuất ngành/trường phù hợp dựa trên mô hình Machine Learning.
- Tư vấn chọn ngành, chọn trường đại học/cao đẳng phù hợp với năng lực, khối thi và học phí mong muốn.
- Định hướng nghề nghiệp, giải đáp tố chất và cơ hội việc làm của từng ngành học.

Chính sách ứng xử & Quy tắc công cụ (Tools):
- Luôn sử dụng ngôn ngữ tiếng Việt lịch sự, ân cần, động viên và mang tính giáo dục.
- BẮT BUỘC tự động gọi các công cụ (tools) phù hợp khi người dùng yêu cầu tra cứu điểm, dự đoán kết quả hoặc tìm trường đại học.
- Tuyệt đối KHÔNG tự bịa đặt điểm số nếu công cụ chưa có dữ liệu.
- Nếu người dùng cung cấp mã học sinh, hãy ưu tiên dùng công cụ `get_student_info` hoặc `predict_score_and_recommend` để tra cứu chính xác.
- Nếu người dùng yêu cầu xem điểm hoặc dự đoán mà chưa cung cấp mã học sinh, hãy lịch sự đề nghị học sinh cung cấp mã (VD: HS1023).
- Đưa ra lời khuyên chân thành, khuyến khích học sinh nỗ lực cải thiện điểm số ở các môn còn yếu."""

# ---------------------------------------------------------------------------
# 2. ĐỊNH NGHĨA DANH SÁCH CÔNG CỤ CHO GEMINI
# ---------------------------------------------------------------------------
ALL_TOOLS = [
    get_student_info,
    get_academic_ranking,
    get_exam_work_history,
    predict_score_and_recommend,
    recommend_universities,
    get_major_guidance,
]

# ---------------------------------------------------------------------------
# 3. HÀM LẤY API KEY AN TOÀN TỪ SECRETS HOẶC MÔI TRƯỜNG
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    # 1. Thử lấy từ Streamlit secrets (.streamlit/secrets.toml)
    try:
        for key_name in ["GEMINI_API_KEY", "APP_API_KEY", "GOOGLE_API_KEY"]:
            if key_name in st.secrets and st.secrets[key_name]:
                return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    # 2. Thử lấy từ biến môi trường hệ thống
    for env_name in ["GEMINI_API_KEY", "APP_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(env_name)
        if val:
            return str(val).strip()

    return ""

# ---------------------------------------------------------------------------
# 4. HÀM THỰC THI AGENT VỚI GEMINI 3.6 FLASH
# ---------------------------------------------------------------------------
def run_agent(messages: list) -> str:
    """
    Thực thi Agent trao đổi trực tiếp với Google Gemini API (model gemini-3.6-flash).
    Tự động xử lý Tool Calling nhiều bước (Multi-step Reasoning).
    """
    api_key = get_api_key()
    if not api_key:
        return "⚠️ Chưa cấu hình khóa API trong `.streamlit/secrets.toml`. Vui lòng thêm GEMINI_API_KEY hoặc APP_API_KEY."

    try:
        # Cấu hình SDK Google Gemini
        genai.configure(api_key=api_key)

        # Khởi tạo model với gemini-3.6-flash (tự động fallback nếu môi trường chưa mở bản 3.6)
        model_name = "gemini-3.6-flash"
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
                tools=ALL_TOOLS
            )
        except Exception:
            # Dự phòng sang gemini-1.5-flash nếu cần
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_PROMPT,
                tools=ALL_TOOLS
            )

        # Chuẩn hóa tin nhắn và trích xuất câu hỏi mới nhất của người dùng
        user_prompt = ""
        past_turns = []

        if isinstance(messages, str):
            user_prompt = messages
        elif isinstance(messages, list):
            # Tách các tin nhắn cũ và tin nhắn cuối
            for msg in messages:
                if isinstance(msg, dict):
                    role = "user" if msg.get("role") == "user" else "model"
                    content = msg.get("content", "")
                else:
                    # Hỗ trợ dạng đối tượng tin nhắn của LangChain nếu có
                    msg_role = getattr(msg, "type", "user")
                    role = "user" if msg_role in ["human", "user"] else "model"
                    content = getattr(msg, "content", "")

                if content:
                    past_turns.append({"role": role, "parts": [content]})

            # Lấy câu hỏi cuối cùng
            if past_turns and past_turns[-1]["role"] == "user":
                last_turn = past_turns.pop()
                user_prompt = last_turn["parts"][0]
            elif past_turns:
                user_prompt = past_turns[-1]["parts"][0]

        if not user_prompt:
            return "Em có thể đặt câu hỏi hoặc gửi mã học sinh để mình hỗ trợ nhé!"

        # Chuẩn hóa lịch sử hội thoại cho Gemini (phải bắt đầu bằng 'user' và xen kẽ 'model')
        sanitized_history = []
        for turn in past_turns:
            if not sanitized_history and turn["role"] != "user":
                continue
            if sanitized_history and sanitized_history[-1]["role"] == turn["role"]:
                sanitized_history[-1]["parts"].extend(turn["parts"])
            else:
                sanitized_history.append(turn)

        # Đảm bảo lịch sử kết thúc bằng 'model' trước khi gửi prompt 'user' mới
        while sanitized_history and sanitized_history[-1]["role"] != "model":
            sanitized_history.pop()

        # Tạo phiên chat có bật tự động gọi hàm (automatic function calling)
        chat = model.start_chat(
            history=sanitized_history,
            enable_automatic_function_calling=True
        )

        response = chat.send_message(user_prompt)

        if response and response.text:
            return response.text
        return "Rất tiếc, mình chưa tìm thấy thông tin phù hợp. Em thử hỏi lại nhé!"

    except Exception as e:
        return f"Lỗi xử lý Gemini API: {str(e)}"

# ---------------------------------------------------------------------------
# 5. LỚP CUSTOM AGENT & HÀM TIỆN ÍCH TƯƠNG THÍCH HOÀN TOÀN VỚI APP.PY
# ---------------------------------------------------------------------------
class CustomAgent:
    def invoke(self, input_data, config=None):
        if isinstance(input_data, dict):
            messages = input_data.get("messages", [])
        elif isinstance(input_data, list):
            messages = input_data
        else:
            messages = [{"role": "user", "content": str(input_data)}]

        return run_agent(messages)

def build_agent():
    return CustomAgent()

def get_final_result(result):
    if not result or result == "None":
        return "Rất tiếc, mô hình chưa đưa ra được phản hồi. Bạn hãy thử lại hoặc đổi câu hỏi nhé!"
    return str(result)

# Alias tương thích
get_final_text = get_final_result
