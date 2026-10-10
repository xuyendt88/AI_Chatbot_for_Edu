import os
import json
import requests
import streamlit as st

# Nhập các hàm nghiệp vụ từ tools.py
from tools import (
    get_student_info,
    get_academic_ranking,
    get_exam_work_history,
    predict_score_and_recommend,
    recommend_universities,
    get_major_guidance,
)

TOOL_MAPPER = {
    "get_student_info": get_student_info,
    "get_academic_ranking": get_academic_ranking,
    "get_exam_work_history": get_exam_work_history,
    "predict_score_and_recommend": predict_score_and_recommend,
    "recommend_universities": recommend_universities,
    "get_major_guidance": get_major_guidance,
}

SYSTEM_PROMPT = """Bạn là Chuyên gia Tư vấn Hướng nghiệp và Tuyển sinh Đại học PathEdu, tận tụy và nhiệt tình.
Bạn BẮT BUỘC phải luôn trả lời hoàn toàn bằng Tiếng Việt chuẩn mực.
Tuyệt đối không sử dụng tiếng Nga, tiếng Trung hay bất kỳ ngôn ngữ nào khác trong câu trả lời.

Nhiệm vụ chính:
- Cung cấp thông tin chính xác về học tập, điểm chuẩn, cơ hội việc làm và định hướng phù hợp cho học sinh.
- Tra cứu thông tin điểm số học tập và điểm thi thử của học sinh khi có mã học sinh (VD: HS1023, HS1181).
- Dự đoán điểm thi THPT Quốc gia và đề xuất ngành/trường phù hợp dựa trên mô hình Machine Learning.
- Tư vấn chọn ngành, chọn trường đại học/cao đẳng phù hợp với năng lực, khối thi và học phí mong muốn.
- Định hướng nghề nghiệp, giải đáp tố chất và cơ hội việc làm của từng ngành học.

Quy tắc công cụ (Tools):
- Tự động gọi các công cụ (tools) phù hợp khi người dùng yêu cầu tra cứu điểm, dự đoán kết quả hoặc tìm trường.
- Tuyệt đối không tự suy đoán, bịa đặt điểm số khi chưa gọi công cụ.
- Nếu người dùng cần tra cứu điểm hoặc dự đoán mà chưa cung cấp mã học sinh, hãy lịch sự đề nghị học sinh cung cấp mã (VD: HS1023)."""

GEMINI_TOOLS_DECLARATION = [
    {
        "function_declarations": [
            {
                "name": "get_student_info",
                "description": "Tra cứu thông tin cá nhân và bảng điểm học tập các môn học theo mã học sinh.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "ma_hoc_sinh": {"type": "STRING", "description": "Mã học sinh cần tra cứu (ví dụ: 'HS1023')."}
                    },
                    "required": ["ma_hoc_sinh"]
                }
            },
            {
                "name": "get_academic_ranking",
                "description": "Tra cứu xếp hạng học lực và điểm trung bình của học sinh.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "ma_hoc_sinh": {"type": "STRING", "description": "Mã học sinh cần tra cứu."}
                    },
                    "required": ["ma_hoc_sinh"]
                }
            },
            {
                "name": "get_exam_work_history",
                "description": "Xem lịch sử thi thử và điểm các môn của học sinh.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "ma_hoc_sinh": {"type": "STRING", "description": "Mã học sinh."}
                    },
                    "required": ["ma_hoc_sinh"]
                }
            },
            {
                "name": "predict_score_and_recommend",
                "description": "Dự đoán điểm thi tốt nghiệp dựa trên Machine Learning và đề xuất tổ hợp môn.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "ma_hoc_sinh": {"type": "STRING", "description": "Mã học sinh."},
                        "khoi_thi": {"type": "STRING", "description": "Khối thi (ví dụ: 'A00', 'D01')."}
                    },
                    "required": ["ma_hoc_sinh", "khoi_thi"]
                }
            },
            {
                "name": "recommend_universities",
                "description": "Gợi ý danh sách trường đại học phù hợp với điểm xét tuyển, khối thi và ngành.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "score": {"type": "NUMBER", "description": "Tổng điểm xét tuyển."},
                        "khoi_thi": {"type": "STRING", "description": "Khối thi xét tuyển."},
                        "major_name": {"type": "STRING", "description": "Tên ngành học muốn tìm."}
                    },
                    "required": ["score", "khoi_thi"]
                }
            },
            {
                "name": "get_major_guidance",
                "description": "Tư vấn định hướng học tập và gợi ý ngành học dựa trên sở thích, kỹ năng.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "interest_keywords": {"type": "STRING", "description": "Từ khóa mô tả sở thích hoặc kỹ năng."}
                    },
                    "required": ["interest_keywords"]
                }
            }
        ]
    }
]

def get_api_key() -> str:
    try:
        for k in ["GEMINI_API_KEY", "APP_API_KEY", "GOOGLE_API_KEY"]:
            if k in st.secrets and st.secrets[k]:
                return str(st.secrets[k]).strip()
    except Exception:
        pass
    for k in ["GEMINI_API_KEY", "APP_API_KEY", "GOOGLE_API_KEY"]:
        val = os.getenv(k)
        if val:
            return str(val).strip()
    return ""

def run_agent(messages: list) -> str:
    api_key = get_api_key()
    if not api_key:
        return "⚠️ Chưa cấu hình GEMINI_API_KEY trong file .streamlit/secrets.toml"

    # Trích xuất câu hỏi mới nhất từ người dùng
    user_prompt = ""
    if isinstance(messages, str):
        user_prompt = messages
    elif isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break
            elif hasattr(msg, "content"):
                user_prompt = getattr(msg, "content", "")
                break

    if not user_prompt:
        return "Em hãy đặt câu hỏi hoặc gửi mã học sinh để mình hỗ trợ nhé!"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    conversation_contents = [
        {"role": "user", "parts": [{"text": user_prompt}]}
    ]

    payload = {
        "contents": conversation_contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "tools": GEMINI_TOOLS_DECLARATION
    }

    try:
        # Vòng lặp Agent: Cho phép AI thực hiện tối đa 4 bước gọi công cụ liên tiếp
        max_turns = 4
        for _ in range(max_turns):
            payload["contents"] = conversation_contents
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            data = res.json()

            if "error" in data:
                return f"Lỗi từ Gemini API: {data['error'].get('message', str(data))}"

            candidates = data.get("candidates", [])
            if not candidates:
                return "Mô hình không trả về kết quả khả dụng."

            candidate = candidates[0]
            content_obj = candidate.get("content", {})
            parts = content_obj.get("parts", [])

            # Thu thập toàn bộ các công cụ mà mô hình yêu cầu thực thi trong lượt này
            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]

            if function_calls:
                # 1. Lưu lại phản hồi của mô hình (chứa lời gọi hàm) vào lịch sử
                conversation_contents.append(content_obj)

                # 2. Thực thi từng công cụ và tạo danh sách functionResponse tương ứng
                function_responses = []
                for fc in function_calls:
                    fn_name = fc.get("name")
                    fn_args = fc.get("args", {})
                    fn_obj = TOOL_MAPPER.get(fn_name)

                    if fn_obj:
                        try:
                            output = fn_obj(**fn_args)
                        except Exception as err:
                            output = {"error": f"Lỗi khi thực thi hàm {fn_name}: {str(err)}"}
                    else:
                        output = {"error": f"Không tìm thấy hàm {fn_name}"}

                    if not isinstance(output, dict):
                        output = {"result": str(output)}

                    function_responses.append({
                        "functionResponse": {
                            "name": fn_name,
                            "response": output
                        }
                    })

                # 3. Gửi toàn bộ kết quả công cụ lại cho mô hình với role "user"
                conversation_contents.append({
                    "role": "user",
                    "parts": function_responses
                })
                # Tiếp tục vòng lặp để AI phân tích kết quả vừa nhận
                continue

            # Nếu mô hình không yêu cầu gọi công cụ nữa, trích xuất câu trả lời hoàn chỉnh
            text_parts = [p.get("text", "") for p in parts if "text" in p and p.get("text")]
            if text_parts:
                return "\n".join(text_parts).strip()

            # Trường hợp kết thúc vì bộ lọc an toàn hoặc giới hạn độ dài
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason != "STOP":
                return f"Mô hình dừng lại với trạng thái: {finish_reason}"

        return "Đã hoàn thành các bước tra cứu nhưng chưa tổng hợp được văn bản phản hồi."

    except Exception as e:
        return f"Lỗi hệ thống khi gọi Gemini: {str(e)}"

# ---------------------------------------------------------------------------
# CÁC CLASS & HÀM ĐỂ TƯƠNG THÍCH VỚI APP.PY
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
        return "Rất tiếc, mô hình chưa đưa ra được phản hồi."
    return str(result)

get_final_text = get_final_result
