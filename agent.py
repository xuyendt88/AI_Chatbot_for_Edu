import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

from tools import (
    get_student_info,
    get_academic_ranking,
    get_exam_work_history,
    predict_score_and_recommend,
    recommend_universities,
    get_major_guidance
)

TOOL_MAPPER = {
    "get_student_info": get_student_info,
    "get_academic_ranking": get_academic_ranking,
    "get_exam_work_history": get_exam_work_history,
    "predict_score_and_recommend": predict_score_and_recommend,
    "recommend_universities": recommend_universities,
    "get_major_guidance": get_major_guidance
}

SYSTEM_PROMPT = """Bạn là Chuyên gia Tư vấn Hướng nghiệp và Tuyển sinh Đại học chăm chỉ, tận tụy và nhiệt tình.
Cung cấp thông tin chính xác về học tập, điểm chuẩn, cơ hội việc làm và định hướng phù hợp cho học sinh."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_student_info",
            "description": "Tra cứu thông tin cá nhân và bảng điểm học tập các môn học theo mã học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {
                        "type": "string",
                        "description": "Mã học sinh cần tra cứu (Ví dụ: 'HS001')."
                    }
                },
                "required": ["ma_hoc_sinh"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_academic_ranking",
            "description": "Tra cứu xếp hạng học lực và tổng điểm trung bình của học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {
                        "type": "string",
                        "description": "Mã học sinh cần tra cứu xếp hạng."
                    }
                },
                "required": ["ma_hoc_sinh"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_exam_work_history",
            "description": "Xem lịch sử thi thử và điểm các môn của học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {
                        "type": "string",
                        "description": "Mã học sinh cần tra cứu lịch sử thi thử."
                    }
                },
                "required": ["ma_hoc_sinh"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_score_and_recommend",
            "description": "Dự đoán điểm thi dựa trên Machine Learning và đề xuất khối THPT/tổ hợp phù hợp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {
                        "type": "string",
                        "description": "Mã học sinh."
                    },
                    "khoi_thi": {
                        "type": "string",
                        "description": "Khối thi cần tính điểm (Ví dụ: 'A00', 'A01', 'D01')."
                    }
                },
                "required": ["ma_hoc_sinh", "khoi_thi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_universities",
            "description": "Gợi ý danh sách trường đại học đào tạo ngành học mong muốn phù hợp với điểm xét tuyển.",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Điểm trung bình hoặc tổng điểm dự đoán."
                    },
                    "khoi_thi": {
                        "type": "string",
                        "description": "Khối thi xét tuyển."
                    },
                    "major_name": {
                        "type": "string",
                        "description": "Tên ngành học muốn tìm trường."
                    }
                },
                "required": ["score", "khoi_thi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_major_guidance",
            "description": "Tư vấn định hướng học tập và gợi ý ngành học dựa trên sở thích, kỹ năng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interest_keywords": {
                        "type": "string",
                        "description": "Từ khóa mô tả sở thích hoặc ngành học cần tư vấn."
                    }
                },
                "required": ["interest_keywords"]
            }
        }
    }
]


class CustomAgent:
    def invoke(self, input_data, config=None):
        # Lấy câu hỏi từ danh sách messages
        messages = input_data.get("messages", [])
        prompt = messages[-1]["content"] if messages else ""
        # Gọi hàm run_agent đã có sẵn trong agent.py
        return run_agent(prompt)

def build_agent():
    return CustomAgent()


def get_final_result(res_data) -> str:
    """Hàm rút trích văn bản kết quả cuối cùng từ API response."""
    if isinstance(res_data, dict) and "choices" in res_data:
        return res_data["choices"][0]["message"].get("content", "")
    return str(res_data)


def run_agent(messages: list) -> str:
    """
    Thực thi Agent trao đổi với OpenRouter API và xử lý Tool Call tự động.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return " Chưa cấu hình OPENROUTER_API_KEY trong môi trường hoặc Streamlit secrets."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": api_messages,
        "tools": TOOLS_SCHEMA
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        res_data = response.json()

        if "choices" not in res_data or not res_data["choices"]:
            error_msg = res_data.get("error", {}).get("message", "Lỗi không xác định từ OpenRouter API")
            return f" Lỗi từ API: {error_msg}"

        msg = res_data["choices"][0]["message"]
        
        if msg.get("tool_calls"):
            tool_calls = msg["tool_calls"]
            tool_call = tool_calls[0]
            func_name = tool_call["function"]["name"]
            
            try:
                func_args = json.loads(tool_call["function"]["arguments"])
            except Exception:
                func_args = {}

            if func_name in TOOL_MAPPER:
                tool_result = TOOL_MAPPER[func_name](**func_args)
            else:
                tool_result = json.dumps({"error": f"Không tìm thấy function {func_name}"})

            api_messages.append(msg)
            api_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": str(tool_result) if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)
            })

            payload["messages"] = api_messages

            second_response = requests.post(url, headers=headers, json=payload, timeout=30)
            second_res_data = second_response.json()

            if "choices" in second_res_data and second_res_data["choices"]:
                return second_res_data["choices"][0]["message"].get("content", "")
            else:
                return " Không thể nhận phản hồi từ mô hình sau khi thực thi công cụ."

        return msg.get("content", "")

    except Exception as e:
        return f" Xảy ra lỗi hệ thống khi kết nối Agent: {str(e)}"
