import os
import json
import requests
from tools import (
    get_major_info,
    suggest_majors_by_interest,
    calculate_admission_chances,
    get_tuition_fees,
    recommend_universities,
    get_major_guidance
)

TOOL_MAPPING = {
    "get_major_info": get_major_info,
    "suggest_majors_by_interest": suggest_majors_by_interest,
    "calculate_admission_chances": calculate_admission_chances,
    "get_tuition_fees": get_tuition_fees,
    "recommend_universities": recommend_universities,
    "get_major_guidance": get_major_guidance,
}

SYSTEM_PROMPT = """Bạn là Chuyên gia Tư vấn Hướng nghiệp và Tuyển sinh Đại học chính quy, tâm huyết và nhiệt tình.
Cung cấp thông tin chính xác về các ngành học, điểm chuẩn, cơ hội việc làm và định hướng cho học sinh."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_major_info",
            "description": "Tra cứu thông tin chi tiết về một ngành học cụ thể bao gồm khối thi, điểm chuẩn, môn học tham khảo, tố chất phù hợp và mở rộng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "major_name": {
                        "type": "string",
                        "description": "Tên ngành học cần tra cứu (ví dụ: 'Công nghệ thông tin', 'Kinh doanh quốc tế')."
                    }
                },
                "required": ["major_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_majors_by_interest",
            "description": "Gợi ý danh sách ngành học phù hợp dựa trên sở thích, tính cách hoặc thế mạnh của học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interest_keywords": {
                        "type": "string",
                        "description": "Từ khóa mô tả sở thích, tính cách hoặc thế mạnh (ví dụ: 'sáng tạo', 'logic', 'ngoại ngữ', 'giao tiếp')."
                    }
                },
                "required": ["interest_keywords"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_admission_chances",
            "description": "Tính toán và đánh giá mức độ cơ hội trúng tuyển dựa vào tổng điểm xét tuyển và điểm chuẩn ngành.",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Tổng điểm xét tuyển của học sinh (ví dụ: 25.5)."
                    },
                    "major": {
                        "type": "string",
                        "description": "Ngành học muốn tra cứu cơ hội trúng tuyển."
                    }
                },
                "required": ["score", "major"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tuition_fees",
            "description": "Tra cứu mức học phí tham khảo của một trường đại học cụ thể.",
            "parameters": {
                "type": "object",
                "properties": {
                    "university_name": {
                        "type": "string",
                        "description": "Tên trường đại học cần tra cứu học phí (ví dụ: 'Đại học Bách khoa', 'Đại học FPT')."
                    }
                },
                "required": ["university_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_universities",
            "description": "Gợi ý danh sách trường đại học đào tạo ngành học mong muốn phù hợp với mức điểm xét tuyển.",
            "parameters": {
                "type": "object",
                "properties": {
                    "major_name": {
                        "type": "string",
                        "description": "Tên ngành học muốn tìm trường."
                    },
                    "score": {
                        "type": "number",
                        "description": "Tổng điểm xét tuyển của học sinh."
                    }
                },
                "required": ["major_name", "score"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_major_guidance",
            "description": "Tư vấn định hướng học tập và kỹ năng cần chuẩn bị cho ngành học.",
            "parameters": {
                "type": "object",
                "properties": {
                    "major_name": {
                        "type": "string",
                        "description": "Tên ngành học cần tư vấn định hướng."
                    }
                },
                "required": ["major_name"]
            }
        }
    }
]


def run_agent(messages: list) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return " Chưa cấu hình OPENROUTER_API_KEY trong biến môi trường hoặc Streamlit secrets."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": TOOLS_SCHEMA
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        res_data = response.json()
        response_msg = res_data["choices"][0]["message"]

        if response_msg.get("tool_calls"):
            messages.append(response_msg)

            for tool_call in response_msg["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])

                if func_name in TOOL_MAPPING:
                    tool_result = TOOL_MAPPING[func_name](**func_args)
                else:
                    tool_result = {"error": f"Công cụ '{func_name}' không khả thi."}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

            payload["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            second_response = requests.post(url, headers=headers, json=payload)
            second_response.raise_for_status()
            final_data = second_response.json()
            return final_data["choices"][0]["message"]["content"]

        return response_msg.get("content", "")

    except Exception as e:
        return f" Đã xảy ra lỗi khi kết nối với AI Agent: {str(e)}"


def build_agent():
    return True


def get_final_result(data: dict) -> str:
    messages = data.get("messages", [])
    return run_agent(messages)
