#Tùng
import json
import os
import requests
from tools import (
    get_academic_ranking,
    get_major_guidance,
    get_mock_exam_history,
    get_student_info,
    predict_score_and_recommend,
    recommend_universities,
)

TOOL_MAPPING = {
    "get_student_info": get_student_info,
    "get_academic_ranking": get_academic_ranking,
    "get_mock_exam_history": get_mock_exam_history,
    "predict_score_and_recommend": predict_score_and_recommend,
    "recommend_universities": recommend_universities,
    "get_major_guidance": get_major_guidance,
}

SYSTEM_PROMPT = """Bạn là Chuyên gia Tư vấn Hướng nghiệp và Tuyển sinh Đại học chính quy, tâm huyết và nhiệt tình.
Cung cấp thông tin chính xác về các ngành học, điểm chuẩn, cơ hội việc làm và định hướng cho học sinh."""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_student_info",
            "description": "Tra cứu thông tin cá nhân và bảng điểm học tập của học sinh theo mã học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {
                        "type": "string",
                        "description": "Mã học sinh cần tra cứu (ví dụ: 'HS001').",
                    }
                },
                "required": ["ma_hoc_sinh"],
            },
        },
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
                        "description": "Mã học sinh cần tra cứu xếp hạng.",
                    }
                },
                "required": ["ma_hoc_sinh"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mock_exam_history",
            "description": "Xem lịch sử thi thử và điểm các môn thi của học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {
                        "type": "string",
                        "description": "Mã học sinh cần tra cứu lịch sử thi thử.",
                    }
                },
                "required": ["ma_hoc_sinh"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_score_and_recommend",
            "description": "Dự đoán điểm thi dựa trên Machine Learning và đề xuất khối thi/trường phù hợp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {
                        "type": "string",
                        "description": "Mã học sinh.",
                    },
                    "khoi_thi": {
                        "type": "string",
                        "description": "Khối thi dự định (ví dụ: 'A00', 'A01', 'D01').",
                    },
                },
                "required": ["ma_hoc_sinh", "khoi_thi"],
            },
        },
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
                        "description": "Tên ngành học muốn tìm trường.",
                    },
                    "score": {
                        "type": "number",
                        "description": "Tổng điểm xét tuyển của học sinh.",
                    },
                },
                "required": ["major_name", "score"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_major_guidance",
            "description": "Tư vấn định hướng học tập và gợi ý ngành học dựa trên sở thích, từ khóa ngành.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interest_keywords": {
                        "type": "string",
                        "description": "Từ khóa mô tả sở thích hoặc tên ngành cần tư vấn.",
                    }
                },
                "required": ["interest_keywords"],
            },
        },
    },
]


def run_agent(messages: list) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return "Chưa cấu hình OPENROUTER_API_KEY trong biến môi trường hoặc Streamlit secrets."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": TOOLS_SCHEMA,
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
                    tool_result = {
                        "error": f"Công cụ '{func_name}' không khả thi."
                    }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            payload["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + messages
            second_response = requests.post(
                url, headers=headers, json=payload
            )
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
