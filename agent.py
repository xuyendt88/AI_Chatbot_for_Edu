# Tùng 
import json
import requests
import streamlit as st
import tools

TOOL_MAPPING = {
    "get_major_info": tools.get_major_info,
    "suggest_majors_by_interest": tools.suggest_majors_by_interest,
    "calculate_admission_chances": tools.calculate_admission_chances,
    "get_tuition_fee": tools.get_tuition_fee,
}

SYSTEM_PROMPT = """Bạn là Chuyên gia Tư vấn Hướng nghiệp và Tuyển sinh Đại học chuyên nghiệp, thấu cảm và nhiệt huyết.

MỤC TIÊU:
- Đồng hành, truyền cảm hứng và định hướng tương lai cho học sinh cấp 3 và phụ huynh.
- Cung cấp thông tin tuyển sinh, điểm chuẩn, ngành học và học phí chính xác.

QUY TẮC BẮT BUỘC:
1. KHÔNG BỊA ĐẶT THÔNG TIN: Tuyệt đối không tự suy đoán hoặc đưa ra thông tin hư cấu về điểm chuẩn, học phí, mức lương hay danh sách ngành học.
2. BẮT BUỘC GỌI CÔNG CỤ (TOOLS): Khi người dùng hỏi về bất kỳ thông tin nào liên quan đến ngành học, học phí, đánh giá điểm số hay tư vấn ngành theo sở thích, bạn BẮT BUỘC phải sử dụng công cụ phù hợp được cung cấp.
3. TÔN TRỌNG KẾT QUẢ TỪ TOOL: Chỉ sử dụng dữ liệu trả về từ công cụ để trả lời người dùng. Nếu công cụ báo không tìm thấy, hãy thông báo lịch sự và đề xuất người dùng thử từ khóa khác.
4. PHONG CÁCH TRẢ LỜI: Thân thiện, gần gũi, sử dụng ngôn từ tích cực, truyền động lực, cấu trúc rõ ràng.
"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_major_info",
            "description": "Tra cứu thông tin chi tiết về một ngành học cụ thể bao gồm khối thi, điểm chuẩn, mức lương tham khảo, tố chất phù hợp và mô tả ngành.",
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
            "description": "Gợi ý danh sách ngành học phù hợp dựa trên sở thích, từ khóa tính cách hoặc thế mạnh của học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interest_keyword": {
                        "type": "string",
                        "description": "Từ khóa thể hiện sở thích, tính cách hoặc thế mạnh (ví dụ: 'sáng tạo', 'logic', 'ngoại ngữ', 'giao tiếp')."
                    }
                },
                "required": ["interest_keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_admission_chances",
            "description": "Tính toán và đánh giá mức độ/cơ hội trúng tuyển dựa vào tổng điểm xét tuyển và tên ngành học.",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "number",
                        "description": "Tổng điểm xét tuyển của học sinh (ví dụ: 25.5)."
                    },
                    "major": {
                        "type": "string",
                        "description": "Tên ngành học muốn đánh giá cơ hội trúng tuyển."
                    }
                },
                "required": ["score", "major"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tuition_fee",
            "description": "Tra cứu mức học phí tham khảo của một trường đại học cụ thể.",
            "parameters": {
                "type": "object",
                "properties": {
                    "university_name": {
                        "type": "string",
                        "description": "Tên trường đại học cần tra cứu học phí (ví dụ: 'Đại học Bách Khoa', 'Đại học FPT')."
                    }
                },
                "required": ["university_name"]
            }
        }
    }
]

def run_agent(messages: list) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return "⚠️ Lỗi: Chưa cấu hình OPENROUTER_API_KEY trong file .streamlit/secrets.toml"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": full_messages,
        "tools": TOOLS_SCHEMA,
        "tool_choice": "auto"
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
        return f"❌ Đã xảy ra lỗi khi kết nối với AI Agent: {str(e)}"
