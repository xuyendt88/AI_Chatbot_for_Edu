# Tùng
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
            "parameters":
