import pandas as pd
import json
import os
import joblib

# ==========================================
# 1. ĐƯỜNG DẪN DỮ LIỆU ĐỘNG & NẠP DỮ LIỆU CSV
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KHOI_THI_MAP = {
    "A00": ["diem_toan", "diem_vat_li", "diem_hoa_hoc"],
    "A01": ["diem_toan", "diem_vat_li", "diem_ngoai_ngu"],
    "B00": ["diem_toan", "diem_hoa_hoc", "diem_sinh_hoc"],
    "C00": ["diem_ngu_van", "diem_lich_su", "diem_dia_li"],
    "D01": ["diem_toan", "diem_ngu_van", "diem_ngoai_ngu"],
    "D07": ["diem_toan", "diem_hoa_hoc", "diem_ngoai_ngu"]
}

def load_resources():
    try:
        # Thêm 'data' vào đường dẫn trỏ tới thư mục chứa CSV
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        
        df_students = pd.read_csv(os.path.join(DATA_DIR, '01_Danh_sach_va_Thanh_tich_hoc_sinh_clean.csv'))
        df_rankings = pd.read_csv(os.path.join(DATA_DIR, '02_Ket_qua_va_Xep_hang_Hoc_tap_clean.csv'))
        df_majors = pd.read_csv(os.path.join(DATA_DIR, '03_Nganh_hoc_va_Huong_nghiep_clean.csv'))
        df_admissions = pd.read_csv(os.path.join(DATA_DIR, '04_Tuyen_sinh_va_Hoc_phi_clean.csv'))
        df_mock_exams = pd.read_csv(os.path.join(DATA_DIR, '05_Ket_qua_thi_thu_clean.csv'))
        
        df_majors['ma_nganh'] = df_majors['ma_nganh'].astype(str)
        df_admissions['ma_nganh'] = df_admissions['ma_nganh'].astype(str)
        
        model_path = os.path.join(BASE_DIR, 'score_predictor_model.pkl')
        ml_model = joblib.load(model_path) if os.path.exists(model_path) else None
        
        return df_students, df_rankings, df_majors, df_admissions, df_mock_exams, ml_model
    except Exception as e:
        print(f"Lỗi khi tải tài nguyên: {e}")
        return None, None, None, None, None, None

df_students, df_rankings, df_majors, df_admissions, df_mock_exams, ml_model = load_resources()


# ===============================================
# 2. TRA CỨU HỒ SƠ & BẢNG ĐIỂM THÔNG TIN HỌC SINH
# ===============================================
def get_student_info(ma_hoc_sinh: str) -> str:
    if df_students is None:
        return json.dumps({"error": "Dữ liệu không khả dụng."})
    
    student = df_students[df_students['ma_hoc_sinh'].str.upper() == ma_hoc_sinh.upper().strip()]
    if student.empty:
        return json.dumps({"error": f"Không tìm thấy học sinh có mã {ma_hoc_sinh}"}, ensure_ascii=False)
    
    row = student.iloc[0]
    result = {
        "ma_hoc_sinh": row['ma_hoc_sinh'],
        "ho_va_ten": row['ho_va_ten'],
        "khoi": int(row['khoi']),
        "lop": row['lop'],
        "bang_diem": {
            "Toán": float(row['diem_toan']),
            "Ngữ văn": float(row['diem_ngu_van']),
            "Ngoại ngữ": float(row['diem_ngoai_ngu']),
            "Vật lí": float(row['diem_vat_li']),
            "Hóa học": float(row['diem_hoa_hoc']),
            "Sinh học": float(row['diem_sinh_hoc']),
            "Lịch sử": float(row['diem_lich_su']),
            "Địa lí": float(row['diem_dia_li']),
            "GDCD": float(row['diem_gdcd']),
            "Tin học": float(row['diem_tin_hoc'])
        },
        "chung_chi_ngoai_ngu": row['chung_chi_ngoai_ngu'] if pd.notna(row['chung_chi_ngoai_ngu']) else "Không có",
        "thanh_tich_ngoai_khoa": row['thanh_tich_ngoai_khoa']
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ==========================================
# 3. SO SÁNH HỌC LỰC & THỨ HẠNG 
# ==========================================
def get_academic_ranking(ma_hoc_sinh: str) -> str:
    if df_rankings is None:
        return json.dumps({"error": "Dữ liệu xếp hạng không khả dụng."})
    
    student_rank = df_rankings[df_rankings['ma_hoc_sinh'].str.upper() == ma_hoc_sinh.upper().strip()]
    if student_rank.empty:
        return json.dumps({"error": f"Không tìm thấy thông tin xếp hạng cho mã HS {ma_hoc_sinh}"}, ensure_ascii=False)
    
    row = student_rank.iloc[0]
    result = {
        "ma_hoc_sinh": row['ma_hoc_sinh'],
        "diem_trung_binh_chung": float(row['diem_trung_binh_chung']),
        "so_sanh_voi_lop": row['so_sanh_voi_lop'],
        "so_sanh_voi_toan_khoi": row['so_sanh_voi_toan_khoi'],
        "xep_hang_tung_mon_so_voi_lop": row['xep_hang_tung_mon_so_voi_lop'],
        "xep_hang_tung_mon_so_voi_khoi": row['xep_hang_tung_mon_so_voi_khoi']
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ==========================================
# 4. TRA CỨU LỊCH SỬ THI THỬ 
# ==========================================
def get_mock_exam_history(ma_hoc_sinh: str) -> str:
    if df_mock_exams is None:
        return json.dumps({"error": "Dữ liệu thi thử không khả dụng."})
    
    exams = df_mock_exams[df_mock_exams['ma_hoc_sinh'].str.upper() == ma_hoc_sinh.upper().strip()]
    if exams.empty:
        return json.dumps({"error": f"Không tìm thấy lịch sử thi thử cho mã HS {ma_hoc_sinh}"}, ensure_ascii=False)
    
    results = []
    for _, row in exams.iterrows():
        results.append({
            "ma_hoc_sinh": row['ma_hoc_sinh'],
            "ngay_thi": str(row['ngay_thi']),
            "khoi_thi": row['khoi_thi'],
            "diem_mon_1": float(row['diem_mon_1']),
            "diem_mon_2": float(row['diem_mon_2']),
            "diem_mon_3": float(row['diem_mon_3']),
            "tong_diem_3_mon": float(row['tong_diem_3_mon']),
            "diem_trung_binh": float(row['diem_trung_binh'])
        })
    return json.dumps(results, ensure_ascii=False, indent=2)


# ==========================================
# 5. DỰ ĐOÁN ĐIỂM BẰNG ML MODEL (THANG 30)
# ==========================================
def predict_score_and_recommend(ma_hoc_sinh: str, khoi_thi: str) -> str:
    if df_students is None:
        return json.dumps({"error": "Dữ liệu không khả dụng."})
    
    khoi_thi_clean = khoi_thi.upper().strip()
    if khoi_thi_clean not in KHOI_THI_MAP:
        return json.dumps({"error": f"Khối thi {khoi_thi} không hợp lệ. Hỗ trợ: {list(KHOI_THI_MAP.keys())}"}, ensure_ascii=False)
    
    student = df_students[df_students['ma_hoc_sinh'].str.upper() == ma_hoc_sinh.upper().strip()]
    if student.empty:
        return json.dumps({"error": f"Không tìm thấy học sinh có mã {ma_hoc_sinh}"}, ensure_ascii=False)
    
    row = student.iloc[0]
    cols = KHOI_THI_MAP[khoi_thi_clean]
    diem_chi_tiet = {col: float(row[col]) for col in cols}
    
    if ml_model is not None:
        input_data = [[diem_chi_tiet[col] for col in cols]]
        predicted_total_score = float(ml_model.predict(input_data)[0])
    else:
        predicted_total_score = sum(diem_chi_tiet.values())

    predicted_total_score = round(predicted_total_score, 2)
    
    result = {
        "ma_hoc_sinh": row['ma_hoc_sinh'],
        "ho_va_ten": row['ho_va_ten'],
        "khoi_thi": khoi_thi_clean,
        "chi_tiet_3_mon": diem_chi_tiet,
        "tong_diem_3_mon_du_doan_ml": predicted_total_score
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ==========================================
# 6. TRA CỨU ĐIỂM CHUẨN, HỌC PHÍ & GỢI Ý TRƯỜNG 
# ==========================================
def recommend_universities(khoi_thi: str, tong_diem_3_mon_du_doan: float) -> str:
    if df_admissions is None or df_majors is None:
        return json.dumps({"error": "Dữ liệu tuyển sinh không khả dụng."})
    
    khoi_clean = khoi_thi.upper().strip()
    diem_trung_binh_so_sanh = tong_diem_3_mon_du_doan / 3.0
    
    filtered = df_admissions[
        (df_admissions['khoi_xet_tuyen'].str.contains(khoi_clean, case=False, na=False)) &
        (df_admissions['diem_chuan'] <= diem_trung_binh_so_sanh)
    ].sort_values(by='diem_chuan', ascending=False)
    
    merged = pd.merge(filtered, df_majors[['ma_nganh', 'ten_nganh']], on='ma_nganh', how='left')
    
    results = []
    for _, row in merged.head(5).iterrows():
        diem_chuan_thang_30 = round(float(row['diem_chuan']) * 3.0, 2)
        results.append({
            "ten_truong": row['ten_truong'],
            "ma_truong": row['ma_truong'],
            "ten_nganh": row['ten_nganh'] if pd.notna(row['ten_nganh']) else f"Ngành {row['ma_nganh']}",
            "diem_chuan_thang_30": diem_chuan_thang_30,
            "hoc_phi_du_kien": row['hoc_phi'],
            "chinh_sach_hoc_bong": row['chinh_sach_hoc_bong'] if pd.notna(row['chinh_sach_hoc_bong']) else "Không có"
        })
        
    if not results:
        return json.dumps({
            "thong_bao": f"Không tìm thấy trường có điểm chuẩn <= {tong_diem_3_mon_du_doan} cho khối {khoi_clean}.",
            "goi_y": "Học sinh có thể xem xét thêm các ngành hoặc khối thi khác."
        }, ensure_ascii=False, indent=2)
        
    return json.dumps(results, ensure_ascii=False, indent=2)


# ==========================================
# 7. TRA CỨU NGÀNH HỌC & TỔ HỢP XÉT TUYỂN 
# ==========================================
def get_major_guidance(ten_nganh_hoac_tukhoa: str) -> str:
    if df_majors is None:
        return json.dumps({"error": "Dữ liệu ngành học không khả dụng."})
    
    kw = ten_nganh_hoac_tukhoa.lower().strip()
    results_df = df_majors[
        df_majors['ten_nganh'].str.lower().str.contains(kw, na=False) |
        df_majors['mon_hoc_trong_tam'].str.lower().str.contains(kw, na=False)
    ]
    
    if results_df.empty:
        return json.dumps({"thong_bao": f"Không tìm thấy ngành học nào với từ khóa: {ten_nganh_hoac_tukhoa}"}, ensure_ascii=False)
    
    results = []
    for _, row in results_df.iterrows():
        results.append({
            "ma_nganh": str(row['ma_nganh']),
            "ten_nganh": row['ten_nganh'],
            "khoi_thi": row['khoi_thi'],
            "to_hop_mon_xet_tuyen": row['to_hop_mon_xet_tuyen'] if 'to_hop_mon_xet_tuyen' in row and pd.notna(row['to_hop_mon_xet_tuyen']) else row['khoi_thi'],
            "mo_ta_nganh": row['mo_ta_nganh'],
            "to_chat_phu_hop": row['to_chat_phu_hop'],
            "co_hoi_viec_lam": row['co_hoi_viec_lam'],
            "muc_luong_khoi_diem": row['muc_luong_khoi_diem']
        })
        
    return json.dumps(results, ensure_ascii=False, indent=2)

# ==========================================
# KHỐI TEST CHẠY THỬ TRÊN GITHUB ACTIONS
# ==========================================
if __name__ == "__main__":
    print("=== 1. TEST THÔNG TIN HỌC SINH ===")
    print(get_student_info("HS1001"))
    
    print("\n=== 2. TEST SO SÁNH THỨ HẠNG ===")
    print(get_academic_ranking("HS1001"))
    
    print("\n=== 3. TEST LỊCH SỬ THI THỬ ===")
    print(get_mock_exam_history("HS1181"))
    
    print("\n=== 4. TEST DỰ ĐOÁN ĐIỂM (MODEL ML) ===")
    print(predict_score_and_recommend("HS1001", "A00"))
    
    print("\n=== 5. TEST GỢI Ý TRƯỜNG ĐẠI HỌC ===")
    print(recommend_universities("A00", 24.5))
    
    print("\n=== 6. TEST ĐỊNH HƯỚNG NGÀNH HỌC ===")
    print(get_major_guidance("Trí tuệ nhân tạo"))



# ALL_TOOL

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_student_info",
            "description": "Tra cứu thông tin cá nhân và bảng điểm học bạ 10 môn của học sinh.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {"type": "string", "description": "Mã học sinh, VD: HS1001"}
                },
                "required": ["ma_hoc_sinh"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_score_and_recommend",
            "description": "Dùng mô hình ML dự đoán điểm trung bình môn theo khối thi (thang 10).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ma_hoc_sinh": {"type": "string", "description": "Mã học sinh, VD: HS1001"},
                    "khoi_thi": {"type": "string", "description": "Khối thi, VD: A00, D01"}
                },
                "required": ["ma_hoc_sinh", "khoi_thi"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_universities",
            "description": "Lọc tối đa 5 trường đại học có điểm chuẩn <= điểm trung bình dự đoán.",
            "parameters": {
                "type": "object",
                "properties": {
                    "khoi_thi": {"type": "string", "description": "Khối thi, VD: A00, D01"},
                    "diem_trung_binh_du_doan": {"type": "number", "description": "Điểm trung bình môn dự đoán (thang điểm 10)"}
                },
                "required": ["khoi_thi", "diem_trung_binh_du_doan"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_major_guidance",
            "description": "Tra cứu mô tả ngành, tố chất phù hợp và cơ hội việc làm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ten_nganh_hoac_tukhoa": {"type": "string", "description": "Tên ngành học, VD: Trí tuệ nhân tạo"}
                },
                "required": ["ten_nganh_hoac_tukhoa"]
            }
        }
    }
]

TOOL_MAPPER = {
    "get_student_info": get_student_info,
    "predict_score_and_recommend": predict_score_and_recommend,
    "recommend_universities": recommend_universities,
    "get_major_guidance": get_major_guidance
}
