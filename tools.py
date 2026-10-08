import pandas as pd
import json
import os
import joblib

# ==========================================
# 1. ĐƯỜNG DẪN DỮ LIỆU ĐỘNG (DYNAMIC PATH)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cấu hình ánh xạ khối thi
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
       
        df_students = pd.read_csv(os.path.join(BASE_DIR, '01_Danh_sach_va_Thanh_tich_hoc_sinh_clean.csv'))
        df_majors = pd.read_csv(os.path.join(BASE_DIR, '03_Nganh_hoc_va_Huong_nghiep_clean.csv'))
        df_admissions = pd.read_csv(os.path.join(BASE_DIR, '04_Tuyen_sinh_va_Hoc_phi_clean.csv'))
        
        df_majors['ma_nganh'] = df_majors['ma_nganh'].astype(str)
        df_admissions['ma_nganh'] = df_admissions['ma_nganh'].astype(str)
        
        model_path = os.path.join(BASE_DIR, 'score_predictor_model.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join(BASE_DIR, 'score_predictor_model.pkl')
            
        ml_model = joblib.load(model_path) if os.path.exists(model_path) else None
        
        return df_students, df_majors, df_admissions, ml_model
    except Exception as e:
        print(f"Lỗi khi tải tài nguyên: {e}")
        return None, None, None, None

df_students, df_majors, df_admissions, ml_model = load_resources()


# ==========================================
# 2. HÀM TRA CỨU THÔNG TIN HỌC SINH
# ==========================================
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
# 3. TÍNH ĐIỂM DỰ ĐOÁN BẰNG ML MODEL (THANG 10)
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
    
    # Dự đoán điểm qua mô hình ML (Thang 10 - Trung bình môn)
    if ml_model is not None:
        input_data = [[diem_chi_tiet[col] for col in cols]]
        predicted_score = float(ml_model.predict(input_data)[0])
    else:
        # Nếu chưa nạp được file .pkl, dùng trung bình cộng 3 môn
        predicted_score = sum(diem_chi_tiet.values()) / 3.0

    predicted_score = round(predicted_score, 2)
    
    result = {
        "ma_hoc_sinh": row['ma_hoc_sinh'],
        "ho_va_ten": row['ho_va_ten'],
        "khoi_thi": khoi_thi_clean,
        "chi_tiet_3_mon": diem_chi_tiet,
        "diem_trung_binh_du_doan_ml": predicted_score
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ==========================================
# 4. MERGE VÀ SO SÁNH ĐÚNG THANG ĐIỂM
# ==========================================
def recommend_universities(khoi_thi: str, diem_trung_binh_du_doan: float) -> str:
    if df_admissions is None or df_majors is None:
        return json.dumps({"error": "Dữ liệu tuyển sinh không khả dụng."})
    
    khoi_clean = khoi_thi.upper().strip()
    
    filtered = df_admissions[
        (df_admissions['khoi_xet_tuyen'].str.contains(khoi_clean, case=False, na=False)) &
        (df_admissions['diem_chuan'] <= diem_trung_binh_du_doan)
    ].sort_values(by='diem_chuan', ascending=False)
    
    merged = pd.merge(filtered, df_majors[['ma_nganh', 'ten_nganh']], on='ma_nganh', how='left')
    
    results = []
    for _, row in merged.head(5).iterrows():
        results.append({
            "ten_truong": row['ten_truong'],
            "ma_truong": row['ma_truong'],
            "ten_nganh": row['ten_nganh'] if pd.notna(row['ten_nganh']) else f"Ngành {row['ma_nganh']}",
            "diem_chuan_trung_binh_mon": float(row['diem_chuan']),
            "hoc_phi_du_kien": row['hoc_phi']
        })
        
    if not results:
        return json.dumps({
            "thong_bao": f"Không tìm thấy trường có điểm chuẩn <= {diem_trung_binh_du_doan} cho khối {khoi_clean}.",
            "goi_y": "Học sinh có thể xem xét thêm các ngành hoặc khối thi khác."
        }, ensure_ascii=False, indent=2)
        
    return json.dumps(results, ensure_ascii=False, indent=2)


# ==========================================
# 5. HÀM TRA CỨU ĐỊNH HƯỚNG NGÀNH HỌC
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
            "mo_ta_nganh": row['mo_ta_nganh'],
            "to_chat_phu_hop": row['to_chat_phu_hop'],
            "co_hoi_viec_lam": row['co_hoi_viec_lam'],
            "muc_luong_khoi_diem": row['muc_luong_khoi_diem']
        })
        
    return json.dumps(results, ensure_ascii=False, indent=2)

