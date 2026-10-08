# Nguyên
import pandas as pd
import json
import os

# ==========================================
# 1. CẤU HÌNH BẢNG ÁNH XẠ KHỐI THI (KHOI_THI_MAP)
# ==========================================
# Ánh xạ tên khối thi với 3 cột điểm môn học tương ứng trong file 01
KHOI_THI_MAP = {
    "A00": ["diem_toan", "diem_vat_li", "diem_hoa_hoc"],
    "A01": ["diem_toan", "diem_vat_li", "diem_ngoai_ngu"],
    "B00": ["diem_toan", "diem_hoa_hoc", "diem_sinh_hoc"],
    "C00": ["diem_ngu_van", "diem_lich_su", "diem_dia_li"],
    "D01": ["diem_toan", "diem_ngu_van", "diem_ngoai_ngu"],
    "D07": ["diem_toan", "diem_hoa_hoc", "diem_ngoai_ngu"]
}

# Đọc các file dữ liệu CSV
DATA_DIR = "./"

def load_datasets():
    try:
        df_students = pd.read_csv(os.path.join(DATA_DIR, '01_Danh_sach_va_Thanh_tich_hoc_sinh.csv'))
        df_majors = pd.read_csv(os.path.join(DATA_DIR, '03_Nganh_hoc_va_Huong_nghiep.csv'))
        df_admissions = pd.read_csv(os.path.join(DATA_DIR, '04_Tuyen_sinh_va_Hoc_phi.csv'))
        return df_students, df_majors, df_admissions
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None, None, None

df_students, df_majors, df_admissions = load_datasets()


# ==========================================
# 2. HÀM TRA CỨU THÔNG TIN HỌC SINH
# ==========================================
def get_student_info(ma_hoc_sinh: str) -> str:
    """
    Tra cứu thông tin cá nhân (họ tên, lớp) và bảng điểm học bạ đầy đủ 10 môn của học sinh.
    """
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
# 3. HÀM DỰ ĐOÁN ĐIỂM THI THPT & TÍNH TỔNG
# ==========================================
def predict_score_and_recommend(ma_hoc_sinh: str, khoi_thi: str) -> str:
    """
    Lấy điểm học bạ 3 môn thuộc khối thi của học sinh và tính tổng điểm dự đoán.
    """
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
    tong_diem_du_doan = round(sum(diem_chi_tiet.values()), 2)
    
    result = {
        "ma_hoc_sinh": row['ma_hoc_sinh'],
        "ho_va_ten": row['ho_va_ten'],
        "khoi_thi": khoi_thi_clean,
        "chi_tiet_3_mon": diem_chi_tiet,
        "tong_diem_du_doan": tong_diem_du_doan
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ==========================================
# 4. HÀM GỢI Ý TRƯỜNG ĐẠI HỌC
# ==========================================
def recommend_universities(khoi_thi: str, tong_diem_du_doan: float) -> str:
    """
    Lọc ra tối đa 5 trường đại học phù hợp với khối thi và có điểm chuẩn năm trước <= tổng điểm dự đoán.
    """
    if df_admissions is None or df_majors is None:
        return json.dumps({"error": "Dữ liệu tuyển sinh không khả dụng."})
    
    khoi_clean = khoi_thi.upper().strip()
    
    # Lọc các trường chấp nhận khối thi và có điểm chuẩn <= tổng điểm dự đoán
    filtered = df_admissions[
        (df_admissions['khoi_xet_tuyen'].str.contains(khoi_clean, case=False, na=False)) &
        (df_admissions['diem_chuan'] <= tong_diem_du_doan)
    ].sort_values(by='diem_chuan', ascending=False)
    
    # Kết hợp thông tin tên ngành từ file 03 qua ma_nganh
    merged = pd.merge(filtered, df_majors[['ma_nganh', 'ten_nganh']], on='ma_nganh', how='left')
    
    results = []
    for _, row in merged.head(5).iterrows():
        results.append({
            "ten_truong": row['ten_truong'],
            "ma_truong": row['ma_truong'],
            "ten_nganh": row['ten_nganh'] if pd.notna(row['ten_nganh']) else f"Ngành {row['ma_nganh']}",
            "diem_chuan_nam_truoc": float(row['diem_chuan']),
            "hoc_phi_du_kien": row['hoc_phi']
        })
        
    if not results:
        return json.dumps({
            "thong_bao": f"Không tìm thấy trường có điểm chuẩn <= {tong_diem_du_doan} cho khối {khoi_clean}.",
            "goi_y": "Học sinh có thể cân nhắc cải thiện điểm thi hoặc xem xét các trường/khối khác."
        }, ensure_ascii=False, indent=2)
        
    return json.dumps(results, ensure_ascii=False, indent=2)


# ==========================================
# 5. HÀM TRA CỨU ĐỊNH HƯỚNG NGÀNH HỌC
# ==========================================
def get_major_guidance(ten_nganh_hoac_tukhoa: str) -> str:
    """
    Tìm kiếm thông tin ngành học từ file 03_Nganh_hoc_va_Huong_nghiep.csv.
    """
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
            "ma_nganh": int(row['ma_nganh']),
            "ten_nganh": row['ten_nganh'],
            "khoi_thi": row['khoi_thi'],
            "mo_ta_nganh": row['mo_ta_nganh'],
            "to_chat_phu_hop": row['to_chat_phu_hop'],
            "co_hoi_viec_lam": row['co_hoi_viec_lam'],
            "muc_luong_khoi_diem": row['muc_luong_khoi_diem']
        })
        
    return json.dumps(results, ensure_ascii=False, indent=2)


