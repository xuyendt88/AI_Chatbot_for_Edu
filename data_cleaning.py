import os
import re
import pandas as pd

# Lấy đường dẫn gốc của dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_tuition(text):
    """Trích xuất khoảng học phí trung bình (triệu VNĐ/năm) từ chuỗi văn bản"""
    if pd.isna(text):
        return None
    numbers = re.findall(r'\d+', str(text))
    if len(numbers) >= 2:
        return (float(numbers[0]) + float(numbers[1])) / 2
    elif len(numbers) == 1:
        return float(numbers[0])
    return None

def get_file_path(filename):
    """Lấy đường dẫn chính xác của file CSV trong thư mục data hoặc thư mục gốc"""
    path_data = os.path.join(BASE_DIR, 'data', filename)
    if os.path.exists(path_data):
        return path_data
    
    path_root = os.path.join(BASE_DIR, filename)
    if os.path.exists(path_root):
        return path_root
        
    for root, dirs, files in os.walk(BASE_DIR):
        if filename in files:
            return os.path.join(root, filename)
            
    raise FileNotFoundError(f"❌ Không tìm thấy file {filename}")

def load_and_clean_data():
    """Hàm làm sạch và tổng hợp dữ liệu"""
    print("⏳ Đang tải và làm sạch dữ liệu...")
    
    # 1. Đọc các file dữ liệu
    df_hoc_sinh = pd.read_csv(get_file_path('01_Danh_sach_va_Thanh_tich_hoc_sinh.csv'))
    df_xep_hang = pd.read_csv(get_file_path('02_Ket_qua_va_Xep_hang_Hoc_tap.csv'))
    df_nganh = pd.read_csv(get_file_path('03_Nganh_hoc_va_Huong_nghiep.csv'))
    df_tuyen_sinh = pd.read_csv(get_file_path('04_Tuyen_sinh_va_Hoc_phi.csv'))
    df_thi_thu = pd.read_csv(get_file_path('05_Ket_qua_thi_thu.csv'))

    # 2. CHUẨN HÓA MÃ HỌC SINH VÀ DỮ LIỆU CHUỖI
    for df in [df_hoc_sinh, df_xep_hang, df_thi_thu]:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()

    # 3. XỬ LÝ FILE TUYỂN SINH & HỌC PHÍ
    if 'diem_chuan' in df_tuyen_sinh.columns:
        # Nếu điểm chuẩn ở thang 10 thì nhân 3, nếu thang 30 sẵn thì giữ nguyên
        max_score = df_tuyen_sinh['diem_chuan'].max()
        if max_score <= 10.0:
            df_tuyen_sinh['diem_chuan_thang_30'] = df_tuyen_sinh['diem_chuan'] * 3.0
        else:
            df_tuyen_sinh['diem_chuan_thang_30'] = df_tuyen_sinh['diem_chuan']

    if 'hoc_phi' in df_tuyen_sinh.columns:
        df_tuyen_sinh['hoc_phi_so'] = df_tuyen_sinh['hoc_phi'].apply(clean_tuition)

    # 4. MERGE DỮ LIỆU HỌC SINH (Loại bỏ cột trùng lặp trước khi merge)
    cols_to_drop_xep_hang = [c for c in df_xep_hang.columns if c in df_hoc_sinh.columns and c != 'ma_hoc_sinh']
    df_xep_hang_clean = df_xep_hang.drop(columns=cols_to_drop_xep_hang)

    df_student_profile = pd.merge(df_hoc_sinh, df_xep_hang_clean, on='ma_hoc_sinh', how='left')

    cols_to_drop_thi_thu = [c for c in df_thi_thu.columns if c in df_student_profile.columns and c != 'ma_hoc_sinh']
    df_thi_thu_clean = df_thi_thu.drop(columns=cols_to_drop_thi_thu)

    df_student_profile = pd.merge(df_student_profile, df_thi_thu_clean, on='ma_hoc_sinh', how='left')

    print("✅ Đã hoàn tất làm sạch dữ liệu!")
    return {
        'students': df_student_profile,
        'majors': df_nganh,
        'admissions': df_tuyen_sinh
    }

if __name__ == "__main__":
    cleaned_data = load_and_clean_data()
    df_students = cleaned_data['students']
    df_admissions = cleaned_data['admissions']

    print("\n" + "="*50)
    print("📌 KIỂM TRA 1: HỌC SINH CÓ ĐIỂM THI THỬ (LỚP 12)")
    print("="*50)
    if 'tong_diem_3_mon' in df_students.columns:
        has_scores = df_students[df_students['tong_diem_3_mon'].notna()]
        cols_student = [c for c in ['ma_hoc_sinh', 'ho_va_ten', 'khoi', 'lop', 'tong_diem_3_mon'] if c in df_students.columns]
        print(has_scores[cols_student].head())
    else:
        print(df_students.head())

    print("\n" + "="*50)
    print("📌 KIỂM TRA 2: DỮ LIỆU TUYỂN SINH & HỌC PHÍ ĐÃ LÀM SẠCH")
    print("="*50)
    cols_to_show = [c for c in ['nganh_hoc', 'ten_nganh', 'ma_nganh', 'diem_chuan', 'diem_chuan_thang_30', 'hoc_phi_so'] if c in df_admissions.columns]
    print(df_admissions[cols_to_show].head())

    print("\n" + "="*50)
    print("📌 KIỂM TRA 3: TỔNG QUAN SỐ CỘT BỊ TRỐNG (NaN)")
    print("="*50)
    nan_counts = df_students.isna().sum()
    print(nan_counts[nan_counts > 0])
