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
    # 1. Tìm trong thư mục data/ (theo đúng cấu trúc repo GitHub)
    path_data = os.path.join(BASE_DIR, 'data', filename)
    if os.path.exists(path_data):
        return path_data
    
    # 2. Tìm trực tiếp ở thư mục gốc
    path_root = os.path.join(BASE_DIR, filename)
    if os.path.exists(path_root):
        return path_root
        
    # 3. Tìm trong thư mục con khác nếu có
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

    # 2. Xử lý file Tuyển sinh & Học phí
    df_tuyen_sinh['diem_chuan_thang_30'] = df_tuyen_sinh['diem_chuan'] * 3.0
    df_tuyen_sinh['hoc_phi_so'] = df_tuyen_sinh['hoc_phi'].apply(clean_tuition)

    # 3. Merge dữ liệu học sinh
    df_student_profile = pd.merge(df_hoc_sinh, df_xep_hang, on='ma_hoc_sinh', how='left')
    df_student_profile = pd.merge(df_student_profile, df_thi_thu, on='ma_hoc_sinh', how='left')

    print("✅ Đã hoàn tất làm sạch dữ liệu!")
    return {
        'students': df_student_profile,
        'majors': df_nganh,
        'admissions': df_tuyen_sinh
    }

if __name__ == "__main__":
    cleaned_data = load_and_clean_data()
    print("Mẫu dữ liệu học sinh sau khi làm sạch:")
    print(cleaned_data['students'].head())
