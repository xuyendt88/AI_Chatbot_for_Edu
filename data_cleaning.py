import os
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Danh sách tệp dữ liệu nằm trong thư mục data/
FILE_LIST = [
    '01_Danh_sach_va_Thanh_tich_hoc_sinh.csv',
    '02_Ket_qua_va_Xep_hang_Hoc_tap.csv',
    '03_Nganh_hoc_va_Huong_nghiep.csv',
    '04_Tuyen_sinh_va_Hoc_phi.csv',
    '05_Ket_qua_thi_thu.csv',
]


def inspect_and_clean_all():
    cleaned_dfs = {}

    for file_name in FILE_LIST:
        input_path = f'data/{file_name}'

        print('=' * 60)
        print(f'🔍 ĐANG KIỂM TRA FILE: {input_path}')
        print('=' * 60)

        if not os.path.exists(input_path):
            print(f'❌ Không tìm thấy tệp: {input_path}')
            continue

        df = pd.read_csv(input_path)

        # 1. Báo cáo tình trạng dữ liệu ban đầu
        print(f'- Kích thước ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột')
        print(f'- Số dòng trùng lặp: {df.duplicated().sum()}')
        print('- Số giá trị thiếu (Missing values) theo cột:')
        missing_info = df.isnull().sum()
        print(missing_info[missing_info > 0])
        if missing_info.sum() == 0:
            print('  (Không có giá trị khuyết thiếu)')

        # 2. Xử lý làm sạch cụ thể cho từng tệp
        df_clean = df.copy()

        # Chuẩn hóa khoảng trắng dư thừa
        for col in df_clean.select_dtypes(include=['object', 'string']).columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

        if file_name == '01_Danh_sach_va_Thanh_tich_hoc_sinh.csv':
            # Điền giá trị khuyết ở cột chứng chỉ ngoại ngữ
            if 'chung_chi_ngoai_ngu' in df_clean.columns:
                df_clean['chung_chi_ngoai_ngu'] = df_clean[
                    'chung_chi_ngoai_ngu'
                ].fillna('Không có')

        elif file_name in [
            '03_Nganh_hoc_va_Huong_nghiep.csv',
            '04_Tuyen_sinh_va_Hoc_phi.csv',
        ]:
            # Loại bỏ cột rỗng 100% nguon_tham_khao
            if 'nguon_tham_khao' in df_clean.columns:
                df_clean = df_clean.drop(columns=['nguon_tham_khao'])

        elif file_name == '05_Ket_qua_thi_thu.csv':
            # Ép kiểu dữ liệu ngày tháng
            if 'ngay_thi' in df_clean.columns:
                df_clean['ngay_thi'] = pd.to_datetime(
                    df_clean['ngay_thi'], format='%d/%m/%Y', errors='coerce'
                )

        # 3. Lưu kết quả làm sạch vào thư mục data/
        output_path = f'data/{file_name.replace(".csv", "_clean.csv")}'
        df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
        cleaned_dfs[file_name] = df_clean

        print(f'\n✅ ĐÃ LÀM SẠCH VÀ LƯU VÀO: {output_path}')
        print(
            f'- Kích thước sau làm sạch: {df_clean.shape[0]} dòng, {df_clean.shape[1]} cột'
        )
        print(
            f'- Tổng giá trị thiếu còn lại: {df_clean.isnull().sum().sum()}\n'
        )

    return cleaned_dfs


if __name__ == '__main__':
    inspect_and_clean_all()
