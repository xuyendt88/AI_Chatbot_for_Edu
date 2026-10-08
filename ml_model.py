import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

KHOI_THI_MAP = {
    'A00': ['diem_toan', 'diem_vat_li', 'diem_hoa_hoc'],
    'A01': ['diem_toan', 'diem_vat_li', 'diem_ngoai_ngu'],
    'B00': ['diem_toan', 'diem_hoa_hoc', 'diem_sinh_hoc'],
    'C00': ['diem_ngu_van', 'diem_lich_su', 'diem_dia_li'],
    'D01': ['diem_toan', 'diem_ngu_van', 'diem_ngoai_ngu'],
    'D07': ['diem_toan', 'diem_hoa_hoc', 'diem_ngoai_ngu'],
}


def train_and_save_models():
    f1_path = os.path.join(
        DATA_DIR, "01_Danh_sach_va_Thanh_tich_hoc_sinh_clean.csv"
    )
    f5_path = os.path.join(DATA_DIR, "05_Ket_qua_thi_thu_clean.csv")

    if not os.path.exists(f1_path) or not os.path.exists(f5_path):
        print("❌ Không tìm thấy file dữ liệu CSV đầu vào trong thư mục ./data!")
        return

    df1 = pd.read_csv(f1_path)
    df5 = pd.read_csv(f5_path)

    # Merge dữ liệu dựa trên mã học sinh
    df = pd.merge(df5, df1, on="ma_hoc_sinh", how="inner")

    print(f"Tổng số bản ghi sau khi merge: {len(df)}")

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        khoi = str(row.get("khoi_thi", "")).strip().upper()
        if khoi in KHOI_THI_MAP:
            cols = KHOI_THI_MAP[khoi]
            m1 = row.get(cols[0], np.nan)
            m2 = row.get(cols[1], np.nan)
            m3 = row.get(cols[2], np.nan)
            target = row.get("tong_diem_3_mon", np.nan)

            if not (
                pd.isna(m1) or pd.isna(m2) or pd.isna(m3) or pd.isna(target)
            ):
                X_list.append([m1, m2, m3])
                y_list.append(target)

    X = np.array(X_list)
    y = np.array(y_list)

    if len(X) < 5:
        print("❌ Không đủ dữ liệu hợp lệ để huấn luyện mô hình!")
        return

    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("✅ Huấn luyện mô hình thành công!")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R2 Score: {r2:.4f}")

    # Lưu mô hình ra file score_predictor_model.pkl ở thư mục dự án
    output_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(output_dir, "score_predictor_model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Đã lưu mô hình vào file: {model_path}")


# Quan trọng: Đoạn này để thực thi hàm khi chạy lệnh python ml_model.py
if __name__ == "__main__":
    train_and_save_models()
