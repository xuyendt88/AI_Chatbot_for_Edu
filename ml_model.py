import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Ưu tiên import từ data_cleaning.py theo đúng cấu trúc repo GitHub
try:
    from data_cleaning import load_and_clean_data
except ImportError:
    from machine import load_and_clean_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def prepare_training_data(cleaned_data):
    """
    Tạo tập dữ liệu huấn luyện thực tế từ dữ liệu đã làm sạch:
    - X_train: Chênh lệch điểm (Điểm 3 môn thi - Điểm chuẩn thang 30)
    - y_train: Nhãn trúng tuyển (1: Đỗ, 0: Trượt)
    """
    df_students = cleaned_data['students']
    df_admissions = cleaned_data['admissions']

    # Chỉ lấy học sinh lớp 12 có điểm thi 3 môn
    valid_students = df_students[df_students['tong_diem_3_mon'].notna()].copy()

    X_list = []
    y_list = []

    # So sánh điểm của từng học sinh với tất cả các ngành trong file tuyển sinh
    for _, student in valid_students.iterrows():
        student_score = student['tong_diem_3_mon']
        
        for _, major in df_admissions.iterrows():
            cutoff_score = major['diem_chuan_thang_30']
            
            # Tính độ chênh lệch điểm (Điểm thi - Điểm chuẩn)
            delta_score = student_score - cutoff_score
            
            # Quy ước nhãn: Nếu chênh lệch >= 0 thì Trúng tuyển (1), ngược lại Trượt (0)
            label = 1 if delta_score >= 0 else 0
            
            X_list.append(delta_score)
            y_list.append(label)

    X = np.array(X_list).reshape(-1, 1)
    y = np.array(y_list)
    
    return X, y

def train_and_save_model():
    """Tải dữ liệu sạch, huấn luyện Logistic Regression và lưu thành file .pkl"""
    print("⏳ Bước 1: Tải và chuẩn bị dữ liệu sạch cho Machine Learning...")
    cleaned_data = load_and_clean_data()
    
    # Chuẩn bị X (chênh lệch điểm) và y (kết quả trúng tuyển)
    X_train, y_train = prepare_training_data(cleaned_data)
    print(f"📊 Đã tạo {len(X_train)} mẫu dữ liệu huấn luyện từ học sinh và ngành học.")

    print("\n⏳ Bước 2: Đang huấn luyện mô hình Logistic Regression...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Bước 3: Lưu mô hình vào file admission_model.pkl
    model_path = os.path.join(BASE_DIR, 'admission_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"\n✅ Đã huấn luyện thành công và lưu file tại: {model_path}")
    
    # Chạy thử nghiệm một vài dự đoán mẫu (Sửa truyền mảng 2D để tránh cảnh báo)
    print("\n" + "="*50)
    print("📌 THỬ NHIỆM DỰ ĐOÁN XÁC SUẤT TRÚNG TUYỂN MẪU:")
    print("="*50)
    test_deltas = [-3.0, -1.0, 0.0, 1.5, 4.0]
    for delta in test_deltas:
        prob = model.predict_proba([[delta]])[0][1] * 100
        status = "Thừa điểm" if delta >= 0 else "Thiếu điểm"
        print(f"Chênh lệch: {delta:>5.1f} điểm ({status:<10}) ---> Tỉ lệ đỗ: {prob:.2f}%")

    return model

if __name__ == "__main__":
    train_and_save_model()
