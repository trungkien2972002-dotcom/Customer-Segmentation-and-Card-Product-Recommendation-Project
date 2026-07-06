import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def run_customer_segmentation_pipeline(data_path, output_path):
    print("="*60)
    print("STARTING CUSTOMER SEGMENTATION PIPELINE")
    print("="*60)
    
    # -------------------------------------------------------------------------
    # BƯỚC 1: ĐỌC DỮ LIỆU GỐC
    # -------------------------------------------------------------------------
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại đường dẫn: {data_path}")
        
    print(f"[1/5] Đang đọc file dữ liệu gốc từ: {data_path}...")
    df_original = pd.read_csv(data_path)
    
    # -------------------------------------------------------------------------
    # BƯỚC 2: LÀM SẠCH VÀ TIỀN XỬ LÝ (DROPNA & DROP ID)
    # -------------------------------------------------------------------------
    print("[2/5] Đang tiến hành làm sạch dữ liệu (Drop Missing Values)...")
    # Loại bỏ CUST_ID để tránh nhiễu mô hình toán, loại bỏ dòng khuyết thiếu
    df_cleaned = df_original.drop(columns=['CUST_ID']).dropna()
    
    # Kỷ lục lại index sau khi dropna để phục vụ map CUST_ID sau này
    valid_indices = df_cleaned.index
    df_cleaned = df_cleaned.reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # BƯỚC 3: BIẾN ĐỔI LOG THEO ĐẶC TRƯNG PHÂN PHỐI DỮ LIỆU
    # -------------------------------------------------------------------------
    print("[3/5] Đang áp dụng Data Transformation (Log1p cho nhóm lệch phải nặng)...")
    # Biến giữ nguyên: Cặp phân phối chữ U và cặp lệch trái (theo quan sát thực tế)
    columns_to_keep = ['PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'BALANCE_FREQUENCY', 'TENURE']
    # Biến lệch phải nặng cần nén đuôi dài
    columns_to_log = [col for col in df_cleaned.columns if col not in columns_to_keep]
    
    df_transformed = df_cleaned.copy()
    df_transformed[columns_to_log] = df_transformed[columns_to_log].apply(np.log1p)
    
    # -------------------------------------------------------------------------
    # BƯỚC 4: CHUẨN HÓA THANG ĐO VÀ HUẤN LUYỆN K-MEANS (K=4)
    # -------------------------------------------------------------------------
    print("[4/5] Đang chuẩn hóa dữ liệu (StandardScaler) và chạy mô hình K-Means...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_transformed)
    
    # Định cấu hình K-Means chuẩn để bàn giao hệ thống production (K=4, random_state cố định)
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Gán nhãn cụm vào bảng dữ liệu sạch ban đầu (dạng thô để IT/Marketing dễ đọc)
    df_cleaned['Cluster'] = cluster_labels
    
    # -------------------------------------------------------------------------
    # BƯỚC 5: KHÔI PHỤC CUST_ID GỐC VÀ ĐÓNG GÓI KẾT QUẢ
    # -------------------------------------------------------------------------
    print("[5/5] Đang khôi phục lại mã định danh CUST_ID chính xác theo index...")
    # Sử dụng .loc với dấu ngoặc vuông [] dựa trên mảng valid_indices đã lưu lại ở Bước 2
    df_cleaned['CUST_ID'] = df_original['CUST_ID'].loc[valid_indices].values
    
    # Đẩy cột CUST_ID lên vị trí đầu bảng tiêu chuẩn
    final_columns = ['CUST_ID', 'Cluster'] + [col for col in df_cleaned.columns if col not in ['CUST_ID', 'Cluster']]
    df_final = df_cleaned[final_columns]
    
    # Xuất file kết quả bàn giao
    df_final.to_csv(output_path, index=False)
    print(f"🎉 PIPELINE THÀNH CÔNG! Đã xuất file kết quả tại: {output_path}")
    print(f"Kích thước tệp dữ liệu đầu ra: {df_final.shape}")
    print("="*60)

if __name__ == "__main__":
    # 1. Lấy đường dẫn của chính thư mục chứa file Full_scrip.py hiện tại (tức là thư mục 'Scr')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Định nghĩa chính xác đường dẫn đến file dữ liệu nằm trong thư mục 'Data'
    # os.path.join sẽ tự động xử lý dấu gạch chéo chuẩn theo hệ điều hành Windows/Linux
    INPUT_FILE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Data', 'CC GENERAL.csv'))
    OUTPUT_FILE_PATH = os.path.join(BASE_DIR, '..', 'final_segmented_customers.csv')
    run_customer_segmentation_pipeline(INPUT_FILE_PATH, OUTPUT_FILE_PATH)