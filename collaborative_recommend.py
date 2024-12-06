import pandas as pd
import pickle

def recommend_products(full_data_file, model_file, customer_id, top_n=6):
    # Load dữ liệu
    full_data = pd.read_csv(full_data_file)
    
    # Chuẩn hóa cột và mã khách hàng
    full_data['ma_khach_hang'] = full_data['ma_khach_hang'].astype(str).str.strip()
    full_data['ma_san_pham'] = full_data['ma_san_pham'].astype(str).str.strip()
    customer_id = str(customer_id).strip()
    
    # Load mô hình
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Lọc các sản phẩm đã đánh giá cao
    df_selected = full_data[(full_data['ma_khach_hang'] == customer_id) & (full_data['so_sao'] >= 3)]['ma_san_pham']
    
    # Chuẩn bị danh sách sản phẩm để dự đoán
    df_score = full_data[['ma_san_pham']].drop_duplicates()
    df_score = df_score[~df_score['ma_san_pham'].isin(df_selected)]
    
    # Dự đoán điểm
    df_score['EstimateScore'] = df_score['ma_san_pham'].apply(
        lambda x: model.predict(customer_id, x).est
    )
    
    # Loại bỏ sản phẩm trùng lặp và lấy top 6
    recommendations = df_score.drop_duplicates(subset='ma_san_pham')\
                               .sort_values(by='EstimateScore', ascending=False)\
                               .head(top_n)
    
    # Gộp thông tin sản phẩm và loại bỏ trùng lặp
    recommendations = recommendations.merge(
        full_data.drop_duplicates(subset='ma_san_pham'), 
        on='ma_san_pham', 
        how='left'
    ).drop_duplicates(subset='ma_san_pham')
    
    return recommendations




