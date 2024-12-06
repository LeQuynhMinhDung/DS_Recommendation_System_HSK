import pandas as pd
import gzip
import pickle

def recommend_products(data_files, model_file, customer_id, top_n=6):
    # Load dữ liệu từ nhiều file
    full_data = pd.concat([pd.read_csv(f) for f in data_files])

    # Chuẩn hóa cột
    full_data['ma_khach_hang'] = full_data['ma_khach_hang'].astype(str).str.strip()
    full_data['ma_san_pham'] = full_data['ma_san_pham'].astype(str).str.strip()
    customer_id = str(customer_id).strip()

    # Load model
    with gzip.open(model_file, 'rb') as f:
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

    # Lọc và sắp xếp
    recommendations = df_score.sort_values(by='EstimateScore', ascending=False).head(top_n)
    recommendations = recommendations.merge(full_data.drop_duplicates(subset='ma_san_pham'), on='ma_san_pham', how='left')
    return recommendations



