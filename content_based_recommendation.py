import pandas as pd
from gensim import corpora, models, similarities
import ast

def recommend_products(product_id, df, weight_content=0.7, weight_rating=0.3, top_n=6):
    """
    Gợi ý sản phẩm tương tự dựa trên nội dung.
    """
    # Đảm bảo cột 'tokens' có định dạng đúng
    df['tokens'] = df['tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Tạo dictionary Gensim
    dictionary = corpora.Dictionary(df['tokens'])
    corpus = [dictionary.doc2bow(text) for text in df['tokens']]

    # Tạo mô hình TF-IDF và biến đổi corpus
    tfidf = models.TfidfModel(corpus)
    tfidf_corpus = tfidf[corpus]

    # Tạo similarity index
    index = similarities.SparseMatrixSimilarity(tfidf_corpus, num_features=len(dictionary.token2id))
    
    # Lấy chỉ số của sản phẩm đầu vào
    try:
        query_idx = df[df['ma_san_pham'] == product_id].index[0]
    except IndexError:
        raise ValueError("Mã sản phẩm không tồn tại trong dữ liệu.")
    
    query_bow = dictionary.doc2bow(df.iloc[query_idx]['tokens'])
    
    # Tính điểm tương tự
    sims = index[tfidf[query_bow]]
    final_scores = []
    
    for idx, sim_score in enumerate(sims):
        rating = df.iloc[idx]['diem_trung_binh']
        final_score = sim_score * weight_content + rating * weight_rating
        final_scores.append((idx, final_score))
    
    # Sắp xếp theo điểm số kết hợp
    final_scores_sorted = sorted(final_scores, key=lambda item: -item[1])
    
    # Lấy top sản phẩm gợi ý
    top_similar_products = []
    for item in final_scores_sorted[:top_n]:
        product_idx = item[0]
        top_similar_products.append({
            'ma_san_pham': df.iloc[product_idx]['ma_san_pham'],
            'ten_san_pham': df.iloc[product_idx]['ten_san_pham'],
            'similarity_score': sims[product_idx],
            'average_rating': df.iloc[product_idx]['diem_trung_binh'],
            'final_score': item[1],
            'hinh_anh': df.iloc[product_idx]['hinh_anh'],  # Lấy trực tiếp từ cột 'hinh_anh'
            'gia_ban': df.iloc[product_idx].get('gia_ban', 'Không có thông tin'),  # Thêm giá bán
            'gia_goc': df.iloc[product_idx].get('gia_goc', 'Không có thông tin'),  # Thêm giá gốc
            'mo_ta': df.iloc[product_idx].get('mo_ta', 'Không có mô tả.')  # Thêm mô tả sản phẩm
        })
    
    # Chuyển đổi danh sách gợi ý thành DataFrame
    df_recommendations = pd.DataFrame(top_similar_products)
    
    return df_recommendations

