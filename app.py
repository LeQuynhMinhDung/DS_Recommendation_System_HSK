import streamlit as st
import pandas as pd
from collaborative_recommend import recommend_products as recommend_collaborative
from content_based_recommendation import recommend_products as recommend_content_based

# Đường dẫn tệp
CONTENT_BASED_DATA_FILE = "data/content_based_preprocessed.csv"
COLLABORATIVE_FULL_DATA_PART1 = "data/collaborative_full_data_part1.csv"
COLLABORATIVE_FULL_DATA_PART2 = "data/collaborative_full_data_part2.csv"
COLLABORATIVE_MODEL_FILE = "model/collaborative_model.pkl.gz"

# Hình ảnh mặc định
DEFAULT_IMAGE = "https://via.placeholder.com/150"

# Hiển thị banner và tiêu đề chính
st.image('banner/hasaki_banner_2.jpg', use_column_width=True)
st.title("Hệ thống gợi ý sản phẩm - Hasaki")

# Tabs để chọn giữa hai phương pháp gợi ý
tab1, tab2 = st.tabs(["Content-Based Filtering", "Collaborative Filtering"])

# Tab 1: Content-Based Filtering
with tab1:
    st.subheader("Gợi ý theo nội dung sản phẩm")
    
    # Đọc dữ liệu sản phẩm
    df_products = pd.read_csv(CONTENT_BASED_DATA_FILE)
    df_products['ten_san_pham'] = df_products['ten_san_pham'].astype(str).fillna('')
    df_products['ma_san_pham'] = df_products['ma_san_pham'].astype(str).fillna('')
    df_products['hinh_anh'] = df_products['hinh_anh'].astype(str).fillna(DEFAULT_IMAGE)

    # Nhập từ khóa tìm kiếm hoặc mã sản phẩm
    search_query = st.text_input("Nhập tên hoặc mã sản phẩm:", "").strip()

    # Lọc sản phẩm dựa trên từ khóa
    filtered_products = df_products[
        (df_products['ten_san_pham'].str.contains(search_query, case=False)) |
        (df_products['ma_san_pham'].str.contains(search_query, case=False))
    ] if search_query else pd.DataFrame()

    if search_query:
        if not filtered_products.empty:
            st.write(f"Đã tìm thấy {len(filtered_products)} sản phẩm phù hợp.")
        else:
            st.write("Không tìm thấy sản phẩm phù hợp.")

    # Hiển thị danh sách sản phẩm
    product_options = (
        [(row['ten_san_pham'], row['ma_san_pham']) for _, row in filtered_products.iterrows()]
        if not filtered_products.empty else
        [(row['ten_san_pham'], row['ma_san_pham']) for _, row in df_products.iterrows()]
    )
    selected_product = st.selectbox(
        "Chọn sản phẩm yêu thích:",
        options=product_options,
        format_func=lambda x: x[0]
    )

    if selected_product:
        product_id = selected_product[1]
        selected_product_data = df_products[df_products['ma_san_pham'] == product_id].iloc[0]
        
        st.write(f"### Sản phẩm bạn đã chọn: {selected_product[0]}")
        st.image(selected_product_data['hinh_anh'], use_column_width=True, caption=selected_product_data['ten_san_pham'])
        
        # Hiển thị thông tin sản phẩm
        gia_ban = selected_product_data.get('gia_ban', 0)
        gia_goc = selected_product_data.get('gia_goc', 0)
        mo_ta = selected_product_data.get('mo_ta', "Không có mô tả.")
        st.markdown(f"**Giá bán:** {gia_ban:,.0f} ₫")
        st.markdown(f"<span style='text-decoration: line-through; color: gray; font-size: 0.8em;'>Giá gốc: {gia_goc:,.0f} ₫</span>", unsafe_allow_html=True)
        st.write(f"**Điểm đánh giá trung bình:** {selected_product_data.get('diem_trung_binh', 'Không có thông tin')}")
        st.write(f"**Mô tả sản phẩm:** {mo_ta}")
        
        # Gợi ý sản phẩm
        recommendations = recommend_content_based(
            product_id=product_id,
            df=df_products,
            weight_content=0.7,
            weight_rating=0.3,
            top_n=6
        )

        st.write("### Sản phẩm gợi ý:")
        cols = st.columns(3)  # Hiển thị lưới 3 cột
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            col = cols[idx % 3]
            with col:
                st.image(row['hinh_anh'], use_column_width=True, caption=row['ten_san_pham'])
                gia_ban = row.get('gia_ban', 0)
                gia_goc = row.get('gia_goc', 0)
                mo_ta = row.get('mo_ta', "Không có mô tả.")
                st.markdown(f"**Giá bán:** {gia_ban:,.0f} ₫")
                st.markdown(f"<span style='text-decoration: line-through; color: gray; font-size: 0.8em;'>Giá gốc: {gia_goc:,.0f} ₫</span>", unsafe_allow_html=True)
                st.write(f"**Điểm đánh giá trung bình:** {row['average_rating']:.2f}")
                
                # Mô tả sản phẩm trong expander
                with st.expander("Xem mô tả sản phẩm"):
                    st.write(f"{mo_ta}")
                st.markdown("---")

# Tab 2: Collaborative Filtering
with tab2:
    st.subheader("Gợi ý theo người dùng")

    @st.cache_data
    def load_customer_data(data_files):
        full_data = pd.concat([pd.read_csv(file) for file in data_files])
        full_data['ma_khach_hang'] = full_data['ma_khach_hang'].astype(str).str.strip()
        return full_data['ma_khach_hang'].unique()

    customer_ids = load_customer_data([COLLABORATIVE_FULL_DATA_PART1, COLLABORATIVE_FULL_DATA_PART2])

    # Sử dụng session_state để lưu trạng thái tên và mã khách hàng
    if "customer_name" not in st.session_state:
        st.session_state.customer_name = ""
    if "customer_id" not in st.session_state:
        st.session_state.customer_id = ""

    # Nhập thông tin khách hàng
    customer_name = st.text_input(
        "Nhập tên của bạn:",
        value=st.session_state.customer_name
    ).strip()
    customer_id = st.selectbox(
        "Nhập hoặc chọn mã khách hàng của bạn:",
        options=[""] + list(customer_ids),
        format_func=lambda x: f"Mã khách hàng: {x}" if x else "",
        index=([""] + list(customer_ids)).index(st.session_state.customer_id) if st.session_state.customer_id in customer_ids else 0
    )

    # Nút đăng nhập
    if st.button("Đăng nhập"):
        st.session_state.customer_name = customer_name  # Lưu tên vào session_state
        st.session_state.customer_id = customer_id     # Lưu mã vào session_state

    # Chỉ hiển thị gợi ý sản phẩm khi đã có thông tin
    if st.session_state.customer_name and st.session_state.customer_id:
        try:
            # Thực hiện gợi ý sản phẩm
            recommendations = recommend_collaborative(
                [COLLABORATIVE_FULL_DATA_PART1, COLLABORATIVE_FULL_DATA_PART2],
                COLLABORATIVE_MODEL_FILE,
                st.session_state.customer_id,
                top_n=6
            )
            
            if not recommendations.empty:
                st.markdown(
                    f"### Các sản phẩm gợi ý dành riêng cho <span style='color:darkgreen; font-style:italic;'>{st.session_state.customer_name}</span>:",
                    unsafe_allow_html=True
                )
                
                # Hiển thị sản phẩm gợi ý
                cols = st.columns(3)  # Hiển thị lưới 3 cột
                for idx, (_, row) in enumerate(recommendations.iterrows()):
                    col = cols[idx % 3]
                    with col:
                        st.image(row['hinh_anh'], use_column_width=True, caption=row['ten_san_pham'])
                        gia_ban = row.get('gia_ban', 0)
                        gia_goc = row.get('gia_goc', 0)
                        mo_ta = row.get('mo_ta', "Không có mô tả.")
                        st.markdown(f"**Giá bán:** {gia_ban:,.0f} ₫")
                        st.markdown(f"<span style='text-decoration: line-through; color: gray; font-size: 0.8em;'>Giá gốc: {gia_goc:,.0f} ₫</span>", unsafe_allow_html=True)
                        st.write(f"**Điểm đánh giá trung bình:** {row['diem_trung_binh']:.2f}")
                        with st.expander("Xem mô tả sản phẩm"):
                            st.write(f"{mo_ta}")
                        st.markdown("---")
            else:
                st.warning("Không tìm thấy sản phẩm gợi ý phù hợp.")
        except Exception as e:
            st.error(f"Đã xảy ra lỗi: {e}")
    elif st.session_state.customer_name and not st.session_state.customer_id:
        st.warning("Vui lòng nhập hoặc chọn mã khách hàng để gợi ý sản phẩm!")
    elif not st.session_state.customer_name and st.session_state.customer_id:
        st.warning("Vui lòng nhập tên của bạn!")
