import streamlit as st
import pandas as pd
from collaborative_recommend import recommend_products as recommend_collaborative
from content_based_recommendation import recommend_products as recommend_content_based

# Đường dẫn tệp
CONTENT_BASED_DATA_FILE = "data/content_based_preprocessed.csv"
COLLABORATIVE_FULL_DATA_PART1 = "data/collaborative_full_data_part1.csv"
COLLABORATIVE_FULL_DATA_PART2 = "data/collaborative_full_data_part2.csv"
COLLABORATIVE_MODEL_FILE = "model/collaborative_model.pkl.gz"

# Set page configuration
st.set_page_config(
    page_title="Hasaki Recommendation System",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Hasaki-themed design
st.markdown("""
    <style>
    /* Overall page background */
    body {
        background-color: #f0f4f0;
        color: #2f6e51;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #e6f2e8;
        border-right: 2px solid #2f6e51;
    }

    /* Header and title styles */
    .header-title {
        color: #2f6e51;
        font-family: 'Arial', sans-serif;
        text-align: center;
        padding: 20px;
        background-color: #d0e5d3;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    /* Banner Styling */
    .banner {
        background: url('banner/hasaki_banner_2.jpg') no-repeat center center;
        background-size: cover;
        height: 150px; /* Adjust height as needed */
        display: flex;
        justify-content: center;
        align-items: center;
        color: white;
        font-size: 32px;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }

    /* Layout for product display */
    .product-layout {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-top: 20px;
    }

    /* Style for product image */
    .product-image img {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Hiển thị banner
banner_path = "banner/hasaki_banner.png"  # Đường dẫn cục bộ

st.image(
    banner_path,
    use_column_width=True,  # Tự động căn chỉnh theo chiều rộng của trang
    caption=None  # Không hiển thị chú thích
)

# Sidebar thông tin nhóm
st.sidebar.title("Thông tin nhóm thực hiện")
st.sidebar.write("""#### Thành viên thực hiện:
- Lê Quỳnh Minh Dung
- Nguyễn Thùy Trang""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")

# Hiển thị tiêu đề chính
st.title("Hasaki gợi ý sản phẩm cho bạn")

# Tabs để chọn giữa hai phương pháp gợi ý
tab1, tab2 = st.tabs(["Content-Based Filtering", "Collaborative Filtering"])

# Tab 1: Content-Based Filtering
with tab1:
    st.subheader("Content-based filtering")

    # Đọc dữ liệu sản phẩm
    df_products = pd.read_csv(CONTENT_BASED_DATA_FILE)
    df_products['ten_san_pham'] = df_products['ten_san_pham'].astype(str).fillna('')
    df_products['ma_san_pham'] = df_products['ma_san_pham'].astype(str).fillna('')
    df_products['hinh_anh'] = df_products['hinh_anh'].astype(str).fillna("https://via.placeholder.com/150")

    # Chọn tên sản phẩm
    product_names = df_products['ten_san_pham'].unique()
    selected_product_name = st.selectbox(
        "Nhập hoặc chọn tên sản phẩm yêu thích:",
        options=[""] + list(product_names),
        format_func=lambda x: x if x else "Chọn sản phẩm",
        key="product_name"
    )

    # Lọc sản phẩm dựa trên lựa chọn
    filtered_products = df_products[df_products['ten_san_pham'] == selected_product_name] if selected_product_name else pd.DataFrame()

    # Hiển thị sản phẩm đã chọn
    if not filtered_products.empty:
        selected_product_data = filtered_products.iloc[0]
        st.markdown("---")
        st.markdown(f"<h2 style='color: #2E8B57; text-align: center;'>{selected_product_data['ten_san_pham']}</h2>", unsafe_allow_html=True)
        
        # Tạo bố cục hai cột
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="product-image">', unsafe_allow_html=True)
            st.image(selected_product_data['hinh_anh'], use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            gia_ban = selected_product_data.get('gia_ban', 0)
            gia_goc = selected_product_data.get('gia_goc', 0)
            tab_info, tab_desc = st.tabs(["Thông tin chi tiết", "Mô tả sản phẩm"])
            with tab_info:
                st.markdown(f"**Mã sản phẩm:** {selected_product_data['ma_san_pham']}")
                st.markdown(f"**Giá bán:** <span style='color: red;'>{gia_ban:,.0f} ₫</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='text-decoration: line-through; color: gray;'>Giá gốc: {gia_goc:,.0f} ₫</span>", unsafe_allow_html=True)
                st.markdown(f"**Điểm đánh giá:** {selected_product_data.get('diem_trung_binh', 'Không có thông tin')}")
            with tab_desc:
                st.markdown(selected_product_data.get('mo_ta', "Không có mô tả."))

        # Gợi ý sản phẩm
        recommendations = recommend_content_based(
            product_id=selected_product_data['ma_san_pham'],
            df=df_products,
            weight_content=0.7,
            weight_rating=0.3,
            top_n=6
        )

        st.write("### Sản phẩm gợi ý:")
        cols = st.columns(3)
        for idx, (_, row) in enumerate(recommendations.iterrows()):
            col = cols[idx % 3]
            with col:
                st.image(
                    row['hinh_anh'], 
                    caption=row['ten_san_pham'], 
                    use_column_width=False,  # Tắt tự động căn chỉnh
                    width=350  # Đặt chiều rộng cố định
                )
                st.markdown(f"**Mã sản phẩm:** <span style='color: blue;'>{row.get('ma_san_pham', 'Không có thông tin')}</span>", unsafe_allow_html=True)
                st.markdown(f"**Giá bán:** <span style='color: red;'>{row.get('gia_ban', 'Không có thông tin')} ₫</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='text-decoration: line-through; color: gray;'>Giá gốc: {row.get('gia_goc', 'Không có thông tin')} ₫</span>", unsafe_allow_html=True)
                st.markdown(f"**Điểm đánh giá:** {row.get('average_rating', 'Không có thông tin')}")
                with st.expander("Xem mô tả sản phẩm"):
                    st.write(row.get('mo_ta', "Không có mô tả."))
                st.markdown("---")

# Tab 2: Collaborative Filtering
with tab2:
    st.subheader("Collaborative filtering")

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