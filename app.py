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
    page_icon="💄",
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

# Sidebar thông tin nhóm
st.sidebar.title("Thông tin nhóm thực hiện")
st.sidebar.write("""#### Thành viên thực hiện:
- Lê Quỳnh Minh Dung
- Nguyễn Thùy Trang""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Khuất Thùy Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:", 
    ["Introduction", "Recommendation Process", "Recommendation System"]
)

# Page 1: Introduction
if page == "Introduction":
    # Hiển thị banner
    st.title("Chào mừng đến với Hasaki.vn!")
    
    col1, col2 = st.columns([1, 3])  # Tạo hai cột, cột đầu nhỏ hơn để chứa logo

    # Cột 1: Hiển thị logo
    with col1:
        st.image("banner/Logo.png", use_column_width=False, width=200, caption="")

    # Cột 2: Hiển thị phần "Về Hasaki"
    with col2:
        st.subheader("Về Hasaki.vn 💄")
        st.write("""
            Hasaki.vn cam kết mang đến những sản phẩm làm đẹp và chăm sóc da tốt nhất cho khách hàng.
            Với trọng tâm là chất lượng và sự hài lòng của khách hàng, Hasaki hướng đến việc làm đẹp trở nên dễ tiếp cận với mọi người.
        """)

    st.subheader("🎯 Mục tiêu chính")
    st.write("""
        Hệ thống gợi ý (Recommend system) của chúng tôi được thiết kế nhằm:
        - 🛍️ Giúp khách hàng của chúng tôi khám phá các sản phẩm phù hợp với sở thích của mình.
        - 💡 Cải thiện trải nghiệm mua sắm bằng cách gợi ý các sản phẩm liên quan.
        - 📊 Ứng dụng các thuật toán tiên tiến như Content-Based Filtering và Collaborative Filtering để mang lại gợi ý cá nhân hóa.
    """)
    st.markdown("<br>", unsafe_allow_html=True)  # Thêm khoảng trống
    st.image("banner/images.png", use_column_width=True, caption="Mang đến trải nghiệm làm đẹp tuyệt vời")

# Page 2: Recommendation Process
elif page == "Recommendation Process":
    st.title("Quy trình xây dựng Recommendation System")
    st.write("""
        Quy trình xây dựng hệ thống gợi ý tại Hasaki.vn được chia thành hai phương pháp chính:
        1. **Content-Based Filtering**: Gợi ý sản phẩm dựa trên nội dung mô tả của sản phẩm.
        2. **Collaborative Filtering**: Gợi ý sản phẩm dựa trên hành vi của khách hàng khác.
    """)

    tab1, tab2, tab3 = st.tabs(["Crawl Data", "Content-Based Filtering", "Collaborative Filtering"])

    # Tab Crawl Data
    with tab1:
        st.subheader("Crawl Data từ Hasaki.vn")

        with st.expander("Code cào dữ liệu (Click để mở)"):
            st.code("""
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd
    import time

    # URL của trang web "Chăm sóc da mặt" trên Hasaki.vn
    BASE_URL = "https://hasaki.vn/danh-muc/cham-soc-da-mat-c4.html"

    # Hàm cào dữ liệu chính
    def scrape_hasaki_data(base_url, num_pages=5):
        products = []
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

        for page in range(1, num_pages + 1):
            url = f"{base_url}?p={page}"  # Cập nhật tham số trang
            print(f"Đang cào dữ liệu từ: {url}")
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    print(f"Không thể truy cập trang: {url} (status code: {response.status_code})")
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                product_items = soup.find_all('div', class_='item_sp_hasaki width_common relative')

                for item in product_items:
                    try:
                        # Lấy thông tin từ HTML
                        product_link_tag = item.find('a', class_='block_info_item_sp width_common card-body')
                        product_id = product_link_tag['data-id'] if product_link_tag and 'data-id' in product_link_tag.attrs else "Không có mã sản phẩm"
                        product_name = product_link_tag['data-name'] if product_link_tag and 'data-name' in product_link_tag.attrs else "Không có tên sản phẩm"
                        price = item.find('strong', class_='item_giamoi txt_16').text.strip().replace(" ₫", "").replace(".", "") if item.find('strong', class_='item_giamoi txt_16') else "Không có giá bán"
                        original_price = item.find('span', class_='item_giacu txt_12 right').text.strip().replace(" ₫", "").replace(".", "") if item.find('span', class_='item_giacu txt_12 right') else "Không có giá gốc"
                        image_url = item.find('img', class_='img_thumb lazy')['data-src'] if item.find('img', class_='img_thumb lazy') else "Không có hình ảnh"

                        # Thêm sản phẩm vào danh sách
                        products.append({
                            "ma_san_pham": product_id,
                            "ten_san_pham": product_name,
                            "gia_ban": price,
                            "gia_goc": original_price,
                            "hinh_anh": image_url,
                        })
                    except Exception as e:
                        print(f"Lỗi khi xử lý sản phẩm: {e}")

                time.sleep(2)  # Nghỉ 2 giây giữa các yêu cầu

            except Exception as e:
                print(f"Lỗi khi truy cập: {url}, {e}")
        
        return products

        # Gọi hàm cào dữ liệu
        data = scrape_hasaki_data(BASE_URL, num_pages=68)

        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame(data)

        # Loại bỏ sản phẩm trùng lặp dựa trên 'ma_san_pham'
        df = df.drop_duplicates(subset=['ma_san_pham'], keep='first')

        # Lưu thành file CSV
        df.to_csv('San_pham_new.csv', index=False, encoding='utf-8-sig')
        print("Dữ liệu đã được lưu vào file San_pham_new.csv!")
                """, language="python")

        st.write("""
            **Giải thích:**
            - **Mục đích:** Cào dữ liệu sản phẩm từ danh mục "Chăm sóc da mặt" trên trang Hasaki.vn.
            - **Chi tiết:** 
                1. **`BeautifulSoup`**: Thư viện để phân tích cú pháp HTML.
                2. **Dữ liệu cào được:**
                    - **`ma_san_pham`**: Mã định danh sản phẩm.
                    - **`ten_san_pham`**: Tên sản phẩm.
                    - **`gia_ban`**: Giá hiện tại của sản phẩm.
                    - **`gia_goc`**: Giá gốc (nếu có).
                    - **`hinh_anh`**: URL hình ảnh sản phẩm.
                3. **Loại bỏ trùng lặp:** Sản phẩm được lọc dựa trên `ma_san_pham`.
                4. **Đầu ra:** Lưu dữ liệu đã xử lý vào file San_pham_new.csv để merge chung với file gốc là San_pham.csv dựa trên cột `ma_san_pham`.
                5. **Thời gian nghỉ (sleep):** Giữa mỗi lần cào một trang, chương trình nghỉ 2 giây để tránh bị chặn.
        """)

        st.write("### Dữ liệu mẫu cào được:")
        # Hiển thị dữ liệu mẫu
        try:
            df_sample = pd.read_csv('data/San_pham_new.csv').head()
            st.dataframe(df_sample)
        except FileNotFoundError:
            st.warning("Không tìm thấy file `San_pham_new.csv`. Vui lòng chạy mã cào dữ liệu trước.")


    # Tab Content-Based Filtering
    with tab2:
        # Streamlit layout
        st.title("Content-Based Filtering: Preprocessing and Analysis")

        st.write("### Các bước tiền xử lý:")
        st.markdown("""
        1. **Đọc dữ liệu từ các tệp CSV:**
            - **San_pham.csv:** Chứa thông tin sản phẩm (tên, mô tả, giá, điểm đánh giá...).
            - **Danh_gia.csv:** Chứa đánh giá của khách hàng.
        2. **Loại bỏ stopwords và làm sạch dữ liệu (Tonkenize).**
        3. **Tính toán đặc trưng:**
            - Tạo nội dung phong phú hơn bằng cách kết hợp tên, mô tả sản phẩm và phân loại để tạo nội dung phong phú hơn.
            - Tính toán điểm tổng hợp cho từng sản phẩm bằng cách kết hợp điểm tương tự nội dung (similarity score) và điểm đánh giá trung bình (average rating). Dùng điểm tổng hợp này để xác định các sản phẩm tương tự nhất nhưng vẫn đảm bảo ưu tiên các sản phẩm có đánh giá tốt hơn.
        """)

        # Load dữ liệu
        san_pham_path = "data/san_pham_updated.csv"
        san_pham_preprocessed_path = "data/content_based_preprocessed.csv"
        san_pham_df = pd.read_csv(san_pham_path)
        san_pham_preprocessed_df = pd.read_csv(san_pham_preprocessed_path)

        # Display raw data
        st.write("### Dữ liệu sản phẩm gốc:")
        st.dataframe(san_pham_df[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh']].head())

        # Display tokenized data
        st.write("### Dữ liệu sau khi tiền xử lý:")
        st.dataframe(san_pham_preprocessed_df[['ma_san_pham', 'processed_content', 'tokens', 'token_count']].head())

        # Token distribution
        st.write("### Phân phối số lượng từ trong mô tả sản phẩm:")
        st.image("banner/distribution.png", use_column_width=True, caption="")

        st.markdown("""
        **Nhận xét:**
        - Độ dài mô tả tập trung trong khoảng 100-150 từ, cho thấy mô tả sản phẩm ở mức độ vừa phải, không quá dài hoặc ngắn.
        - Một số sản phẩm có mô tả rất ngắn (<100 tokens) hoặc rất dài (>300 tokens), có thể cần được chuẩn hóa.
        - Phân phối đều, không có sự lệch đáng kể.
        """)

        # Relationship between token count and average rating
        st.write("### Quan hệ giữa điểm đánh giá và độ dài mô tả:")
        st.image("banner/relationship.png", use_column_width=True, caption="")

        st.markdown("""
        **Nhận xét:**
        - Không có mối quan hệ tuyến tính rõ ràng giữa độ dài mô tả (token_count) và điểm đánh giá trung bình (diem_trung_binh).
        - Các sản phẩm có điểm cao (4-5) xuất hiện ở nhiều mức token, từ ngắn đến dài.
        - Các sản phẩm có điểm bằng 0 trải rộng ở mọi độ dài mô tả, cần kiểm tra lại dữ liệu.
        """)

        # Correlation heatmap
        st.write("### Heatmap tương quan:")
        st.image("banner/heatmap.png", use_column_width=True, caption="")

        st.markdown("""
        **Nhận xét:**
        - Tương quan giữa diem_trung_binh và token_count là -0.02 (gần 0), cho thấy độ dài mô tả không ảnh hưởng đến điểm đánh giá.
        - Cần phân tích thêm các yếu tố khác (giá cả, loại sản phẩm, hình ảnh) để tìm mối quan hệ có ý nghĩa hơn.
        """)

        # Top 10 most frequent words
        st.write("### Top 10 từ phổ biến nhất trong mô tả sản phẩm:")
        st.image("banner/frequent_words.png", use_column_width=True, caption="")

        st.markdown("""
        **Nhận xét:**
        - "hàng", "đơn", "da", "hóa": Đây là các từ phổ biến nhưng mang tính tổng quát và thường không cung cấp nhiều thông tin đặc trưng cho sản phẩm.
        - "đỏ", "hasaki": Các từ này có thể là đặc trưng của sản phẩm (màu sắc hoặc thương hiệu) nhưng cần kiểm tra xem những từ này có xuất hiện quá thường xuyên và có giá trị phân tích hay không.
        - "không", "xuất", "h": Đây là những từ có khả năng không mang lại ý nghĩa đặc biệt cho mô tả sản phẩm, đặc biệt từ như "h" hoặc "không" có thể bị xem là stopwords.
        """)

        # Tab Collaborative Filtering
        with tab3:
            st.subheader("Collaborative Filtering")
            st.write("### Bước 1: Xử lý dữ liệu khách hàng và sản phẩm")
            st.code("""
            import pandas as pd

            def preprocess_data(input_file, output_file):
                # Code xử lý dữ liệu Collaborative Filtering
                pass
                    """, language="python")
            st.write("### Bước 2: Huấn luyện mô hình Collaborative Filtering")
            st.code("""
            from surprise import Dataset, KNNBaseline

            def train_model(data_file, model_file):
                # Code huấn luyện Collaborative Filtering
                pass
                    """, language="python")

# Page 3: Recommendation System
elif page == "Recommendation System":
    # Hiển thị banner
    banner_path = "banner/hasaki_banner.png"  # Đường dẫn cục bộ

    st.image(
        banner_path,
        use_column_width=True,  # Tự động căn chỉnh theo chiều rộng của trang
        caption=None  # Không hiển thị chú thích
    )
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
                top_n=7
            )
            # Loại bỏ sản phẩm đã chọn khỏi danh sách gợi ý
            recommendations = recommendations[recommendations['ma_san_pham'] != selected_product_data['ma_san_pham']]

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