# Customer-Segmentation-and-Card-Product-Recommendation-Project
# 💳 Credit Card Customer Segmentation using Machine Learning

Một dự án Khoa học Dữ liệu (Data Science) nhằm phân khúc danh mục khách hàng sử dụng thẻ tín dụng, từ đó kiểm định các giả thuyết kinh doanh và đề xuất chiến lược tối ưu hóa lợi nhuận, quản trị rủi ro cho Ngân hàng.

---

## 📌 Tổng Quan Dự Án (Project Overview)
Dự án này áp dụng thuật toán học máy không giám sát (**Unsupervised Learning - K-Means Clustering**) để phân tích hành vi tài chính của tập khách hàng sử dụng thẻ tín dụng. Qua đó, mô hình đã bóc tách danh mục thành **4 cụm khách hàng** riêng biệt, giúp doanh nghiệp thấu hiểu sâu sắc chân dung người dùng để đưa ra các quyết định Marketing và Quản trị rủi ro chính xác.

---

## 📊 3 Giả Thuyết Kinh Doanh & Kết Quả Kiểm Định (Hypotheses & Validation)

### 🧪 Giả Thuyết 1: Nhóm Khách Hàng Cao Cấp (The VIP Power Spenders)
*   **Giả thuyết:** Khách hàng có dư nợ (`BALANCE`) và tỷ lệ tất toán (`PRC_FULL_PAYMENT`) cao (>80%) là nhóm thu nhập cao, chi tiêu lớn, sử dụng thẻ để tận dụng dòng tiền chứ không chịu trả lãi. Họ đóng đóng góp phần lớn phí giao dịch (`PURCHASES`) cho ngân hàng.
*   **Kết quả (XÁC NHẬN):** Phân tích từ mô hình cho thấy nhóm này kéo một dải chi tiêu vượt trội từ $10,000$ đến gần $50,000$, áp đảo hoàn toàn danh mục. Một lượng lớn khách hàng tập trung sát đáy trục Y (Dư nợ thấp, tất toán hết cuối kỳ), chứng minh họ quẹt thẻ liên tục để ăn điểm thưởng/ưu đãi chứ không để tích nợ.

### ⚡ Giả Thuyết 2: Nhóm Rút Tiền Mặt (The Inactive / Cash Borrowers)
*   **Giả thuyết:** Khách hàng có tần suất rút tiền mặt (`CASH_ADVANCE_FREQUENCY`) cao thường có tỷ lệ tất toán thấp và nghĩa vụ trả tối thiểu (`MINIMUM_PAYMENTS`) cao. Đây là nhóm rủi ro cao (Subprime) nhưng mang lại biên lợi nhuận lãi vay lớn.
*   **Kết quả (XÁC NHẬN):** Biểu đồ chỉ ra nhóm này có hành vi bám chặt mốc 0 của trục chi tiêu mua sắm, thẻ chỉ dùng để rút tiền mặt. Tỷ lệ tất toán nợ của nhóm này thấp kỷ lục (chỉ ~3%), đồng nghĩa 97% dư nợ được gối đầu qua các kỳ, mang lại nguồn thu lãi vay khổng lồ nhưng tiềm ẩn nguy cơ nợ xấu đóng băng.

### 🛒 Giả Thuyết 3: Nhóm Mua Trả Góp (The Balanced Installment Users)
*   **Giả thuyết:** Khách hàng tích cực mua trả góp (`INSTALLMENTS_PURCHASES`) sẽ có lòng trung thành (`TENURE`) cao hơn và tần suất tương tác với thẻ ổn định hơn.
*   **Kết quả (CONFIRMED):** Do đặc tính dữ liệu phần lớn khách hàng đều đạt `TENURE` tối đa 12 tháng, lòng trung thành được chứng minh qua khía cạnh duy trì tương tác (Engagement). Nhóm trả góp đạt tần suất quẹt thẻ lên tới **67.6%**, vượt trội hơn nhóm chi tiêu tiết kiệm (55.4%) và nhóm rút mặt (2.4%). Các khoản trả góp đóng vai trò như một "mỏ neo" tâm lý duy trì thói quen dùng thẻ của khách hàng.

---

## 🛠️ Công Cụ & Thư Viện Sử Dụng (Tech Stack)
*   **Ngôn ngữ:** Python
*   **Phân tích dữ liệu:** Pandas, NumPy
*   **Trực quan hóa:** Matplotlib, Seaborn
*   **Học máy:** Scikit-learn (K-Means Clustering, StandardScaler, PCA)

---

## 📁 Cấu Trúc Thư Mục Dự Án (Project Structure)
```text
├── data/
│   └── credit_card_data.csv       # Tập dữ liệu gốc chứa hành vi khách hàng
├── notebooks/
│   └── segmentation_analysis.ipynb # File Jupyter Notebook chứa toàn bộ mã nguồn
├── README.md                      # Tệp tổng quan dự án (File này)
└── requirements.txt               # Danh sách các thư viện cần cài đặt
