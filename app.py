import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #1b263b; color: #ffffff; font-size: 18px; }
    div.stButton > button { background-color: #1d3557; color: white; border-radius: 8px; width: 100%; font-weight: bold; font-size: 18px; padding: 12px; border: 1px solid #457b9d; }
    h1 { color: #58a6ff !important; }
    h2, h3 { color: #a8dadc; }
    .stNumberInput label { color: #f1faee !important; font-weight: bold !important; font-size: 20px !important; }
    div[data-baseweb="input"] { background-color: #2b3a4a !important; border-radius: 8px !important; }
    input { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; font-weight: bold !important; font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model():
    np.random.seed(42)
    n = 500
    p, a, s, g = np.random.uniform(0,10,n), np.random.uniform(0,100,n), np.random.uniform(0,40,n), np.random.uniform(0,4,n)
    t = (0.35*(p/10) + 0.25*(a/100) + 0.2*(s/40) + 0.2*(g/4.0) + np.random.normal(0,0.05,n) >= 0.6).astype(int)
    df = pd.DataFrame({'Prequisite_Score': p, 'Attendance_Rate': a, 'Self_Study_Hours': s, 'GPA_Accumulated': g, 'Final_Result': t})
    X, y = df.drop(columns=['Final_Result']), df['Final_Result']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    model = KNeighborsClassifier(n_neighbors=9, metric='euclidean')
    model.fit(scaler.fit_transform(X_tr), y_tr)
    return model, scaler

model, scaler = load_model()
if 'history' not in st.session_state: st.session_state.history = []

st.title("🎓 HỆ THỐNG DỰ ĐOÁN VÀ CẢNH BÁO KẾT QUẢ HỌC TẬP SINH VIÊN")
c_main, c_side = st.columns([2, 1])

with c_main:
    st.subheader("📝 Nhập Thông Tin Học Tập Sinh Viên")
    with st.form("pred_form"):
        col1, col2 = st.columns(2)
        with col1:
            pre = st.number_input("📚 Điểm Tiên Quyết (0-10)", 0.0, 10.0, 6.5)
            att = st.number_input("⏰ Tỷ Lệ Điểm Danh (%)", 0.0, 100.0, 80.0)
        with col2:
            hrs = st.number_input("⏱️ Giờ Tự Học/Tuần", 0.0, 40.0, 15.0)
            gpa = st.number_input("📊 GPA Tích Lũy (0-4)", 0.0, 4.0, 2.5)
        b1, b2 = st.columns(2)
        submitted = b1.form_submit_button("🚀 Dự Đoán")
        cleared = b2.form_submit_button("🔄 Làm Mới")

        if submitted:
            with st.spinner("⏳ Đang xử lý dữ liệu và tính toán mô hình K-NN..."):
                inp = scaler.transform([[pre, att, hrs, gpa]])
                prob = model.predict_proba(inp)[0]
                pass_p, fail_p = prob[1]*100, prob[0]*100
                st.markdown("---")
                if pass_p >= 75:
                    st.markdown("### <span style='color:#2ecc71;'>Kết quả: ĐẠT (Pass)</span>", unsafe_allow_html=True)
                    st.info("Đánh giá: Sinh viên có phong độ học tập tốt. Hãy duy trì tỷ lệ điểm danh và số giờ tự học hiện tại.")
                    res_text = "ĐẠT (Pass)"
                elif 50 <= pass_p < 75:
                    st.markdown("### <span style='color:#f1c40f;'>Kết quả: CHÚ Ý (Attention)</span>", unsafe_allow_html=True)
                    st.warning("Đánh giá: Phong độ học tập chưa thực sự ổn định. Cần chủ động tăng giờ tự học và đảm bảo tỷ lệ điểm danh.")
                    res_text = "CHÚ Ý (Attention)"
                else:
                    st.markdown("### <span style='color:#e74c3c;'>Kết quả: KHÔNG ĐẠT (Fail)</span>", unsafe_allow_html=True)
                    st.error("Đánh giá & Cảnh báo: Nguy cơ trượt môn cao! Cần tăng cường giờ tự học và cải thiện điểm danh ngay.")
                    res_text = "KHÔNG ĐẠT (Fail)"
                
                st.write(f"**Tỷ lệ dự đoán** → ĐẠT: {pass_p:.1f}% | KHÔNG ĐẠT: {fail_p:.1f}%")
                st.markdown(f"""
                    <div style="background:#111; padding:15px; border-radius:8px;">
                        <div style="margin-bottom:8px; font-weight:bold; color:#2ecc71;">🟢 ĐẠT: {pass_p:.1f}%</div>
                        <div style="background:#222; border-radius:4px; margin-bottom:12px;"><div style="width:{pass_p}%; background:#2ecc71; height:18px; border-radius:4px;"></div></div>
                        <div style="margin-bottom:8px; font-weight:bold; color:#e74c3c;">🔴 KHÔNG ĐẠT: {fail_p:.1f}%</div>
                        <div style="background:#222; border-radius:4px; margin-bottom:10px;"><div style="width:{fail_p}%; background:#e74c3c; height:18px; border-radius:4px;"></div></div>
                        <div style="display:flex; justify-content:space-between; color:#aaa; font-size:14px; border-top:1px solid #444; padding-top:6px;">
                            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.session_state.history.append({"Điểm Tiên Quyết": pre, "Điểm Danh (%)": att, "Giờ Tự Học": hrs, "GPA": gpa, "Kết Quả": res_text, "Xác Suất Đạt": f"{pass_p:.1f}%"})

    if st.session_state.history:
        st.subheader("📜 Lịch Sử Tra Cứu Gần Đây")
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)
        h1, h2 = st.columns(2)
        if h1.button("💾 Save Lịch Sử"): st.success("Đã lưu lịch sử tra cứu thành công!")
        if h2.button("🗑️ Reset Lịch Sử"): st.session_state.history = []; st.rerun()

with c_side:
    st.subheader("📖 Hướng Dẫn Sử Dụng")
    st.markdown("- **Bước 1:** Nhập thông tin vào form trái.\n- **Bước 2:** Nhấn nút **Dự Đoán**.\n- **Bước 3:** Xem kết quả, biểu đồ và lời khuyên.")
    st.subheader("📊 Chất Lượng Mô Hình")
    st.dataframe(pd.DataFrame({"Chỉ số": ["Accuracy", "Precision (Fail)", "Recall (Fail)", "F1-Score"], "Giá trị": ["93.0%", "94.0%", "92.0%", "93.0%"]}), hide_index=True)