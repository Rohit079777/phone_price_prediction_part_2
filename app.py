# app.py
import os
import pickle as p
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# ---------- Page setup ----------
st.set_page_config(page_title="ML-Based Smartphone Price Prediction", page_icon="📱", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
h1, h2, h3 {margin-bottom: .4rem;}
.small-note {font-size: 0.85rem; color: #666;}
.card {padding: 0.75rem 1rem; border: 1px solid #eee; border-radius: 10px; background: #fafafa;}
</style>
""", unsafe_allow_html=True)

# ---------- Load model & data ----------
with open("used_phone.pkl", "rb") as f:
    model = p.load(f)

df = pd.read_csv("used_phone.csv")

# ---------- Encoders ----------
le_brand = LabelEncoder().fit(df["brand"])
le_model = LabelEncoder().fit(df["model"])
le_condition = LabelEncoder().fit(df["condition"])

# ---------- Brand → models map ----------
brand_model_map = df.groupby("brand")["model"].unique().to_dict()
brand_list = sorted(list(brand_model_map.keys()))

# ---------- Sidebar Navigation ----------
page = st.sidebar.radio("Navigation", ["Home", "Compare Mobiles"])

# ---------------------- HOME PAGE ----------------------
if page == "Home":
    st.title("📱 ML-Based Smartphone Price Prediction")

    # ---------- Sidebar Inputs ----------
    st.sidebar.header("Enter Your Phone Details")
    selected_brand = st.sidebar.selectbox("Select Brand", brand_list)
    selected_model = st.sidebar.selectbox("Select Model", sorted(brand_model_map[selected_brand]))
    ram = st.sidebar.selectbox("RAM (GB)", sorted(df["ram_gb"].dropna().unique()))
    storage = st.sidebar.selectbox("Storage (GB)", sorted(df["storage_gb"].dropna().unique()))
    condition = st.sidebar.selectbox("Condition", sorted(df["condition"].dropna().unique()))
    battery = st.sidebar.slider("Battery Health (%)", 50, 100, 80)
    age = st.sidebar.slider("Age of Phone (Years)", 0, 5, 1)
    original_price = st.sidebar.number_input("Original Price (INR)", 1000, 200000, 15000, step=500)

    # ---------- Encode selection ----------
    brand_encoded = int(le_brand.transform([selected_brand])[0])
    model_encoded = int(le_model.transform([selected_model])[0])
    condition_encoded = int(le_condition.transform([condition])[0])

    # ---------- Prepare model input ----------
    input_row = pd.DataFrame({
        "brand": [brand_encoded],
        "model": [model_encoded],
        "ram_gb": [ram],
        "storage_gb": [storage],
        "condition": [condition_encoded],
        "battery_health": [battery],
        "age_years": [age],
        "original_price": [original_price],
    })

    # ---------- Main content ----------
    c_left, c_right = st.columns([1.1, 1.4], vertical_alignment="top")

    with c_left:
        st.subheader("Result")
        if st.sidebar.button("Predict", use_container_width=True):
            try:
                predicted_price = float(model.predict(input_row)[0])
                p_int = int(predicted_price)
                lo = int(predicted_price * 0.95)
                hi = int(predicted_price * 1.05)

                st.success(f"Estimated Used Phone Price : **₹{p_int:,}**")
                st.info(f"Suggested range (±5%): **₹{lo:,} – ₹{hi:,}**")

                # Quick metrics
                colm1, colm2, colm3 = st.columns(3)
                colm1.metric("Battery", f"{battery} %")
                colm2.metric("Age", f"{age} yr")
                drop_amt = max(original_price - p_int, 0)
                colm3.metric("Drop from Original", f"₹{drop_amt:,}")

                # Price comparison chart
                fig1, ax1 = plt.subplots(figsize=(4, 2.6))
                ax1.bar(["Original", "Predicted"], [original_price, p_int])
                ax1.set_ylabel("INR")
                ax1.set_title("Original vs Predicted")
                st.pyplot(fig1)

                # Depreciation curve
                years = np.arange(0, 6)
                curve = [original_price * (0.85 ** y) for y in years]  # assume 15%/year
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                ax2.plot(years, curve, marker="o")
                ax2.set_xlabel("Age (years)")
                ax2.set_ylabel("Estimated Value (INR)")
                ax2.set_title("Depreciation Over Time")
                st.pyplot(fig2)

                # Recommendation
                st.markdown("#### Recommendation")
                if p_int >= original_price * 0.6:
                    st.success("🚀 **Good resale value** — consider selling soon.")
                elif p_int >= original_price * 0.35:
                    st.info("📊 **Average value** — negotiate a bit or wait for a better market.")
                else:
                    st.warning("🧿 **Low resale value** — better keep using for now.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with c_right:
        st.subheader(f"{selected_brand} {selected_model}")
        st.image("phone_images/default_phone.jpg", width=200, caption="Default Phone Image")
        with st.container(border=True):
            st.write(f"• RAM: **{ram} GB**")
            st.write(f"• Storage: **{storage} GB**")
            st.write(f"• Condition: **{condition}**")
            st.write(f"• Battery Health: **{battery} %**")
            st.write(f"• Age: **{age} years**")
            st.write(f"• Original Price: **₹{original_price:,}**")

# ---------------------- COMPARE MOBILES PAGE ----------------------
elif page == "Compare Mobiles":
    st.title("📊 Compare Mobile Prices")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mobile 1")
        brand1 = st.selectbox("Brand", brand_list, key="brand1")
        model1 = st.selectbox("Model", sorted(brand_model_map[brand1]), key="model1")
        ram1 = st.selectbox("RAM (GB)", sorted(df["ram_gb"].unique()), key="ram1")
        storage1 = st.selectbox("Storage (GB)", sorted(df["storage_gb"].unique()), key="storage1")
        condition1 = st.selectbox("Condition", sorted(df["condition"].unique()), key="cond1")
        battery1 = st.slider("Battery Health (%)", 50, 100, 80, key="bat1")
        age1 = st.slider("Age (Years)", 0, 5, 1, key="age1")
        original_price1 = st.number_input("Original Price (INR)", 1000, 200000, 15000, step=500, key="op1")

    with col2:
        st.subheader("Mobile 2")
        brand2 = st.selectbox("Brand", brand_list, key="brand2")
        model2 = st.selectbox("Model", sorted(brand_model_map[brand2]), key="model2")
        ram2 = st.selectbox("RAM (GB)", sorted(df["ram_gb"].unique()), key="ram2")
        storage2 = st.selectbox("Storage (GB)", sorted(df["storage_gb"].unique()), key="storage2")
        condition2 = st.selectbox("Condition", sorted(df["condition"].unique()), key="cond2")
        battery2 = st.slider("Battery Health (%)", 50, 100, 80, key="bat2")
        age2 = st.slider("Age (Years)", 0, 5, 1, key="age2")
        original_price2 = st.number_input("Original Price (INR)", 1000, 200000, 15000, step=500, key="op2")

    if st.button("Compare Prices"):
        try:
            row1 = pd.DataFrame({
                "brand": [le_brand.transform([brand1])[0]],
                "model": [le_model.transform([model1])[0]],
                "ram_gb": [ram1],
                "storage_gb": [storage1],
                "condition": [le_condition.transform([condition1])[0]],
                "battery_health": [battery1],
                "age_years": [age1],
                "original_price": [original_price1],
            })
            row2 = pd.DataFrame({
                "brand": [le_brand.transform([brand2])[0]],
                "model": [le_model.transform([model2])[0]],
                "ram_gb": [ram2],
                "storage_gb": [storage2],
                "condition": [le_condition.transform([condition2])[0]],
                "battery_health": [battery2],
                "age_years": [age2],
                "original_price": [original_price2],
            })

            price1 = int(model.predict(row1)[0])
            price2 = int(model.predict(row2)[0])

            comp_df = pd.DataFrame({
                "Mobile": ["Mobile 1", "Mobile 2"],
                "Brand": [brand1, brand2],
                "Model": [model1, model2],
                "Predicted Price (₹)": [price1, price2]
            })

            st.write("### 📊 Price Comparison")
            st.table(comp_df)

        except Exception as e:
            st.error(f"Comparison failed: {e}")

# ---------- Footer ----------
st.markdown('<p class="small-note">💡 Tip: Compare mobiles to find the best resale value.</p>', unsafe_allow_html=True)
