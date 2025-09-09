import streamlit as st
import pandas as pd
import joblib

# --- Load your trained model and encoder ---
model = joblib.load('./encoders/laptop_price_model.pkl')  # your trained model
encoder = joblib.load('./encoders/encoder.pkl')           # your fitted OneHotEncoder

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")
st.title("💻 Laptop Price Predictor")

# --- User Input ---

st.sidebar.header("Laptop Specifications")

# Use encoder categories for all categorical columns
company = st.sidebar.selectbox("Company", encoder.categories_[0])
product = st.sidebar.selectbox("Product Name", encoder.categories_[1])
type_name = st.sidebar.selectbox("Type", encoder.categories_[2])
opsys = st.sidebar.selectbox("Operating System", encoder.categories_[3])
cpu_brand = st.sidebar.selectbox("CPU Brand", encoder.categories_[4])
gpu_brand = st.sidebar.selectbox("GPU Brand", encoder.categories_[5])

# Numerical inputs remain unchanged
inches = st.sidebar.number_input("Screen Size (inches)", 10.0, 20.0, 13.3)
ram = st.sidebar.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
weight = st.sidebar.number_input("Weight (kg)", 0.5, 5.0, 1.5)
ssd = st.sidebar.number_input("SSD (GB)", 0, 2000, 256)
hdd = st.sidebar.number_input("HDD (GB)", 0, 2000, 0)
hybrid = st.sidebar.number_input("Hybrid Storage (GB)", 0, 2000, 0)
flash = st.sidebar.number_input("Flash Storage (GB)", 0, 1000, 0)
touchscreen = st.sidebar.selectbox("Touchscreen", [0,1])
x_res = st.sidebar.number_input("Screen X Resolution", 800, 4000, 1920)
y_res = st.sidebar.number_input("Screen Y Resolution", 600, 3000, 1080)

# --- Create input dataframe ---
input_dict = {
    'Company':[company],
    'Product':[product],
    'TypeName':[type_name],
    'Inches':[inches],
    'Ram':[ram],
    'OpSys':[opsys],
    'Weight':[weight],
    'SSD':[ssd],
    'HDD':[hdd],
    'Hybrid':[hybrid],
    'Flash_Storage':[flash],
    'Touchscreen':[touchscreen],
    'X_res':[x_res],
    'Y_res':[y_res],
    'Cpu_brand':[cpu_brand],
    'Gpu_brand':[gpu_brand]
}

input_df = pd.DataFrame(input_dict)

# --- Encode categorical columns ---
categorical_cols = ['Company', 'Product', 'TypeName', 'OpSys', 'Cpu_brand', 'Gpu_brand']
encoded_input = encoder.transform(input_df[categorical_cols])
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
encoded_df = pd.DataFrame(encoded_input, columns=encoded_col_names)

# --- Combine with numerical features ---
numerical_cols = ['Inches','Ram','Weight','SSD','HDD','Hybrid','Flash_Storage','Touchscreen','X_res','Y_res']
final_input = pd.concat([input_df[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# --- Predict ---
prediction = model.predict(final_input)[0]
st.subheader("💰 Predicted Laptop Price:")
st.write(f"€ {prediction:.2f}")
