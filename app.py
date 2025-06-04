import streamlit as st
import pandas as pd
import joblib

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Data Description", "Prediction", "About Naive Bayes"])
# Page 1: Data description
if page == "Data Description":
    st.title("Data Description")
    
    # Input for name
    name = st.text_input("Enter your name:", "")
    
    # Slider for age
    age = st.slider("Select your age:", min_value=0, max_value=100, value=25)
    
    if name:
        st.write(f"Hello, **{name}**! You are **{age}** years old.")
    else:
        st.write("Please enter your name.")

# Page 2: Prediction
elif page == "Prediction":
    model = joblib.load('naive_bayes_model.pkl')
    st.title("Prediksi Jenis Bunga Iris 🌸")
    st.write("Masukkan data yang akan diprediksi:")

    # Input fitur dari pengguna
    seplen = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=2.0)
    sepwid = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=2.0)
    petlen = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=2.0)
    petwid = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=2.0)

    # Prediksi saat tombol ditekan
    if st.button("Prediksi"):
    input_data = pd.DataFrame([[seplen, sepwid, petlen, petwid]],
                              columns=["sepal.length", "sepal.width", "petal.length", "petal.width"])
    st.dataframe(input_data)

    hasil = model.predict(input_data)
    st.success(f"Hasil Prediksi: {hasil[0]}")

# Page 3: About Naive Bayes
elif page == "About Naive Bayes":
    st.title("About Naive Bayes")
    st.write("""
        Naive Bayes is a classification technique based on Bayes' Theorem with an assumption of independence among predictors.
        It works well with large datasets and is particularly suited for text classification problems such as spam detection.
    """)

