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
    st.title("Prediksi Kelangsungan Hidup Penumpang Titanic 🚢")
st.write("Masukkan data penumpang di bawah ini:")

# Input fitur
pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3])
sex = st.selectbox("Jenis Kelamin", ["male", "female"])
age = st.slider("Usia", 0, 100, 25)
sibsp = st.number_input("Jumlah Saudara/Partner di kapal (SibSp)", min_value=0, step=1)
parch = st.number_input("Jumlah Orang Tua/Anak di kapal (Parch)", min_value=0, step=1)
fare = st.number_input("Tarif Tiket (Fare)", min_value=0.0, step=1.0)

# Encode jenis kelamin
sex_encoded = 1 if sex == "male" else 0

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare]],
                              columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    
    st.subheader("Data Input:")
    st.dataframe(input_data)

    hasil = model.predict(input_data)
    label = "Selamat" if hasil[0] == 1 else "Tidak Selamat"

    st.success(f"Hasil Prediksi: {label}")
# Page 3: About Naive Bayes
elif page == "About Naive Bayes":
    st.title("About Naive Bayes")
    st.write("""
        Naive Bayes is a classification technique based on Bayes' Theorem with an assumption of independence among predictors.
        It works well with large datasets and is particularly suited for text classification problems such as spam detection.
    """)

