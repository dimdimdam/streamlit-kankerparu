import streamlit as st
import pandas as pd
import joblib

# Load model dan fitur
model = joblib.load("model_lung_cancer.pkl")
feature_order = joblib.load("feature_columns.pkl")

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Kanker Paru-paru",
    layout="centered",  # Bisa diganti ke "wide" jika butuh layar penuh
    initial_sidebar_state="auto"
)

# Judul Aplikasi
st.title("🫁 Prediksi Penyakit Kanker Paru-paru")
st.markdown("Silakan isi data berikut untuk melakukan prediksi kemungkinan risiko kanker paru-paru.")


# Fungsi bantu konversi nilai
def encode_binary(val):
    return 1 if val == "Ya" else 0


# Input Usia
AGE = st.number_input("Usia (dalam tahun)", min_value=20, max_value=100, value=50)

# Daftar pertanyaan biner
questions = {
    "GENDER": "Apakah Anda seorang laki-laki?",
    "SMOKING": "Apakah Anda merokok? (aktif maupun vape)",
    "FINGER_DISCOLORATION": "Apakah ujung jari Anda terlihat kekuningan?",
    "EXPOSURE_TO_POLLUTION": "Apakah Anda sering terpapar polusi udara?",
    "LONG_TERM_ILLNESS": "Apakah Anda memiliki riwayat penyakit kronis?",
    "IMMUNE_WEAKNESS": "Apakah Anda merasa memiliki daya tahan tubuh yang lemah?",
    "BREATHING_ISSUE": "Apakah Anda sering mengalami masalah pernapasan?",
    "ALCOHOL_CONSUMPTION": "Apakah Anda mengonsumsi alkohol?",
    "THROAT_DISCOMFORT": "Apakah Anda sering merasa tidak nyaman di tenggorokan?",
    "CHEST_TIGHTNESS": "Apakah Anda pernah merasakan dada sesak?",
    "FAMILY_HISTORY": "Apakah ada keluarga Anda yang menderita kanker paru-paru?",
    "SMOKING_FAMILY_HISTORY": "Apakah ada keluarga yang merokok secara rutin di rumah?",
    "STRESS_IMMUNE": "Apakah stres berdampak pada daya tahan tubuh Anda?",
    "MENTAL_STRESS": "Apakah stres sering memengaruhi kondisi fisik Anda?",
    "ENERGY_LEVEL_KATEGORI": "Apakah tubuh Anda mudah lelah tanpa alasan?",
    "OXYGEN_SATURATION_KATEGORI": "Apakah saturasi oksigen Anda pernah di bawah 95%?"
}

# Tempat menyimpan input pengguna
user_input = {}

# Bagi pertanyaan ke dua kolom untuk responsivitas
st.markdown("### 📝 Kuesioner Gejala & Riwayat")
cols = st.columns(2)
i = 0
for key, question in questions.items():
    col = cols[i % 2]
    with col:
        response = st.radio(question, ["-", "Tidak", "Ya"], key=key, index=0)
        if response == "-":
            st.warning(f"Silakan isi semua pertanyaan untuk melanjutkan.")
            st.stop()
        user_input[key] = encode_binary(response)
    i += 1

# Gabungkan input
input_data = pd.DataFrame([{'AGE': AGE, **user_input}])
input_data = input_data[feature_order]  # urutkan fitur

# Tombol prediksi
st.markdown("---")
if st.button("🔍 Lakukan Prediksi"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities[prediction]

    # Tampilkan hasil prediksi
    st.subheader("📊 Hasil Prediksi")
    if prediction == 1:
        st.error("**POSITIF** – Anda *berpotensi memiliki gejala atau risiko kanker paru-paru*. "
                 "Segera konsultasikan dengan dokter untuk pemeriksaan lebih lanjut.")
    else:
        st.success("**NEGATIF** – Anda *tidak menunjukkan indikasi signifikan terhadap kanker paru-paru*. "
                   "Tetap jaga kesehatan dan lakukan pemeriksaan rutin jika diperlukan.")

    st.info(f"🔎 Tingkat Keyakinan Model: **{confidence * 100:.2f}%**")

