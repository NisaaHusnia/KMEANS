import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="KMeans Clustering Wine", page_icon="🍷", layout="wide")

# Header aplikasi dengan gambar logo dan penjelasan aplikasi
st.markdown("""
    <div style="text-align:center;">
    </div>
    <h1 style="text-align:center;">🍷 Aplikasi KMeans Clustering Wine</h1>
    <p style="text-align:center;">Gunakan Elbow Method untuk menemukan jumlah klaster optimal pada dataset wine.</p>
""", unsafe_allow_html=True)

# Membuat bagian dengan pilihan
st.write("### Tentang Aplikasi")
st.write("""
    Aplikasi ini menggunakan algoritma KMeans untuk menganalisis dataset wine dan menentukan jumlah klaster yang optimal
    menggunakan Elbow Method. Anda bisa mengubah nilai K maksimal dan melihat hasilnya pada grafik.
""")

# Menambahkan pilihan untuk input slider dan tombol
try:
    # Memuat model KMeans
    with open("kmeans_wine_model.pkl", "rb") as file:
        kmeans_model = pickle.load(file)

    # Simulasikan dataset wine
    wine_data = pd.DataFrame({
        "alcohol": np.random.uniform(10, 15, 200),
        "total_phenols": np.random.uniform(0.1, 5, 200),
    })
    wine_features = wine_data.values

    # Input untuk jumlah maksimal K
    max_k = st.slider("Pilih Maksimal K", min_value=2, max_value=20, value=17)

    # Tombol untuk menampilkan Elbow Method
    if st.button("Tampilkan Grafik Elbow Method"):
        # Hitung SSE untuk setiap nilai K
        sse = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=10)
            kmeans.fit(wine_features)
            sse.append(kmeans.inertia_)

        # Plot grafik Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), sse, marker='o', color='#FF4B4B')
        plt.xlabel("Jumlah Kluster (K)", fontsize=12)
        plt.ylabel("Sum of Squared Errors (SSE)", fontsize=12)
        plt.title("Elbow Method untuk Menentukan Nilai Optimal K", fontsize=14)
        plt.grid(True)

        # Tampilkan grafik di Streamlit
        st.pyplot(plt)

except FileNotFoundError:
    st.error("Model `kmeans_wine_model.pkl` tidak ditemukan! Pastikan file ada di direktori yang sama.")
