import streamlit as st
from ultralytics import YOLO
from gtts import gTTS
from io import BytesIO
from PIL import Image
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="A-Eyes: Asisten Tunanetra", layout="centered")

# --- LOAD MODEL (Cache biar gak loading terus) ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- HARGA UANG ---
currency_map = {
    '100k': 100000,
    '50k': 50000,
    '20k': 20000,
    '10k': 10000,
    '5k': 5000,
    '2k': 2000,
    '1k': 1000
}

# --- FUNGSI SUARA (GOOGLE TTS) ---
def speak_cloud(text):
    # Simpan suara ke memory (buffer), bukan file fisik biar cepat
    sound_file = BytesIO()
    tts = gTTS(text, lang='id')
    tts.write_to_fp(sound_file)
    return sound_file

# --- TAMPILAN UTAMA ---
st.title("👁️ A-Eyes")
st.write("Arahkan kamera ke uang, lalu tekan **'Ambil Foto'**.")

# 1. INPUT KAMERA NATIVE STREAMLIT
# Ini bekerja di HP Android/iPhone temanmu!
img_file = st.camera_input("Jendela Kamera")

if img_file is not None:
    # 2. PROSES GAMBAR
    # Convert file upload jadi gambar yang bisa dibaca YOLO
    image = Image.open(img_file)
    img_array = np.array(image)
    
    # YOLO butuh input gambar
    results = model(img_array, conf=0.5)
    
    # 3. HITUNG DUIT
    total_uang = 0
    terdeteksi = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            if cls_name in currency_map:
                total_uang += currency_map[cls_name]
                terdeteksi.append(cls_name)

    # 4. TAMPILKAN HASIL GAMBAR (Dengan Kotak)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)

    # 5. TAMPILKAN TOTAL & SUARA
    if total_uang > 0:
        st.success(f"💰 Total: Rp {total_uang:,}")
        
        # Buat teks ucapan
        teks_suara = f"Total uang terdeteksi adalah {total_uang} rupiah."
        audio_bytes = speak_cloud(teks_suara)
        
        # Putar Audio (Autoplay di beberapa browser mungkin di-block, jadi user harus play manual kadang)
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
    else:
        st.warning("Tidak ada uang yang terdeteksi. Coba lagi!")