import os
import sqlite3
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ======================================================
# 1. Criar diretórios automaticamente
# ======================================================
BASE_DIR = "data"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ======================================================
# 2. Conectar SQLite
# ======================================================
DB_PATH = os.path.join(BASE_DIR, "images.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    mask_path TEXT,
    model_mask_path TEXT,
    classification TEXT
)
""")
conn.commit()

# ======================================================
# 3. Carregar modelo U-Net
# ======================================================
MODEL_PATH = "modelo_sicapv2_unet.h5"
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success(f"Modelo carregado de: {MODEL_PATH}")
else:
    st.warning("⚠️ Modelo não encontrado: modelo_sicapv2_unet.h5")

# ======================================================
# 4. Função: máscara por claridade
# ======================================================
def create_brightness_mask(img: Image.Image):
    gray = img.convert("L")
    arr = np.array(gray)
    threshold = arr.mean()
    mask = (arr > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask)

# ======================================================
# 5. Função: prever máscara usando U-Net
# ======================================================
def run_unet_segmentation(img: Image.Image, target_size=(256, 256)):
    if model is None:
        st.error("⚠️ Modelo não carregado. Não é possível realizar a segmentação.")
        return None

    # Redimensionar a imagem e normalizar
    img_resized = img.resize(target_size)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Adiciona a dimensão batch

    # Prever a máscara
    pred = model.predict(arr)[0]
    
    # Limiar para binarizar a saída
    pred_mask = (pred[:, :, 0] > 0.5).astype(np.uint8) * 255

    # Retornar a imagem binarizada
    return Image.fromarray(pred_mask)

# ======================================================
# 6. Classificação técnica: “provável presença” / “provável ausência”
# ======================================================
def classify_mask(mask_img: Image.Image):
    arr = np.array(mask_img)
    
    # Contar pixels com valor maior que 0 (indicando presença de área segmentada)
    tumor_pixels = np.sum(arr > 0)

    # Visualizar o número de pixels "ativos" na máscara
    st.write(f"Pixels com valor maior que 0 (indicação de tumor): {tumor_pixels}")

    # Ajustar o limiar com base na quantidade de pixels
    return "provável presença (técnico)" if tumor_pixels > 100 else "provável ausência (técnico)"

# ======================================================
# 7. Interface Streamlit
# ======================================================
st.title("🔬 Análise Técnica de Imagens — SICAPv2 (DEMO)")

# Upload da imagem
uploaded_file = st.file_uploader("Envie uma imagem (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem enviada", use_column_width=True)

    img_name = uploaded_file.name
    img_path = os.path.join(IMG_DIR, img_name)
    img.save(img_path)

    # Salvar no banco de dados
    cursor.execute("INSERT INTO dataset (image_path) VALUES (?)", (img_path,))
    conn.commit()

    # Criar máscara por claridade
    st.subheader("🧪 Criar máscara por claridade")
    if st.button("Gerar máscara por claridade"):
        brightness_mask = create_brightness_mask(img)
        mask_path = os.path.join(MASK_DIR, f"brightness_{img_name}")
        brightness_mask.save(mask_path)

        cursor.execute("UPDATE dataset SET mask_path = ? WHERE image_path = ?", (mask_path, img_path))
        conn.commit()

        st.success("Máscara criada!")
        st.image(brightness_mask, caption="Máscara por claridade", use_column_width=True)

    # Rodar modelo U-Net
    st.subheader("🤖 Rodar Segmentação com a U-Net")

    if st.button("Rodar IA (U-Net)"):

        if model is None:
            st.error("⚠️ Modelo não carregado. Não é possível realizar a segmentação.")
        else:
            pred_mask = run_unet_segmentation(img, model)

            if pred_mask:
                model_mask_path = os.path.join(MASK_DIR, f"modelmask_{img_name}")
                pred_mask.save(model_mask_path)

                cursor.execute("UPDATE dataset SET model_mask_path = ? WHERE image_path = ?", (model_mask_path, img_path))
                conn.commit()

                st.success("Segmentação gerada!")
                st.image(pred_mask, caption="Segmentação da IA (não médica)")

                # Classificação técnica
                st.subheader("📘 Classificação Técnica (NÃO MÉDICA)")
                classification = classify_mask(pred_mask)

                cursor.execute("UPDATE dataset SET classification = ? WHERE image_path = ?", (classification, img_path))
                conn.commit()

                st.info(f"Resultado técnico: **{classification}**")



