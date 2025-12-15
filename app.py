import os
import sqlite3
import uuid
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ======================================================
# CONFIGURAÇÕES GERAIS
# ======================================================
IMG_SIZE = (256, 256)
BASE_DIR = "data"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ======================================================
# BANCO DE DADOS
# ======================================================
DB_PATH = os.path.join(BASE_DIR, "images.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    model_mask_path TEXT,
    classification TEXT,
    threshold REAL,
    activation_mean REAL
)
""")
conn.commit()

# ======================================================
# CARREGAR MODELO
# ======================================================
MODEL_PATH = "modelo_sicapv2_unet.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("🔬 Análise Técnica de Segmentação — U-Net (DEMO)")

if model is None:
    st.error("❌ Modelo U-Net não encontrado")
    st.stop()

st.success("✅ Modelo carregado com sucesso")

# ======================================================
# FUNÇÃO DE SEGMENTAÇÃO (COM DEBUG)
# ======================================================
def run_unet_segmentation(img: Image.Image, threshold: float):
    # Pré-processamento correto
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predição
    pred = model.predict(arr, verbose=0)[0, :, :, 0]

    # Debug essencial
    st.subheader("🧪 Debug da saída do modelo")
    st.write({
        "min": float(pred.min()),
        "max": float(pred.max()),
        "mean": float(pred.mean())
    })

    # Mapa de probabilidade
    prob_map = (pred * 255).astype(np.uint8)
    st.image(prob_map, caption="Mapa de probabilidade (saída bruta da U-Net)")

    # Binarização com threshold ajustável
    mask = (pred > threshold).astype(np.uint8) * 255

    return Image.fromarray(mask), float(pred.mean())

# ======================================================
# CLASSIFICAÇÃO TÉCNICA
# ======================================================
def classify_mask(mask_img: Image.Image):
    arr = np.array(mask_img)
    tumor_pixels = np.sum(arr > 0)
    total_pixels = arr.size
    ratio = tumor_pixels / total_pixels

    st.write(f"📊 Proporção segmentada: {ratio:.4f}")

    return (
        "provável presença (técnico)"
        if ratio > 0.01
        else "provável ausência (técnico)"
    )

# ======================================================
# OVERLAY (VISUAL PROFISSIONAL)
# ======================================================
def overlay_mask(image: Image.Image, mask: Image.Image):
    image = image.resize(IMG_SIZE)
    image_np = np.array(image)
    mask_np = np.array(mask)

    overlay = image_np.copy()
    overlay[mask_np > 0] = [255, 0, 0]  # vermelho

    blended = (0.7 * image_np + 0.3 * overlay).astype(np.uint8)
    return Image.fromarray(blended)

# ======================================================
# INTERFACE
# ======================================================
uploaded_file = st.file_uploader(
    "Envie uma imagem (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem original", use_column_width=True)

    # Slider de threshold (ESSENCIAL)
    threshold = st.slider(
        "🎚️ Limiar de segmentação",
        min_value=0.01,
        max_value=0.9,
        value=0.2,
        step=0.01
    )

    if st.button("🤖 Rodar IA (U-Net)"):

        mask, activation_mean = run_unet_segmentation(img, threshold)

        # Salvar arquivos
        img_name = f"{uuid.uuid4()}_{uploaded_file.name}"
        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, f"mask_{img_name}")

        img.save(img_path)
        mask.save(mask_path)

        st.success("✅ Segmentação concluída")

        st.image(mask, caption="Máscara binária")
        overlay_img = overlay_mask(img, mask)
        st.image(overlay_img, caption="Overlay da segmentação")

        classification = classify_mask(mask)
        st.info(f"Resultado técnico: **{classification}**")

        # Salvar no banco
        cursor.execute("""
        INSERT INTO dataset (
            image_path, model_mask_path, classification, threshold, activation_mean
        ) VALUES (?, ?, ?, ?, ?)
        """, (img_path, mask_path, classification, threshold, activation_mean))
        conn.commit()
