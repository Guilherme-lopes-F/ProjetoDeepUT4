import os
import sqlite3
import uuid
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ======================================================
# CONFIGURAÇÕES
# ======================================================
IMG_SIZE = (256, 256)
BASE_DIR = "data"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ======================================================
# BANCO SQLITE (AUTO / SEGURO PARA GITHUB + STREAMLIT)
# ======================================================
DB_PATH = os.path.join(BASE_DIR, "images.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Tabela base
cursor.execute("""
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    model_mask_path TEXT,
    classification TEXT
)
""")
conn.commit()

# Função para garantir colunas (migração segura)
def ensure_column(table, column, col_type):
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if column not in cols:
        cursor.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
        )
        conn.commit()

# Colunas extras
ensure_column("dataset", "threshold", "REAL")
ensure_column("dataset", "activation_mean", "REAL")

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
# FUNÇÕES
# ======================================================
def run_unet(img: Image.Image):
    img_resized = img.resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0, :, :, 0]
    return pred

def classify(mask: np.ndarray):
    ratio = np.sum(mask > 0) / mask.size
    st.write(f"📊 Proporção segmentada: {ratio:.4f}")
    return (
        "provável presença (técnico)"
        if ratio > 0.01
        else "provável ausência (técnico)"
    )

def make_overlay(img: Image.Image, mask: np.ndarray):
    img = img.resize(IMG_SIZE)
    img_np = np.array(img)
    overlay = img_np.copy()
    overlay[mask > 0] = [255, 0, 0]
    blended = (0.7 * img_np + 0.3 * overlay).astype(np.uint8)
    return Image.fromarray(blended)

# ======================================================
# INTERFACE
# ======================================================
uploaded_file = st.file_uploader(
    "Envie uma imagem (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem original", use_column_width=True)

    if st.button("🤖 Rodar IA (U-Net)"):

        pred = run_unet(img)

        # Debug
        st.subheader("🧪 Debug da saída do modelo")
        st.json({
            "min": float(pred.min()),
            "max": float(pred.max()),
            "mean": float(pred.mean())
        })

        st.image(
            (pred * 255).astype(np.uint8),
            caption="Mapa de probabilidade"
        )

        # Threshold automático seguro
        auto_threshold = float(
            np.clip(pred.mean() + 0.05, 0.2, 0.9)
        )

        threshold = st.slider(
            "🎚️ Limiar de segmentação",
            min_value=0.2,
            max_value=0.9,
            value=auto_threshold,
            step=0.01
        )

        mask = (pred > threshold).astype(np.uint8) * 255

        st.image(mask, caption="Máscara binária")

        overlay = make_overlay(img, mask)
        st.image(overlay, caption="Overlay da segmentação")

        classification = classify(mask)
        st.info(f"Resultado técnico: **{classification}**")

        # Salvar arquivos
        img_name = f"{uuid.uuid4()}_{uploaded_file.name}"
        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, f"mask_{img_name}")

        img.save(img_path)
        Image.fromarray(mask).save(mask_path)

        # Inserir no banco (NÃO QUEBRA)
        cursor.execute("""
        INSERT INTO dataset (
            image_path,
            model_mask_path,
            classification,
            threshold,
            activation_mean
        ) VALUES (?, ?, ?, ?, ?)
        """, (
            img_path,
            mask_path,
            classification,
            float(threshold),
            float(pred.mean())
        ))
        conn.commit()

        st.success("✅ Resultado salvo com sucesso")
