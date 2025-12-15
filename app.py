import os
import sqlite3
import uuid
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ======================================================
# AVISO IMPORTANTE
# ======================================================
st.warning(
    "⚠️ ATENÇÃO:\n"
    "Este sistema é apenas uma DEMONSTRAÇÃO TÉCNICA.\n"
    "Não realiza diagnóstico médico.\n"
    "Use apenas imagens histológicas compatíveis com o treino."
)

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
# BANCO SQLITE (AUTO / SEGURO)
# ======================================================
DB_PATH = os.path.join(BASE_DIR, "images.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    model_mask_path TEXT,
    classification TEXT
)
""")
conn.commit()

def ensure_column(table, column, col_type):
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cursor.fetchall()]
    if column not in cols:
        cursor.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
        )
        conn.commit()

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

st.title("🔬 Segmentação Técnica com U-Net (DEMO)")

if model is None:
    st.error("❌ Modelo não encontrado")
    st.stop()

st.success("✅ Modelo carregado")

# ======================================================
# FUNÇÕES
# ======================================================
def is_histology_like(img: Image.Image):
    arr = np.array(img)
    mean_color = arr.mean(axis=(0, 1))
    # lâminas histológicas tendem a tons rosados/arroxeados
    return mean_color[0] > 120 and mean_color[2] > 120

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
        "ativação detectada (modelo técnico, não médico)"
        if ratio > 0.01
        else "nenhuma ativação relevante (modelo técnico)"
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
    "Envie uma imagem HISTOLÓGICA (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem enviada", use_column_width=True)

    # 🔒 BLOQUEIO DE IMAGEM FORA DO DOMÍNIO
    if not is_histology_like(img):
        st.error(
            "❌ Imagem fora do domínio do modelo.\n"
            "Envie apenas imagens histológicas microscópicas."
        )
        st.stop()

    if st.button("🤖 Rodar IA (U-Net)"):

        pred = run_unet(img)

        # DEBUG
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

        # SALVAR
        img_name = f"{uuid.uuid4()}_{uploaded_file.name}"
        img_path = os.path.join(IMG_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, f"mask_{img_name}")

        img.save(img_path)
        Image.fromarray(mask).save(mask_path)

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
