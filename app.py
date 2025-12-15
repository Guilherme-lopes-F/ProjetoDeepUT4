import os
import sqlite3
import uuid
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ======================================================
# AVISO LEGAL / ÉTICO
# ======================================================
st.warning(
    "⚠️ AVISO IMPORTANTE:\n"
    "Este sistema é apenas uma DEMONSTRAÇÃO TÉCNICA.\n"
    "Não realiza diagnóstico médico.\n"
    "Os resultados indicam apenas padrões de ativação do modelo."
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
# BANCO SQLITE (AUTO)
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
    pattern_analysis TEXT,
    threshold REAL,
    activation_mean REAL,
    activation_max REAL
)
""")
conn.commit()

# ======================================================
# MODELO
# ======================================================
MODEL_PATH = "modelo_sicapv2_unet.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

st.title("🔬 Análise Técnica com U-Net (DEMO)")

if model is None:
    st.error("❌ Modelo não encontrado")
    st.stop()

# ======================================================
# FUNÇÕES
# ======================================================
def is_histology_like(img):
    arr = np.array(img)
    mean_color = arr.mean(axis=(0, 1))
    return mean_color[0] > 120 and mean_color[2] > 120

def run_unet(img):
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)[0, :, :, 0]
    return pred

def make_overlay(img, mask):
    img = img.resize(IMG_SIZE)
    img_np = np.array(img)
    overlay = img_np.copy()
    overlay[mask > 0] = [255, 0, 0]
    return Image.fromarray((0.7 * img_np + 0.3 * overlay).astype(np.uint8))

# ======================================================
# CLASSIFICAÇÕES
# ======================================================
def technical_presence(mask):
    ratio = np.sum(mask > 0) / mask.size
    st.write(f"📊 Proporção segmentada: {ratio:.4f}")

    return (
        "ativação detectada (técnico)"
        if ratio > 0.01
        else "nenhuma ativação relevante (técnico)"
    )

def pattern_analysis(mask, pred):
    ratio = np.sum(mask > 0) / mask.size
    mean_act = pred.mean()
    max_act = pred.max()

    # 🔬 Heurística técnica
    if ratio < 0.2 and mean_act > 0.25 and max_act > 0.4:
        return "padrão de ativação compatível com tecido tumoral (técnico)"
    elif ratio > 0.6:
        return "ativação difusa não específica (técnico)"
    else:
        return "padrão indefinido / inconclusivo (técnico)"

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

    if not is_histology_like(img):
        st.error("❌ Imagem fora do domínio do modelo.")
        st.stop()

    if st.button("🤖 Rodar IA"):

        pred = run_unet(img)

        st.subheader("🧪 Debug do Modelo")
        st.json({
            "min": float(pred.min()),
            "max": float(pred.max()),
            "mean": float(pred.mean())
        })

        st.image((pred * 255).astype(np.uint8), caption="Mapa de probabilidade")

        threshold = float(np.clip(pred.mean() + 0.05, 0.2, 0.9))
        st.caption(f"Limiar automático usado: {threshold:.2f}")

        mask = (pred > threshold).astype(np.uint8) * 255
        st.image(mask, caption="Máscara binária")

        overlay = make_overlay(img, mask)
        st.image(overlay, caption="Overlay")

        # ===== RESULTADOS =====
        st.subheader("📘 Resultado Técnico")
        presence = technical_presence(mask)
        pattern = pattern_analysis(mask, pred)

        st.info(f"**Presença técnica:** {presence}")
        st.info(f"**Análise de padrão:** {pattern}")

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
            pattern_analysis,
            threshold,
            activation_mean,
            activation_max
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            img_path,
            mask_path,
            presence,
            pattern,
            threshold,
            float(pred.mean()),
            float(pred.max())
        ))
        conn.commit()

        st.success("✅ Análise técnica concluída e salva")
