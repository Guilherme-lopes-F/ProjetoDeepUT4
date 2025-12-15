import os
import sqlite3
import uuid
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ======================================================
# AVISO ÉTICO (FORTE E VISÍVEL)
# ======================================================
st.error(
    "⚠️ AVISO MUITO IMPORTANTE ⚠️\n\n"
    "Este sistema é APENAS UM TESTE EXPERIMENTAL.\n"
    "Ele NÃO realiza diagnóstico médico.\n"
    "Os resultados exibidos NÃO devem ser usados para decisões reais.\n"
    "Classificação feita exclusivamente por um modelo de IA."
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
    cancer_result TEXT,
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

st.title("🧪 Classificação Experimental — IA (DEMO)")

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
    return model.predict(arr, verbose=0)[0, :, :, 0]

def overlay(img, mask):
    img = img.resize(IMG_SIZE)
    img_np = np.array(img)
    ov = img_np.copy()
    ov[mask > 0] = [255, 0, 0]
    return Image.fromarray((0.7 * img_np + 0.3 * ov).astype(np.uint8))

# ======================================================
# DECISÃO EXPERIMENTAL (CÂNCER / NÃO CÂNCER)
# ======================================================
def experimental_cancer_decision(mask, pred):
    ratio = np.sum(mask > 0) / mask.size
    mean_act = pred.mean()
    max_act = pred.max()

    # 🔬 REGRA EXPERIMENTAL (NÃO MÉDICA)
    if ratio < 0.4 and mean_act > 0.22 and max_act > 0.35:
        return "CÂNCER (resultado experimental do modelo)"
    else:
        return "NÃO CÂNCER (resultado experimental do modelo)"

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

    if st.button("🤖 Rodar IA (EXPERIMENTAL)"):

        pred = run_unet(img)

        st.subheader("🧪 Debug do Modelo")
        st.json({
            "min": float(pred.min()),
            "max": float(pred.max()),
            "mean": float(pred.mean())
        })

        threshold = float(np.clip(pred.mean() + 0.05, 0.2, 0.9))
        mask = (pred > threshold).astype(np.uint8) * 255

        st.image(mask, caption="Máscara gerada")
        st.image(overlay(img, mask), caption="Overlay")

        # ===== RESULTADO FINAL =====
        st.subheader("🔴 Resultado do Modelo (EXPERIMENTAL)")
        cancer_result = experimental_cancer_decision(mask, pred)

        if "CÂNCER" in cancer_result:
            st.error(cancer_result)
        else:
            st.success(cancer_result)

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
            cancer_result,
            threshold,
            activation_mean,
            activation_max
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            img_path,
            mask_path,
            cancer_result,
            threshold,
            float(pred.mean()),
            float(pred.max())
        ))
        conn.commit()

        st.success("✅ Análise experimental concluída")
