import os
import sqlite3
import uuid
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

# ======================================================
# 1. Diretórios
# ======================================================
BASE_DIR = "data"
IMG_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

# ======================================================
# 2. Banco SQLite
# ======================================================
DB_PATH = os.path.join(BASE_DIR, "images.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT,
    mask_path TEXT,
    model_mask_path TEXT,
    classification TEXT
)
""")
conn.commit()

# ======================================================
# 3. Carregar modelo
# ======================================================
MODEL_PATH = "modelo_sicapv2_unet.h5"
model = None

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("Modelo U-Net carregado com sucesso")
else:
    st.error("Modelo não encontrado")

# ======================================================
# 4. Segmentação com U-Net (CORRIGIDA)
# ======================================================
def run_unet_segmentation(img: Image.Image, model):
    if model is None:
        return None

    # ✅ Forçar tamanho do treino
    img_resized = img.resize((256, 256))

    # ✅ Normalização correta
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Predição
    pred = model.predict(arr, verbose=0)[0, :, :, 0]

    # ✅ Limiar padrão
    mask = (pred > 0.5).astype(np.uint8) * 255

    return Image.fromarray(mask)

# ======================================================
# 5. Classificação técnica (CORRIGIDA)
# ======================================================
def classify_mask(mask_img: Image.Image):
    arr = np.array(mask_img)
    tumor_pixels = np.sum(arr > 0)
    total_pixels = arr.size

    ratio = tumor_pixels / total_pixels
    st.write(f"Proporção segmentada: {ratio:.4f}")

    return (
        "provável presença (técnico)"
        if ratio > 0.01
        else "provável ausência (técnico)"
    )

# ======================================================
# 6. Interface Streamlit
# ======================================================
st.title("🔬 Análise Técnica de Imagens — SICAPv2 (DEMO)")

uploaded_file = st.file_uploader(
    "Envie uma imagem (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem enviada", use_column_width=True)

    # ✅ Evita sobrescrever arquivos
    img_name = f"{uuid.uuid4()}_{uploaded_file.name}"
    img_path = os.path.join(IMG_DIR, img_name)
    img.save(img_path)

    if st.button("🤖 Rodar IA (U-Net)"):

        pred_mask = run_unet_segmentation(img, model)

        if pred_mask:
            mask_name = f"mask_{img_name}"
            model_mask_path = os.path.join(MASK_DIR, mask_name)
            pred_mask.save(model_mask_path)

            classification = classify_mask(pred_mask)

            # Salvar tudo no banco
            cursor.execute("""
            INSERT INTO dataset (image_path, model_mask_path, classification)
            VALUES (?, ?, ?)
            """, (img_path, model_mask_path, classification))
            conn.commit()

            st.success("Segmentação concluída")
            st.image(pred_mask, caption="Máscara gerada pela IA")

            st.info(f"Resultado técnico: **{classification}**")

# ======================================================
# 7. Treino (desativado)
# ======================================================
st.subheader("⚙️ Treinamento do Modelo")
st.warning("Treinamento desativado nesta versão.")
