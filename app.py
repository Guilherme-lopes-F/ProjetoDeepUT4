import os
import sqlite3
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split

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
# 3. Função para criar o modelo U-Net
# ======================================================
def build_unet(input_size=(256, 256, 3)):
    inputs = tf.keras.layers.Input(input_size)
    
    # Camadas U-Net
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same")(c4)

    u1 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c4)
    u1 = tf.keras.layers.concatenate([u1, c3])
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(u1)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same")(c5)

    u2 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(u2)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c6)

    u3 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
    u3 = tf.keras.layers.concatenate([u3, c1])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u3)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c7)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)  # Saída binária
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

# ======================================================
# 4. Função: carregar imagens e máscaras
# ======================================================
def load_data(img_dir, mask_dir):
    images = []
    masks = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Máscara em escala de cinza

        images.append(np.array(img) / 255.0)  # Normaliza a imagem
        masks.append(np.array(mask) / 255.0)  # Normaliza a máscara

    return np.array(images), np.array(masks)

# ======================================================
# 5. Função: prever máscara usando U-Net
# ======================================================
def run_unet_segmentation(img: Image.Image, model, target_size=(256, 256)):
    # Redimensionar e normalizar a imagem
    img_resized = img.resize(target_size)
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Adiciona a dimensão de batch

    # Prever a máscara
    pred = model.predict(arr)[0]
    
    # Limiar para binarizar a saída
    pred_mask = (pred[:, :, 0] > 0.5).astype(np.uint8) * 255

    # Retornar a imagem binarizada
    return Image.fromarray(pred_mask)

# ======================================================
# 6. Interface Streamlit
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

        if "model" not in st.session_state:
            st.error("Modelo não carregado.")
        else:
            model = st.session_state["model"]
            pred_mask = run_unet_segmentation(img, model)

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

# Treinamento do modelo U-Net
st.subheader("⚙️ Treinamento do Modelo U-Net")

if st.button("🚀 Treinar Modelo U-Net"):
    images, masks = load_data(IMG_DIR, MASK_DIR)

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    # Criar e treinar o modelo
    model = build_unet(input_size=(256, 256, 3))
    model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

    # Salvar o modelo
    model.save("unet_model.h5")
    st.session_state["model"] = model

    st.success("Modelo treinado com sucesso!")

