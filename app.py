# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 23:20:23 2026

@author: user
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2
from scipy.ndimage import center_of_mass

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="MNIST Ultra-Predictor",
    page_icon="🔢",
    layout="centered"
)

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_mnist_model():
    model_path = "mnist_model.keras"
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = load_mnist_model()

# --- PIPELINE DE PRÉTRAITEMENT MNIST (CLOUD SAFE) ---
def expert_mnist_pipeline(img: Image.Image):
    # 1️⃣ Conversion en niveaux de gris
    gray = np.array(img.convert("L"))

    # 2️⃣ Lissage (anti-bruit)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3️⃣ Binarisation automatique (fond noir / chiffre blanc)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4️⃣ Épaississement léger du trait
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # 5️⃣ Bounding box du chiffre
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    roi = thresh[y:y + h, x:x + w]

    # 6️⃣ Redimensionnement proportionnel (max 20x20)
    max_side = max(w, h)
    scale = 20.0 / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 7️⃣ Centrage par centre de masse dans un canvas 28x28
    canvas = np.zeros((28, 28), dtype=np.float32)
    cy, cx = center_of_mass(roi)

    if not np.isnan(cy) and not np.isnan(cx):
        off_y = int(14 - cy)
        off_x = int(14 - cx)

        for i in range(new_h):
            for j in range(new_w):
                ty, tx = i + off_y, j + off_x
                if 0 <= ty < 28 and 0 <= tx < 28:
                    canvas[ty, tx] = roi[i, j]

    # 8️⃣ Normalisation finale
    canvas /= 255.0
    return canvas.reshape(1, 28, 28, 1)

# --- INTERFACE UTILISATEUR ---
st.title("🔢 MNIST Ultra-Predictor")
st.markdown("### Pipeline de précision : Seuil adaptatif + Centrage par moments")

if model is None:
    st.error("❌ Modèle non trouvé. Place `mnist_model.keras` dans `/models/`.")
    st.stop()

source = st.radio(
    "Source de l'image :",
    ("📸 Caméra", "📁 Téléverser"),
    horizontal=True
)

file = (
    st.camera_input("Prendre une photo")
    if "📸" in source
    else st.file_uploader("Importer une image", type=["jpg", "png"])
)

if file:
    img = Image.open(file)

    with st.spinner("Analyse et normalisation..."):
        processed = expert_mnist_pipeline(img)

    if processed is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Image originale", use_container_width=True)
            st.image(
                processed.reshape(28, 28),
                caption="Vision IA (28×28)",
                width=150
            )

        with col2:
            preds = model.predict(processed, verbose=0)[0]
            label = int(np.argmax(preds))
            confidence = float(np.max(preds) * 100)

            st.success(f"## Résultat : {label}")
            st.metric("Confiance", f"{confidence:.2f}%")
            st.progress(confidence / 100)

            with st.expander("Détail des probabilités"):
                for i, p in enumerate(preds):
                    st.write(f"{i} → {p * 100:.1f}%")
    else:
        st.warning("⚠️ Aucun chiffre détecté. Écris plus gros et plus contrasté.")

st.divider()
st.caption(
    "Conseil : écris un seul chiffre bien centré, noir sur fond blanc."
)
