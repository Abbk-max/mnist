# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 23:20:23 2026

@author: user
"""

import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import cv2
from rembg import remove
from scipy.ndimage import center_of_mass

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="MNIST Ultra-Predictor", page_icon="🔢", layout="centered")

@st.cache_resource
def load_mnist_model():
    model_path = "models/mnist_model.keras"
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_mnist_model()

# --- PIPELINE DE PRÉTRAITEMENT AVANCÉ (7 ÉTAPES + OPTIMISATIONS) ---
def expert_mnist_pipeline(img):
    # ÉTAPE 1 : ISOLATION (rembg)
    # Suppression agressive du fond pour ne garder que le tracé
    img_rgba = remove(img)
    alpha = np.array(img_rgba)[:, :, 3] # Le canal alpha est notre meilleur masque
    
    # ÉTAPE 2 : CONVERSION & NETTOYAGE
    # On transforme le masque en niveaux de gris et on lisse pour l'anti-aliasing
    gray = cv2.GaussianBlur(alpha, (3, 3), 0)
    
    # ÉTAPE 3 : NORMALISATION DU CONTRASTE & SEUIL
    # On s'assure que le chiffre est bien blanc (255) sur noir (0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ÉTAPE 4 : DILATATION (Optionnel mais recommandé)
    # Si le trait est trop fin (stylo bille), on l'épaissit légèrement pour l'IA
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # ÉTAPE 5 : LOCALISATION (Bounding Box)
    coords = cv2.findNonZero(thresh)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    
    # Extraction du chiffre de l'image GRISE (pour garder les nuances)
    roi = gray[y:y+h, x:x+w]

    # ÉTAPE 6 : REDIMENSIONNEMENT MNIST (20x20 proportionnel)
    # Le chiffre ne doit pas toucher les bords, il doit faire 20px max dans un 28x28
    max_side = max(h, w)
    scale = 20.0 / max_side
    new_w, new_h = int(w * scale), int(h * scale)
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ÉTAPE 7 : CENTRAGE PAR CENTRE DE MASSE (Canvas 28x28)
    canvas = np.zeros((28, 28), dtype=np.float32)
    
    # Calcul mathématique du centre de gravité des pixels
    cy, cx = center_of_mass(roi_resized)
    
    if not np.isnan(cy) and not np.isnan(cx):
        # On calcule le décalage pour aligner le centre de masse sur le centre géométrique (14, 14)
        off_y, off_x = int(14 - cy), int(14 - cx)
        
        # Placement dans le canvas avec protection contre les sorties de bord
        h_r, w_r = roi_resized.shape
        for i in range(h_r):
            for j in range(w_r):
                ty, tx = i + off_y, j + off_x
                if 0 <= ty < 28 and 0 <= tx < 28:
                    canvas[ty, tx] = roi_resized[i, j]

    # NORMALISATION FINALE [0, 1]
    # Inversion si nécessaire (le modèle attend le chiffre en blanc sur noir)
    final_img = canvas / 255.0
    return final_img.reshape(1, 28, 28, 1)

# --- INTERFACE UTILISATEUR ---
st.title("🔢 MNIST Ultra-Predictor")
st.markdown("### Pipeline de précision : Isolation IA + Centrage par Moments")

if model is None:
    st.error("❌ Modèle non trouvé. Placez `mnist_model.keras` dans le dossier `/models/`.")
    st.stop()

# Mode de capture
source = st.radio("Source de l'image :", ("📸 Caméra", "📁 Téléverser"), horizontal=True)
file = st.camera_input("Prendre une photo") if "📸" in source else st.file_uploader("Importer", type=["jpg", "png"])

if file:
    img = Image.open(file)
    
    with st.spinner("Analyse et normalisation..."):
        processed = expert_mnist_pipeline(img)
        
    if processed is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Original", use_container_width=True)
            st.image(processed.reshape(28,28), caption="Vision IA (Standardisée)", width=150)
            
        with col2:
            # Prédiction
            preds = model.predict(processed, verbose=0)[0]
            label = np.argmax(preds)
            confidence = np.max(preds) * 100
            
            st.success(f"## Résultat : {label}")
            st.metric("Niveau de confiance", f"{confidence:.2f}%")
            st.progress(float(confidence/100))
            
            # Analyse des probabilités
            with st.expander("Voir le détail des probabilités"):
                for i, prob in enumerate(preds):
                    st.write(f"Chiffre {i} : {prob*100:.1f}%")
                    st.sidebar.progress(float(prob)) # Petit bonus visuel
    else:
        st.warning("⚠️ Aucun tracé détecté. Essayez d'écrire plus gros ou d'utiliser un feutre plus noir.")

st.divider()
st.caption("Conseil : Pour un résultat optimal, écrivez un seul chiffre bien net sur un papier blanc.")