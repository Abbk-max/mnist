def refine_image_for_thin_strokes(self, raw_img):
    """Prétraitement optimisé pour les traits fins et épais"""
    # 1. Conversion en niveaux de gris
    if raw_img.shape[-1] == 4:
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)

    # 2. Amélioration du contraste (CLAHE) pour les traits pâles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. Seuillage inversé (le chiffre doit être blanc sur fond noir)
    # On utilise OTSU mais on dilate si le trait est trop fin
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- ÉTAPE CRUCIALE POUR LES ÉCRITURES FINES ---
    # On calcule la densité du tracé
    nb_pixels_blancs = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    densite = nb_pixels_blancs / total_pixels

    # Si la densité est faible (trait fin), on applique une dilation
    if densite < 0.05: # Seuil à ajuster selon tes tests
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
    # -----------------------------------------------

    # 4. Crop automatique autour du chiffre
    coords = cv2.findNonZero(thresh)
    if coords is None: return None
    x, y, w, h = cv2.boundingRect(coords)
    roi = thresh[y:y+h, x:x+w]

    # 5. Redimensionnement avec INTER_AREA (meilleur pour la réduction)
    # On garde une marge pour ne pas toucher les bords du 28x28
    final_size = 20
    if w > h:
        new_w = final_size
        new_h = int(h * (final_size / w))
    else:
        new_h = final_size
        new_w = int(w * (final_size / h))
    
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 6. Centrage dans un cadre 28x28
    display_img = np.zeros((28, 28), dtype=np.uint8)
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    display_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = roi_resized

    # 7. Lissage final pour simuler le flou de MNIST
    display_img = cv2.GaussianBlur(display_img, (3, 3), 0)

    return (display_img / 255.0).reshape(1, 28, 28, 1)
