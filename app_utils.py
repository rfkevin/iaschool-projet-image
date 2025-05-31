import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st



def read_image_opencv(uploaded_file):
    """ Convertir fichier uploadé Streamlit en image OpenCV (BGR) """
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

# ---------------- Normalisation ----------------

def normalize_image(image, size=(320, 320)):
    """
    Redimensionne et normalise une image pour SSD et YOLO.
    """
    image_ssd = cv2.resize(image, size) / 255.0
    image_yolo = cv2.resize(image, size) / 255.0
    return image_ssd, image_yolo

# ---------------- Conversion Torch ----------------

def torch_image(image):
    """
    Convertit une image en tenseur PyTorch format NCHW.
    """
    image_tensor = torch.from_numpy(image).float()
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.permute(0, 3, 1, 2)

# ---------------- Seuil de confiance ----------------

def filtrer_detections(results, model, threshold=0.5):
    """
    Filtre les résultats YOLO selon un seuil de confiance.
    """
    resultats_trier = []
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf >= threshold:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                resultats_trier.append({
                    "Classe": cls_name,
                    "ID": cls_id,
                    "Confiance": conf,
                    "Boîte": xyxy
                })
    return resultats_trier

# ---------------- Affichage console ----------------

def afficher_detections(results, model):
    """
    Affiche les résultats de détection dans la console.
    """
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]
            st.write(f"**Classe** : {cls_name} (id {cls_id}), **Confiance** : {conf:.2f}")

# ---------------- Couleurs et Boîtes ----------------

def get_class_colors(class_ids, cmap_name="tab20"):
    """
    Attribue une couleur RGB à chaque classe.
    """
    cmap = plt.get_cmap(cmap_name)
    return {
        cid: tuple((np.array(cmap(cid % cmap.N)[:3]) * 255).astype(int))
        for cid in class_ids
    }

def draw_boxes(image_rgb, detections, colors_dict):
    """
    Dessine les boîtes de détection sur l'image (OpenCV).
    """
    image_copy = image_rgb.copy()
    for item in detections:
        cid = item["ID"]
        x1, y1, x2, y2 = map(int, item["Boîte"])
        color_rgb = colors_dict[cid]
        color_bgr = tuple(int(c) for c in reversed(color_rgb))
        print(f"x1, y1, x2, y2 = {x1}, {y1}, {x2}, {y2}")
        print(f"color_bgr = {color_bgr} type={type(color_bgr)}")
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color_bgr, 2)
    return image_copy
  
def draw_legend_on_image(image, detections, colors_dict, position=(10, 30), font_scale=0.7, thickness=2):
    # Regroupe les classes uniques
    classes = sorted(set(item['Classe'] for item in detections))
    legend_text = ", ".join(classes)
    # Convertir la couleur du premier ID en BGR pour le texte (on peut aussi choisir noir ou blanc)
    if detections:
        first_id = detections[0]['ID']
        color_bgr = tuple(int(c) for c in reversed(colors_dict.get(first_id, (255, 255, 255))))
    else:
        color_bgr = (255, 255, 255)
    
    # Ajout d'un fond sombre semi-transparent sous le texte pour la lisibilité
    (text_width, text_height), baseline = cv2.getTextSize(legend_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline), (0, 0, 0), -1)
    
    # Dessine le texte par-dessus
    cv2.putText(image, legend_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, thickness, cv2.LINE_AA)
    return image