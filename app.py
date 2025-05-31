from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from app_utils import normalize_image, torch_image, filtrer_detections, get_class_colors, draw_boxes, read_image_opencv, afficher_detections, draw_legend_on_image


model = YOLO("mon_modele.pt")

st.title("Détection d'objets avec Yolo ")

uploaded_file = st.file_uploader("Uploadez une image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = read_image_opencv(uploaded_file)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Image uploadée", use_container_width=True)
    image_ssd, image_yolo = normalize_image(image)
    tensor = torch_image(image_yolo)
    results = model(tensor)
    st.warning(f"Nombre de détections : {len(results[0].boxes.xywh)}")
    detections = filtrer_detections(results, model)
    afficher_detections(results, model)
    if len(detections) == 0:
        st.warning("Aucune détection avec ce seuil.")
    else:
        st.warning(len(detections))
        class_ids = sorted(set(item['ID'] for item in detections))
        colors = get_class_colors(class_ids)

        if image_yolo.dtype != np.uint8:
            image_yolo = (image_yolo * 255).astype(np.uint8)

        image_with_boxes = draw_boxes(image_yolo, detections, colors)
        image_with_boxes = image_with_boxes.astype(np.uint8)
        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption="Image annotée", use_container_width=True)
        is_success, buffer = cv2.imencode(".jpg", image_with_boxes)
        if is_success:
            btn = st.download_button(
            label="Télécharger l'image annotée",
            data=buffer.tobytes(),
            file_name="image_annotée.jpg",
            mime="image/jpeg"
        )
            

        # Affichage légende simple
        legend = ", ".join(set(item['Classe'] for item in detections))
        st.markdown(f"**Classes détectées:** {legend}")
