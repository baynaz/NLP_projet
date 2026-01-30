import streamlit as st
from PIL import Image
import tempfile
import os
import sys
from transformers import AutoTokenizer

token = st.secrets["HF_TOKEN"]

tokenizer = AutoTokenizer.from_pretrained(
    "OpenGVLab/InternVL3_5-1B",
    use_auth_token=token
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_logic import InternVLModel
from streaming.inference_multi import (
    process_video_with_inference,
    analyze_full_video,
    format_results
)


# Configuration Streamlit

st.set_page_config(page_title="InternVL – Image & Video Analyzer", layout="centered")
st.title("Analyse d’images et vidéos")


@st.cache_resource
def load_model():
    return InternVLModel(streaming_mode=True)

st.sidebar.write("⚡ Chargement du modèle IA...")
model = load_model()
st.sidebar.write("Modèle prêt")


mode = st.radio(
    "Que voulez-vous analyser ?",
    ("Image", "Vidéo")
)


# MODE IMAGE
if mode == "Image":
    st.subheader("Capture d'image via webcam")

    img_file = st.camera_input("Prends une photo")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Image capturée", use_container_width=True)

        with st.spinner(" Analyse IA en cours..."):
            response = model.analyze_frame(
                image,
                max_new_tokens=40
            )

        st.markdown("### Résultat IA")
        st.write(response)


# MODE VIDÉO

elif mode == "Vidéo":
    st.subheader("Uploader une vidéo")

    uploaded_video = st.file_uploader(
        "Choisissez une vidéo (mp4, avi, mov)",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        # Sauvegarde temporaire
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.success("Vidéo uploadée avec succès")
        st.video(temp_video.name)

        # Choix du type d’analyse
        analysis_mode = st.radio(
            "Type d'analyse",
            (
                "Description globale de la vidéo",
                "Analyse segment par segment"
            )
        )

        # Paramètre commun
        num_segments = st.slider(
            "Nombre de segments / frames analysés",
            min_value=2,
            max_value=12,
            value=4
        )

        # ANALYSE GLOBALE
        if analysis_mode == "Description globale de la vidéo":
            st.info(" Analyse globale de la vidéo")

            with st.spinner("Analyse en cours..."):
                result = analyze_full_video(
                    video_path=temp_video.name,
                    model=model,
                    num_segments=num_segments
                )

            st.subheader("📝 Description complète")
            st.write(result)

        # ANALYSE PAR SEGMENTS
        elif analysis_mode == "Analyse segment par segment":
            st.info(" Analyse segmentée de la vidéo")

            with st.spinner("Analyse en cours..."):
                results = process_video_with_inference(
                    video_path=temp_video.name,
                    num_segments=num_segments,
                    model=model,
                    cleanup_segments=True
                )

            st.subheader(" Résultats par segment")
            st.markdown(format_results(results, output_format="markdown"))
