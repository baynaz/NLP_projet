import sys
import os
import streamlit as st

# Force X11 backend for OpenCV
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_logic import InternVLModel
from streaming.VideoStream import run_view_mode
from streaming.RegisterStream import run_register_mode


# MODELE (1 seule fois)
@st.cache_resource
def load_model():
    return InternVLModel(streaming_mode=True)


# STREAMLIT UI
def main():
    st.set_page_config(
        page_title="Système Vision IA",
        layout="centered"
    )

    st.title("Vision IA - INternVL")
    st.write("Contrôle via Streamlit — Caméra via OpenCV")

    model = load_model()

    st.warning(
        "La caméra s’ouvrira dans une fenêtre OpenCV séparée.\n"
        "Fermez-la pour revenir à Streamlit."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Mode View "):
            st.info("Ouverture caméra (View)...")
            run_view_mode(model)

    with col2:
        if st.button("Mode Enregistrement "):
            st.info("Ouverture caméra (Record)...")
            run_register_mode(model)

    st.markdown("---")
    st.caption("Appuyez sur Q dans la fenêtre OpenCV pour fermer la caméra.")


if __name__ == "__main__":
    main()
