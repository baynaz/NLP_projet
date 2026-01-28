import sys
import os

# Force X11 backend for OpenCV BEFORE any other imports
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Astuce pour que Python trouve les fichiers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_logic import InternVLModel
from streaming.VideoStream import run_view_mode
from streaming.RegisterStream import run_register_mode


def main():
    print("Initialisation du système...")
    # notre IA avec mode streaming activé pour les performances
    global_model = InternVLModel(streaming_mode=True)

    while True:
        ask = """
        ---------------------------
        SYSTEME VISION IA PRÊT
        ---------------------------
        [v] Voir (View only) - Nécessite une caméra
        [i] Enregistrer (Record) - Nécessite une caméra
        [q] Quitter
        
        Choix : """

        key = input(ask).strip().lower()

        if key == "v":
            # On donne le modèle déjà chargé à la fonction
            run_view_mode(global_model)

        elif key == "i":
            run_register_mode(global_model)

        elif key == "q":
            print("Au revoir.")
            break
        else:
            print("Touche inconnue.")


if __name__ == "__main__":
    main()
