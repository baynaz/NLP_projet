import os
import sys

# Force X11 backend for OpenCV to avoid Wayland issues
os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

import cv2
import threading
from PIL import Image
import time
import textwrap  # Pour couper le texte proprement

state = {"text": "Appuyez sur ESPACE", "busy": False}


def find_available_camera():
    """Find the first available camera device."""
    # Try common camera indices
    for index in range(10):
        try:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                # Test if we can actually read a frame
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"Caméra trouvée à l'index {index}")
                    return index
        except Exception as e:
            continue
    return None


def ai_worker(model, frame):
    try:
        # Important pour gérer les couleurs, OpenCV voit par defaut Bleu-Vert-Rouge
        # Or le monde est en Rouge-Vert-Bleu
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Use optimized token generation for streaming (30 tokens = faster response)
        res = model.analyze_frame(pil_img, max_new_tokens=30)
        state["text"] = res
    except Exception as e:
        state["text"] = "Erreur IA"
        print(e)
    finally:
        state["busy"] = False


def run_view_mode(model):
    print("Lancement du flux vidéo.")
    
    # Find available camera
    camera_index = find_available_camera()
    if camera_index is None:
        print("\n" + "="*60)
        print("ERREUR: Aucune caméra détectée!")
        print("="*60)
        print("Vérifiez que:")
        print("  - Votre caméra est branchée")
        print("  - Les permissions sont correctes")
        print("  - Aucune autre application n'utilise la caméra")
        print("="*60 + "\n")
        return

    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return
    
    # On force une résolution raisonnable
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Test reading a frame first
    print("Test de lecture de la caméra...")
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Erreur: Impossible de lire depuis la caméra")
        cap.release()
        return
    
    print(f"[OK] Caméra OK - Résolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
    print("\nFenêtre vidéo en cours d'ouverture...")
    print("Si la fenêtre ne s'affiche pas, essayez: export QT_QPA_PLATFORM=xcb")
    
    window_name = "Mode Vue - Appuyez ESPACE pour analyser, Q pour quitter"
    
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        print("[OK] Fenêtre créée")
    except Exception as e:
        print(f"Erreur création fenêtre: {e}")
        cap.release()
        return
    
    frame_count = 0
    print("\n[START] Streaming démarré! Appuyez ESPACE pour analyser.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Erreur lecture frame, tentative de récupération...")
                time.sleep(0.1)
                continue

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF

            if key == 32 and not state["busy"]:  # ESPACE
                state["busy"] = True
                state["text"] = "Analyse en cours..."
                print(f"[ANALYSE] Analyse de la frame {frame_count}...")
                threading.Thread(target=ai_worker, args=(model, frame.copy()), daemon=True).start()

            # Afficher le texte sur la frame
            lines = textwrap.wrap(state["text"], width=60)
            box_height = 40 + (len(lines) * 30)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], box_height), (0, 0, 0), -1)

            y = 30
            for line in lines:
                cv2.putText(
                    frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                y += 30

            # Status indicator
            status_color = (0, 165, 255) if state["busy"] else (0, 255, 0)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 15, status_color, -1)

            try:
                cv2.imshow(window_name, frame)
            except Exception as e:
                print(f"Erreur affichage: {e}")
                break

            if key == ord("q") or key == 27:  # Q ou ESC
                print("\n[STOP] Arrêt demandé...")
                break

    except KeyboardInterrupt:
        print("\n[STOP] Interruption clavier...")
    except Exception as e:
        print(f"\n[ERROR] Erreur: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Nettoyage...")
        cap.release()
        cv2.destroyAllWindows()
        print("[OK] Caméra fermée proprement")


def main():
    import sys
    import os
    
    # Add parent directory to path to allow imports from backend/
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    from ai_logic import InternVLModel
    
    print("=" * 60)
    print("STREAMING EN DIRECT AVEC INFÉRENCE IA")
    print("=" * 60)
    print("Initialisation du modèle IA (mode streaming)...")
    
    # Initialize model with streaming mode for optimal performance
    model = InternVLModel(streaming_mode=True)
    
    print("\nInstructions:")
    print("  - Appuyez sur ESPACE pour analyser la frame actuelle")
    print("  - Appuyez sur Q pour quitter")
    print("=" * 60 + "\n")
    
    # Run the streaming view
    run_view_mode(model)

if __name__ == "__main__":
    main()

