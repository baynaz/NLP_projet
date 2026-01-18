import cv2
import threading
from PIL import Image
import time
import textwrap  #Pour couper le texte proprement

state = {'text': "Appuyez sur ESPACE", 'busy': False}

def ai_worker(model, frame):
    try:
        #Important pour gérer les couleurs, OpenCV voit par defaut Bleu-Vert-Rouge
        #Or le monde est en Rouge-Vert-Bleu
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = model.analyze_frame(pil_img)
        state['text'] = res
    except Exception as e:
        state['text'] = "Erreur IA"
        print(e)
    finally:
        state['busy'] = False

def run_view_mode(model):
    print("Lancement du flux vidéo.")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # On force une résolution un peu meilleure si possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # On envoie une photo en jpg au modèle
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    if not cap.isOpened():
        print("Erreur Caméra")
        return

    window_name = 'Mode Vue'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Permet de redimensionner
    cv2.resizeWindow(window_name, 1024, 768)        # Taille de départ
    

    while True:
        ret, frame = cap.read()
        if not ret: 
            time.sleep(0.1)
            continue

        key = cv2.waitKey(1) & 0xFF # Nombre Bizarre & 0xFF = Code ASCII Propre

        if key == 32 and not state['busy']:# en ASCII, 32 = espace
            state['busy'] = True
            state['text'] = "Analyse de l'image"
            threading.Thread(target=ai_worker, args=(model, frame.copy())).start()

        
        # On coupe le texte tous les 60 caractères pour qu'il tienne
        lines = textwrap.wrap(state['text'], width=60)
        
        # On dessine un rectangle noir qui s'adapte à la taille du texte
        box_height = 40 + (len(lines) * 30)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], box_height), (0,0,0), -1)
        
        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            y += 30

        cv2.imshow(window_name, frame)
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()