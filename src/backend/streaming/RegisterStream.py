import cv2
import threading
import datetime
import os
import textwrap # <--- NOUVEAU
from PIL import Image

state = {'text': "Appuyez sur ESPACE", 'busy': False}

def ai_worker(model, frame):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = model.analyze_frame(pil_img)
        state['text'] = res
    except:
        state['text'] = "Erreur"
    finally:
        state['busy'] = False

def run_register_mode(model):
    print("Préparation enregistrement...")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    save_dir = os.path.join(os.path.dirname(__file__), "records")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"video_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi")
    
    # On récupère la vraie taille de la camera
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width, height))
    
    print(f"Enregistrement : {filename}")

    window_name = 'Mode REC'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        key = cv2.waitKey(1) & 0xFF

        if key == 32 and not state['busy']:
            state['busy'] = True
            state['text'] = "Analyse..."
            threading.Thread(target=ai_worker, args=(model, frame.copy())).start()

        lines = textwrap.wrap(state['text'], width=60)
        box_height = 40 + (len(lines) * 30)
        
        cv2.rectangle(frame, (0, 0), (width, box_height), (0,0,0), -1)
        y = 30
        for line in lines:
            # Police un peu plus grosse (0.8) et plus épaisse (2) pour bien lire
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            y += 30

        out.write(frame)
        cv2.imshow(window_name, frame)
        if key == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()