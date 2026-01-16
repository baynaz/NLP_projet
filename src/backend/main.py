import sys
import os

# Astuce pour que Python trouve les fichiers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_logic import InternVLModel
from streaming.VideoStream import run_view_mode
from streaming.RegisterStream import run_register_mode

def main():
    print("Initialisation du système...")
    # notre IA
    global_model = InternVLModel() 
    
    while True:
        ask = """
        ---------------------------
        SYSTEME VISION IA PRÊT
        ---------------------------
        [v] Voir (View only)
        [i] Enregistrer (Record)
        [q] Quitter
        
        Choix : """
        
        key = input(ask).strip().lower()
        
        if key == 'v':
            # On donne le modèle déjà chargé à la fonction
            run_view_mode(global_model) 
            
        elif key == 'i':
            run_register_mode(global_model)
            
        elif key == 'q':
            print("Au revoir.")
            break
        else:
            print("Touche inconnue.")

if __name__ == "__main__":
    main()