#!/usr/bin/env python

# http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html#saving-a-video
# FFMPEG FOURCC tags: http://ffmpeg.org/doxygen/trunk/isom_8c-source.html
# FOURCC codecs: http://www.fourcc.org/codecs.php
# $ ffmpeg --codecs
# http://www.ftyps.com/

from __future__ import print_function
import os
import sys

# Force X11 backend for OpenCV to avoid Wayland issues
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import time
from pathlib import Path

# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------

def run_register_mode(model=None):
    """Run video recording mode with optional AI inference.
    
    Args:
        model: Optional AI model for inference (currently not used in recording mode)
    """
    # keys
    KEY_R = ord("r")  # start recording
    KEY_S = ord("s")  # stop recording
    KEY_Q = ord("q")  # quit
    KEY_ESC = 27  # quit

    # font
    FONT = cv2.FONT_HERSHEY_PLAIN

    # video file size
    VIDEO_FILE_SIZE = 10 * 1024 * 1024  # split to 10 MB files

    # states
    running = True
    recording = False
    create_new_file = True

    # window name
    window_name = time.strftime("%Y.%m.%d  %H.%M.%S", time.localtime())

    # create VideoCapture
    camera_index = find_available_camera()
    if camera_index is None:
        print("\n" + "="*60)
        print("ERREUR: Aucune caméra détectée!")
        print("="*60)
        print("Vérifiez que:")
        print("  - Votre caméra est branchée")
        print("  - Les permissions sont correctes")
        print("  - Aucune autre application n'utilise la caméra")
        print("\nPour lister les caméras disponibles sous Linux:")
        print("  $ ls -l /dev/video*")
        print("  $ v4l2-ctl --list-devices")
        print("="*60 + "\n")
        return

    vcap = cv2.VideoCapture(camera_index)

    # check if video capturing has been initialized already
    if not vcap.isOpened():
        print("ERROR INITIALIZING VIDEO CAPTURE")
        return
    
    print("OK INITIALIZING VIDEO CAPTURE")

    # get vcap property
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30.0

    print("VCAP width :", width)
    print("VCAP height:", height)
    print("VCAP fps   :", fps)

    video_dir = Path(__file__).parent / "VideoRecorded"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    fout = None

    try:
        while running:
            # grab, decode and return the next video frame (and "return" status)
            ret, frame = vcap.read()
            
            if not ret:
                print("Erreur lecture frame")
                time.sleep(0.1)
                continue

            # write the next video frame
            if recording:
                if create_new_file:
                    # VideoWriter constructors
                    filename = video_dir / (
                        time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime()) + ".avi"
                    )

                    fourcc = cv2.VideoWriter_fourcc(*"MP42")  # .avi
                    fout = cv2.VideoWriter(str(filename), fourcc, fps, (width, height))
                    create_new_file = False

                    # check if video writer has been successfully initialized
                    if not fout.isOpened():
                        print("ERROR INITIALIZING VIDEO WRITER")
                        break
                    else:
                        print("OK INITIALIZING VIDEO WRITER")

                # write frame to file
                if fout and fout.isOpened():
                    fout.write(frame)

                # check file size
                if filename.stat().st_size >= VIDEO_FILE_SIZE:
                    fout.release()  # close current file
                    create_new_file = True  # time to create new file in next loop

                # add REC to frame
                cv2.putText(frame, "REC", (40, 40), FONT, 3, (0, 0, 255), 2)
                cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)

            # add instruction to frame
            cv2.putText(
                frame, "R - START RECORDING", (width - 200, 20), FONT, 1, (255, 255, 255)
            )
            cv2.putText(
                frame, "S - STOP RECORDING", (width - 200, 40), FONT, 1, (255, 255, 255)
            )
            cv2.putText(frame, "Q - QUIT", (width - 200, 60), FONT, 1, (255, 255, 255))
            cv2.putText(frame, "ESC - QUIT", (width - 200, 80), FONT, 1, (255, 255, 255))

            # displays an image in the specified window (with menu)
            cv2.imshow(window_name + " (with menu)", frame)

            # get key (get only lower 8-bits to work with chars)
            key = cv2.waitKey(1) & 0xFF

            # check what to do
            if key == KEY_R and not recording:
                print("START RECORDING")
                recording = True
                create_new_file = True
            elif key == KEY_S and recording:
                print("STOP RECORDING")
                recording = False
                create_new_file = False
                if fout:
                    fout.release()
            elif key == KEY_Q or key == KEY_ESC:
                print("EXIT")
                running = False

    except KeyboardInterrupt:
        print("\n[STOP] Interruption clavier...")
    except Exception as e:
        print(f"\n[ERROR] Erreur: {e}")
    finally:
        # Release everything
        if fout:
            fout.release()
        vcap.release()
        cv2.destroyAllWindows()
        print("[OK] Ressources libérées")


def main():
    """Main function for standalone execution."""
    run_register_mode()


if __name__ == "__main__":
    main()
