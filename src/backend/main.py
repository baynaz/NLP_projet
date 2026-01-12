import os
import subprocess
KEY_V = ord('v') # just show camera
KEY_I = ord('i') # register video like input

def choose_mode(key: str) -> None:
    match key:
        case 'v':
            subprocess.run(["uv", "run", "src/backend/streaming/VideoStream.py"])
        case 'i':
            subprocess.run(["uv", "run", "src/backend/streaming/RegisterStream.py"])
        case _:
            raise ValueError(f"Unrecognized key: {key}")
def main():
    ask_question_string_before_beginning: str = """
        Please select a streaming mode by pressing the corresponding key:
        [v] View only: Display the camera feed.
        [i] Record: Display and record the camera feed.
    """
    key_mode: str = str(input(ask_question_string_before_beginning))
    choose_mode(key_mode)

if __name__ == "__main__":
    main()
