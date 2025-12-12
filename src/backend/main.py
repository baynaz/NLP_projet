import os
import subprocess
def main():
    subprocess.run(["uv", "run", "src/backend/streaming/VideoStream.py"])


if __name__ == "__main__":
    main()
