# NLP_projet

# Python Installation dependencies
if uv is not install use this :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
then in the root project type this
uv sync
```
wait until everything is finished that's all

## Camera Troubleshooting

If you encounter "Cannot open camera" errors:

1. **Check camera connection**: Ensure your camera is properly connected
2. **Check camera availability**: Make sure no other application is using the camera
3. **Try different camera indices**: The code will automatically try indices 0, 1, 2, and -1
4. **Permissions**: On Linux, you might need to add your user to the video group:
   ```bash
   sudo usermod -aG video $USER
   ```
5. **Run with sudo**: Try running the application with sudo privileges
6. **Check camera devices**: List available video devices:
   ```bash
   ls /dev/video*
   ```
https://huggingface.co/OpenGVLab/InternVL3_5-1B
https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro
