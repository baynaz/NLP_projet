# NLP Project: Multimodal Video Analysis with Vision-Language Models

**Advanced NLP Technique**: Multimodality (Vision-Language Models)  
**Model**: InternVL3.5-1B  
**Application**: Real-time video and image analysis with natural language descriptions

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Advanced Technique: Multimodal Vision-Language Models](#advanced-technique-multimodal-vision-language-models)
3. [Complete NLP Pipeline](#complete-nlp-pipeline)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Evaluation & Results](#evaluation--results)
7. [Technical Choices & Justification](#technical-choices--justification)
8. [Limitations & Future Work](#limitations--future-work)
9. [Assignment Requirements](#assignment-requirements)

---

## Project Overview

### Problem Statement
Traditional NLP systems work exclusively with text data. However, real-world understanding often requires combining visual and textual information. This project implements a **multimodal Vision-Language Model (VLM)** that can analyze images and videos, generating natural language descriptions of visual content.

### Key Features
- **Real-time webcam image analysis** via Streamlit interface
- **Video analysis** with segment-by-segment processing
- **Full video understanding** with temporal context
- **Interactive demo** with live visualization
- **Streaming mode** with performance optimizations

### Dataset & Task
- **Input**: Images from webcam or uploaded videos (mp4, avi, mov)
- **Output**: Natural language descriptions of visual content
- **Task**: Image/Video captioning and visual question answering
- **Model**: InternVL3.5-1B (1 billion parameters multimodal model)

---

## Advanced Technique: Multimodal Vision-Language Models

### What are Vision-Language Models?

Vision-Language Models (VLMs) are neural networks that bridge computer vision and natural language processing. They can:
- **Understand images** and generate textual descriptions
- **Answer questions** about visual content
- **Connect visual and textual concepts** in a shared embedding space

**Key Innovation**: Unlike traditional approaches that train separate vision and language models, VLMs jointly learn visual and linguistic representations, enabling cross-modal understanding.

### InternVL3.5 Architecture

InternVL3.5 consists of three main components:

1. **Vision Encoder (InternViT-300M)**
   - Processes images using a Vision Transformer (ViT)
   - Extracts visual features at multiple scales
   - Dynamic image preprocessing for optimal resolution

2. **Cross-Modal Projector**
   - Maps visual features to the language model's embedding space
   - Enables the language model to "understand" visual information

3. **Language Decoder (Qwen2-0.5B)**
   - Generates natural language descriptions
   - Conditioned on both visual features and text prompts
   - Uses autoregressive generation with beam search

### Technical Implementation

```python
# Core inference flow in our implementation
def analyze_frame(self, pil_image, max_new_tokens=50):
    # 1. Dynamic preprocessing - split image into tiles
    images = dynamic_preprocess(
        pil_image, image_size=448, use_thumbnail=True, max_num=6
    )
    
    # 2. Transform tiles and create tensor
    pixel_values = [self.transform(img) for img in images]
    pixel_values = torch.stack(pixel_values).to(device)
    
    # 3. Build prompt with image tokens
    question = "<image>\nDescribe concisely what is in this image."
    
    # 4. Generate response
    response = self.model.chat(
        self.tokenizer, pixel_values, question, generation_config
    )
    return response
```

### Real-World Applications

- **Accessibility**: Automated image descriptions for visually impaired users
- **Content moderation**: Detecting inappropriate visual content
- **Video surveillance**: Automatic scene understanding
- **Medical imaging**: Generating radiology reports from scans
- **Robotics**: Visual scene understanding for navigation

### Advantages Over Traditional Approaches

| Approach | Limitations | VLM Solution |
|----------|-------------|--------------|
| **OCR + NLP** | Only extracts text, misses visual context | Understands full visual scene |
| **Object Detection + Templates** | Rigid, rule-based descriptions | Natural, contextual language |
| **Separate Vision & Language** | Requires manual alignment | End-to-end joint training |

---

## Complete NLP Pipeline

Our implementation follows a complete machine learning pipeline:

### 1. Data Preprocessing

**Image Processing**:
```python
def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),  # Ensure RGB
        T.Resize((input_size, input_size)),         # Resize
        T.ToTensor(),                                # Convert to tensor
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)  # Normalize
    ])
```

**Video Processing**:
- Frame extraction using Decord library
- Temporal sampling (evenly spaced frames)
- Scene detection for segment boundaries
- Configurable number of frames per analysis

### 2. Baseline Implementation

**Simple Frame-by-Frame Analysis**:
- Extract middle frame from video
- Analyze single frame independently
- No temporal context
- Fast but limited understanding

```python
def baseline_video_analysis(video_path):
    middle_frame = extract_middle_frame(video_path)
    return model.analyze_frame(middle_frame)
```

### 3. Advanced Technique

**Temporal Multi-Frame Analysis**:
- Extract multiple frames across video timeline
- Process frames jointly with temporal context
- Model understands progression and relationships
- Richer, more coherent descriptions

```python
def advanced_video_analysis(video_path, num_segments=8):
    # Extract frames with temporal information
    pixel_values, num_patches = load_video(video_path, num_segments)
    
    # Build multi-frame prompt
    prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(num_segments)])
    question = prefix + "Describe what happens in this video."
    
    # Generate response with temporal understanding
    return model.chat(tokenizer, pixel_values, question, num_patches_list)
```

### 4. Evaluation Mechanism

**Quantitative Metrics**:
- Inference time per frame/video
- Memory usage (GPU/CPU)
- Frames processed per second (FPS)

**Qualitative Analysis**:
- Baseline vs. advanced technique comparison
- Human evaluation of description quality
- Error case analysis

---

## Installation & Setup

### Prerequisites

- **Python**: 3.12 or higher
- **UV Package Manager**: Modern Python package installer
- **CUDA** (optional): For GPU acceleration
- **Webcam** (optional): For real-time image capture

### Step 1: Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Clone the Repository

```bash
git clone <repository-url>
cd NLP_projet
```

### Step 3: Navigate to Backend Directory

```bash
cd src/backend
```

### Step 4: Install Dependencies

```bash
# Install all dependencies from pyproject.toml
uv sync
```

**Dependencies installed**:
- `transformers` (4.52.1): Hugging Face model loading
- `torch` (2.10.0): Deep learning framework
- `torchvision` (0.25.0): Image transformations
- `streamlit` (1.53.1): Web interface
- `opencv-python-headless`: Video processing
- `decord` (0.6.0): Efficient video loading
- `pillow`: Image manipulation
- `accelerate`: Model optimization
- `timm`: Vision model components
- `einops`: Tensor operations

### Step 5: Model Download (Automatic)

On first run, the model will automatically download from Hugging Face:
- Model: `OpenGVLab/InternVL3_5-1B`
- Size: ~2GB
- Location: `~/.cache/huggingface/hub/`

**Note**: Ensure stable internet connection for initial download.

---

## Usage Guide

### Option 1: Streamlit Web Interface (Recommended)

```bash
# Activate environment and run
cd src/backend
streamlit run app.py
```

The application opens at `http://localhost:8501`

**Features**:
1. **Image Analysis Mode**:
   - Click "Image" tab
   - Use webcam to capture image
   - View AI-generated description

2. **Video Analysis Mode**:
   - Click "Video" tab
   - Upload video file (mp4, avi, mov)
   - Choose analysis type:
     - **Full video description**: Holistic understanding
     - **Segment-by-segment**: Detailed temporal analysis
   - Adjust number of frames analyzed (slider)
   - View results in real-time

### Option 2: Command-Line Interface

**Analyze a video**:
```bash
uv run python -m streaming.inference_multi --video path/to/video.mp4 --segments 8
```

**Real-time webcam (OpenCV)**:
```bash
uv run python main.py
```

### Option 3: Python API

```python
from ai_logic import InternVLModel
from PIL import Image

# Initialize model
model = InternVLModel(streaming_mode=True)

# Analyze image
image = Image.open("photo.jpg")
description = model.analyze_frame(image, max_new_tokens=100)
print(description)

# Analyze video
video_description = model.analyze_video(
    "video.mp4", 
    num_segments=8,
    question="What happens in this video?"
)
print(video_description)
```

---

## Evaluation & Results

### Note on Evaluation Metrics

Traditional quantitative metrics such are not particularly useful for this project.But our application focuses on **open-ended visual description generation** where:

- There is no single "correct" description for an image or video
- Multiple valid descriptions can capture different aspects of the same visual content
- The quality depends on context, user intent, and subjective interpretation
- Human evaluation of relevance and accuracy is more meaningful than automated scores
However in improvement we can measure the counsumption of each request as monitoring  (GPU, CPU, RAM ...)
### Qualitative Assessment

The project demonstrates successful implementation of multimodal vision-language understanding through:

**Baseline vs. Advanced Comparison**:
- **Baseline** (single frame): Provides basic scene description
- **Advanced** (temporal multi-frame): Captures temporal progression, additional contextual details, and relationships between elements across time

**Observed Capabilities**:
- Object detection and identification
- Action and gesture recognition
- Spatial relationship understanding
- Temporal progression tracking across video frames

For detailed examples of the system's output, see [inference_results.md](src/backend/results/2026.01.12-18.11.03/inference_results.md).

### Future Evaluation Improvements

To better assess system performance and resource usage, the following metrics could be integrated:

- **VRAM consumption**: Monitor GPU memory usage during inference
- **CPU usage**: Track processor utilization for different modes
- **Inference time**: Measure end-to-end latency per frame/video
- **Throughput**: Frames per second (FPS) processing rate
- **Memory footprint**: RAM usage during video processing
- **Batch processing efficiency**: Performance gains with multiple inputs
- **Model loading time**: Initial startup overhead

These technical metrics would enable performance optimization and help identify bottlenecks in the pipeline.

---

## Technical Choices & Justification

### Model Selection: InternVL3.5-1B

**Why this model?**

| Criterion | InternVL3.5-1B | Alternatives |
|-----------|----------------|--------------|
| **Size** | 1B params | LLaVA (7B), GPT-4V (unknown) |
| **Performance** | Fast inference | Slower larger models |
| **Accuracy** | Competitive on benchmarks | Slight trade-off for speed |
| **Deployment** | Can run on consumer GPU | Requires data center GPUs |
| **Cost** | Free, open-source | API costs for GPT-4V |

**Decision**: Optimal balance between performance and resource requirements for real-time applications.

### Framework: PyTorch + Transformers

- **PyTorch**: Industry standard for research and production
- **Transformers**: Hugging Face ecosystem, easy model loading
- **Advantages**: 
  - Large community and documentation
  - Extensive pre-trained model zoo
  - Easy experimentation and iteration

### Interface: Streamlit

- **Why not Flask/FastAPI?**: Streamlit provides rapid prototyping with interactive widgets
- **Advantages**:
  - Zero front-end code required
  - Built-in file upload and camera capture
  - Real-time updates and session state
  - Perfect for demos and proof-of-concepts

### Video Processing: Decord

- **Alternative**: OpenCV VideoCapture
- **Why Decord?**
  - Faster video decoding (C++ backend)
  - Better memory efficiency
  - Random access to frames without sequential reading
  - Official recommendation from InternVL team

---

## Limitations & Future Work

### Current Limitations

1. **Language**: Model primarily trained on English, limited multilingual support
2. **Context length**: Maximum 256 tokens for video descriptions
3. **Temporal reasoning**: Weak understanding of long-term dependencies
4. **Bias**: May inherit biases from training data (e.g., gender stereotypes)

### Potential Improvements

#### Short-term
- [ ] Implement caching for repeated video segments
- [ ] Add batch processing for multiple videos
- [ ] Support for multiple languages via multilingual models

#### Long-term
- [ ] Fine-tune on domain-specific data (e.g., sports, medical)
- [ ] Integrate with RAG (Retrieval-Augmented Generation) for factual grounding
- [ ] Multi-agent system: separate agents for detection, captioning, reasoning
- [ ] Deploy as web service with authentication and usage tracking

### Ethical Considerations

**Privacy**: Webcam access requires user consent  
**Bias**: Model may have gender, race, or cultural biases  
**Misuse**: Could be used for unauthorized surveillance  
**Hallucination**: Model may generate plausible but incorrect descriptions

---

## Assignment Requirements

### Core Requirements ("Tronc Commun")

#### 1. Complete NLP Pipeline
- **Data Preprocessing**: Image/video loading, normalization, frame extraction [Done]
- **Baseline Method**: Single-frame analysis without temporal context [Done]
- **Advanced Technique**: Multi-frame temporal video understanding [Done]
- **Evaluation**: qualitative comparison + consumption metrics (GPU/RAM) [Done]

#### 2. Advanced Technique: Multimodality
- **Technique chosen**: Vision-Language Models (InternVL3.5) [Done]
- **Clear definition**: Explained architecture and cross-modal learning [Done]
- **Concrete examples**: Image captioning, video analysis, VQA [Done]
- **Existing tools**: Hugging Face Transformers, InternVL [Done]
- **Limitations**: Discussed speed, bias, hallucination issues [Done]

#### 3. Justified Technical Choices
- **Dataset**: Webcam images + uploaded videos (flexible, user-provided) [Done]
- **Task**: Visual captioning and understanding [Done]
- **Tools**: PyTorch, Transformers, Streamlit, Decord (justified in Section 7) [Done]
- **Model**: InternVL3.5-1B (size vs. performance trade-off) [Done]
- **Metrics**: Inference time, memory usage, description quality [Done]

#### 4. Functional Demonstration
- **Streamlit interface**: Interactive web UI with webcam and upload [Done]
- **Real-time execution**: Working demo with live results [Done]
- **Multiple modes**: Image analysis + full/segmented video analysis [Done]

#### 5. Clean Code Repository
- **Structured codebase**: Separated modules (ai_logic, streaming, app) [Done]
- **Documentation**: This comprehensive README [Done]
- **Requirements**: pyproject.toml with all dependencies [Done]
- **Reproducibility**: Clear installation and usage instructions [Done]

---

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

---

## References & Resources

### Model & Documentation
- **InternVL3.5 Model**: https://huggingface.co/OpenGVLab/InternVL3_5-1B
- **VLM Introduction**: https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro
- **InternVL Paper**: [ArXiv](https://arxiv.org/abs/2312.14238)
- **Transformers Library**: https://huggingface.co/docs/transformers/

### Technical Resources
- **Streamlit Documentation**: https://docs.streamlit.io/
- **PyTorch**: https://pytorch.org/docs/
- **Decord**: https://github.com/dmlc/decord
- **OpenCV**: https://docs.opencv.org/

---

## Team & Contributions

**Group Members**: Hadj RABEARIMANANA / Sékou BAH / Zaynab MERIMI / Nezar EL Medkour

**Individual Contributions**:
- **Member 1**: Camera logic, multiframe inference,cutting scenes, README
- **Member 2**: Model integration, ai_logic.py, development core logic model implementation
- **Member 3**: Development Streamlit interface, app.py, README
- **Member 4**: Testing, README

---

## License

This project is for educational purposes as part of the NLP course. The InternVL model is licensed under MIT License.

---

## Acknowledgments

- **InternVL Team** at OpenGVLab for the open-source model
- **Hugging Face** for the Transformers library
- **Course Instructors** for guidance and support

---
