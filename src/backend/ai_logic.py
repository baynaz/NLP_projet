import numpy as np
import torch

# Patch torch.linspace to avoid meta tensor issues during model init
_original_linspace = torch.linspace
def _patched_linspace(*args, **kwargs):
    # Remove device argument and create on CPU first, then move if needed
    device = kwargs.pop('device', None)
    result = _original_linspace(*args, device='cpu', **kwargs)
    if device is not None and device != 'cpu':
        result = result.to(device)
    return result
torch.linspace = _patched_linspace

import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """Calculate frame indices for video sampling."""
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """Load video and extract frames for analysis."""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


class InternVLModel:
    def __init__(self, streaming_mode=False):
        self.path = "OpenGVLab/InternVL3_5-1B"
        self.streaming_mode = streaming_mode
        
        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"Chargement du modèle IA sur GPU ({torch.cuda.get_device_name(0)})")
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            print("Chargement du modèle IA sur CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.path, trust_remote_code=True, use_fast=False
        )
        
        # Load model - disable all memory optimization features that cause meta tensor issues
        load_kwargs = {
            "trust_remote_code": True,
            "use_flash_attn": False,
            "low_cpu_mem_usage": False,
        }
        
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = self.dtype
            load_kwargs["device_map"] = "auto"
        
        self.model = AutoModel.from_pretrained(self.path, **load_kwargs).eval()
        
        # Explicitly move to device if not using device_map
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        
        # Pre-build transform for streaming to avoid rebuilding
        self.transform = build_transform(input_size=448)
        
        # Warmup model for faster first inference
        if streaming_mode:
            self._warmup_model()
        
        print("IA Prête.")
    
    def _warmup_model(self):
        """Warmup model with dummy input for faster first inference."""
        try:
            print("Préchauffage du modèle...")
            dummy_img = Image.new('RGB', (448, 448), color='black')
            with torch.no_grad():
                self.analyze_frame(dummy_img, max_new_tokens=10)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Préchauffage terminé.")
        except Exception as e:
            print(f"Préchauffage échoué: {e}")

    def analyze_frame(self, pil_image, max_new_tokens=50):
        """Analyze a single frame with optimizations for streaming.
        
        Args:
            pil_image: PIL Image to analyze
            max_new_tokens: Maximum tokens to generate (lower = faster)
        """
        try:
            with torch.no_grad():  # Disable gradient computation for inference
                # For streaming mode, use simpler preprocessing (1 tile instead of 6)
                max_num = 1 if self.streaming_mode else 6
                
                images = dynamic_preprocess(
                    pil_image, image_size=448, use_thumbnail=True, max_num=max_num
                )
                pixel_values = [self.transform(img) for img in images]
                pixel_values = torch.stack(pixel_values).to(self.dtype).to(self.device)

                question = "<image>\nDescribe concisely what is in this image."
                generation_config = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1  # Use greedy decoding for speed
                )

                response = self.model.chat(
                    self.tokenizer, pixel_values, question, generation_config
                )
                
                # Clear GPU cache after inference in streaming mode
                if self.streaming_mode and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return response
        except Exception as e:
            return f"Erreur : {e}"

    def analyze_video(self, video_path, num_segments=8, question=None):
        """Analyze a video by extracting and processing multiple frames."""
        try:
            with torch.no_grad():  # Disable gradient computation
                # Load video frames
                pixel_values, num_patches_list = load_video(
                    video_path, num_segments=num_segments, max_num=1
                )
                pixel_values = pixel_values.to(self.dtype).to(self.device)

                # Build the question with frame references
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                if question is None:
                    question = "Describe what is happening in this video."
                full_question = video_prefix + question

                generation_config = dict(
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1  # Use greedy decoding for speed
                )

                # Get response
                response = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    full_question, 
                    generation_config,
                    num_patches_list=num_patches_list
                )
                
                # Clear cache after video processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return response
        except Exception as e:
            return f"Erreur : {e}"

    def analyze_video_segment(self, video_path, start_time=None, end_time=None, num_segments=8, question=None):
        """Analyze a specific segment of a video (by time in seconds)."""
        try:
            with torch.no_grad():  # Disable gradient computation
                bound = None
                if start_time is not None and end_time is not None:
                    bound = (start_time, end_time)
                
                # Load video frames for the segment
                pixel_values, num_patches_list = load_video(
                    video_path, bound=bound, num_segments=num_segments, max_num=1
                )
                pixel_values = pixel_values.to(self.dtype).to(self.device)

                # Build the question with frame references
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                if question is None:
                    question = "Describe what is happening in this video segment."
                full_question = video_prefix + question

                generation_config = dict(
                    max_new_tokens=256,
                    do_sample=False,
                    num_beams=1  # Use greedy decoding for speed
                )

                # Get response
                response = self.model.chat(
                    self.tokenizer, 
                    pixel_values, 
                    full_question, 
                    generation_config,
                    num_patches_list=num_patches_list
                )
                
                # Clear cache after segment processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return response
        except Exception as e:
            return f"Erreur : {e}"
