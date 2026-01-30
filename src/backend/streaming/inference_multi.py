import os
import sys
import json
from pathlib import Path
from typing import List, Dict
from datetime import timedelta
import cv2
from PIL import Image

# Add parent directory to path to allow imports from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .CutScenes import split_video
from ai_logic import InternVLModel


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    return str(timedelta(seconds=int(seconds)))


def open_video(video_path: str) -> cv2.VideoCapture:
    """Open a video file."""
    return cv2.VideoCapture(video_path)


def get_video_properties(cap: cv2.VideoCapture) -> tuple:
    """Extract FPS and frame count from video capture."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, frame_count


def calculate_duration(fps: float, frame_count: int) -> float:
    """Calculate video duration from FPS and frame count."""
    return frame_count / fps if fps > 0 else 0.0


def get_video_duration(video_path: str) -> float:
    """Get the duration of a video in seconds."""
    cap = open_video(video_path)
    if not cap.isOpened():
        return 0.0
    fps, frame_count = get_video_properties(cap)
    cap.release()
    return calculate_duration(fps, frame_count)


def get_segment_name(video_path: str) -> str:
    """Extract filename from path."""
    return Path(video_path).name


def extract_middle_frame(video_path: str) -> Image.Image:
    """Extract the middle frame from video."""
    cap = open_video(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to extract the middle frame from video: {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def extract_sample_frames(video_path: str, num_frames: int = 3) -> List[Image.Image]:
    """Extract evenly spaced frames from video."""
    cap = open_video(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [
        int(i * frame_count / (num_frames + 1)) for i in range(1, num_frames + 1)
    ]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def analyze_frame_with_model(model: InternVLModel, frame: Image.Image) -> str:
    """Analyze single frame with AI model."""
    return model.analyze_frame(frame)


def combine_frame_analyses(analyses: List[str]) -> str:
    """Combine multiple frame analysis results."""
    if not analyses:
        return "No frames analyzed"
    return " | ".join(
        [f"Frame {i + 1}: {analysis}" for i, analysis in enumerate(analyses)]
    )


def inference_on_segment(video_segment_path: str, model: InternVLModel = None) -> str:
    """Run inference on a single video segment using direct video analysis."""
    if model is None:
        return "Model not initialized"
    
    try:
        # Use the new analyze_video method instead of extracting individual frames
        analysis = model.analyze_video(
            video_segment_path, 
            num_segments=4,  # Extract 4 frames from the segment
            question="Describe what is happening in this video segment."
        )
        return analysis
    except Exception as e:
        return f"Error analyzing segment: {e}"


def analyze_full_video(video_path: str, model: InternVLModel = None, num_segments: int = 8) -> str:
    """Analyze the full video without splitting into segments."""
    if model is None:
        return "Model not initialized"
    
    try:
        print(f"Analyzing full video: {video_path}")
        analysis = model.analyze_video(
            video_path,
            num_segments=num_segments,
            question="Provide a detailed description of what happens throughout this video."
        )
        return analysis
    except Exception as e:
        return f"Error analyzing full video: {e}"


def analyze_video_by_time_segments(video_path: str, time_segments: List[tuple], model: InternVLModel = None) -> List[str]:
    """Analyze specific time segments of a video."""
    if model is None:
        return ["Model not initialized"] * len(time_segments)
    
    results = []
    for i, (start_time, end_time) in enumerate(time_segments):
        try:
            print(f"Analyzing time segment {i+1}: {start_time:.1f}s - {end_time:.1f}s")
            analysis = model.analyze_video_segment(
                video_path,
                start_time=start_time,
                end_time=end_time,
                num_segments=4,
                question="Describe what is happening in this video segment."
            )
            results.append(analysis)
        except Exception as e:
            results.append(f"Error analyzing segment {i+1}: {e}")
    
    return results


def print_processing_info(video_path: str, num_segments: int):
    """Print video processing information."""
    print(f"Processing video: {video_path}")
    print(f"Splitting into {num_segments} segments...\n")


def split_video_safe(video_path: str, num_segments: int, output_dir: str) -> List[str]:
    """Split video with error handling."""
    try:
        return split_video(video_path, num_segments, output_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to split video: {e}")


def calculate_segment_duration(total_duration: float, num_segments: int) -> float:
    """Calculate duration per segment."""
    return total_duration / num_segments


def calculate_time_range(
    idx: int, segment_duration: float, num_segments: int, total_duration: float
) -> tuple:
    """Calculate start and end time for a segment."""
    start_time = (idx - 1) * segment_duration
    end_time = idx * segment_duration if idx < num_segments else total_duration
    return start_time, end_time


def create_result_entry(
    idx: int, segment_path: str, start_time: float, end_time: float, inference_text: str
) -> Dict:
    """Create a result dictionary entry."""
    return {
        "segment_number": idx,
        "segment_file": segment_path,
        "time_range": f"{format_timestamp(start_time)} - {format_timestamp(end_time)}",
        "start_seconds": start_time,
        "end_seconds": end_time,
        "inference_text": inference_text,
    }


def process_single_segment(
    idx: int,
    segment_path: str,
    segment_duration: float,
    num_segments: int,
    total_duration: float,
    model: InternVLModel,
) -> Dict:
    """Process a single video segment."""
    print(f"Running inference on segment {idx}/{num_segments}...")
    start_time, end_time = calculate_time_range(
        idx, segment_duration, num_segments, total_duration
    )
    inference_text = inference_on_segment(segment_path, model)
    return create_result_entry(idx, segment_path, start_time, end_time, inference_text)


def handle_segment_error(
    idx: int,
    segment_path: str,
    segment_duration: float,
    num_segments: int,
    total_duration: float,
    error: Exception,
) -> Dict:
    """Handle error during segment processing."""
    print(f"Error processing segment {idx}: {error}")
    start_time, end_time = calculate_time_range(
        idx, segment_duration, num_segments, total_duration
    )
    return create_result_entry(
        idx, segment_path, start_time, end_time, f"ERROR: {str(error)}"
    )


def initialize_model() -> InternVLModel:
    """Initialize the AI model."""
    print("Initializing AI model...")
    return InternVLModel()


def process_all_segments(
    segment_paths: List[str],
    segment_duration: float,
    num_segments: int,
    total_duration: float,
    model: InternVLModel,
) -> List[Dict]:
    """Process all video segments with inference."""
    results = []
    for idx, segment_path in enumerate(segment_paths, 1):
        try:
            result = process_single_segment(
                idx, segment_path, segment_duration, num_segments, total_duration, model
            )
        except Exception as e:
            result = handle_segment_error(
                idx, segment_path, segment_duration, num_segments, total_duration, e
            )
        results.append(result)
    return results


def delete_segment_file(segment_path: str):
    """Delete a single segment file."""
    try:
        os.remove(segment_path)
    except Exception as e:
        print(f"Warning: Could not delete {segment_path}: {e}")


def cleanup_segment_files(segment_paths: List[str]):
    """Clean up temporary segment files."""
    for segment_path in segment_paths:
        delete_segment_file(segment_path)


def process_video_with_inference(
    video_path: str,
    num_segments: int,
    output_dir: str = "segments",
    cleanup_segments: bool = False,
    model: InternVLModel = None,
) -> List[Dict[str, str]]:
    """Split video into segments and run inference on each segment."""
    print_processing_info(video_path, num_segments)
    if model is None:
        model = initialize_model()
    segment_paths = split_video_safe(video_path, num_segments, output_dir)
    total_duration = get_video_duration(video_path)
    segment_duration = calculate_segment_duration(total_duration, num_segments)
    results = process_all_segments(
        segment_paths, segment_duration, num_segments, total_duration, model
    )
    if cleanup_segments:
        cleanup_segment_files(segment_paths)
    return results


def format_as_json(results: List[Dict[str, str]]) -> str:
    """Format results as JSON."""
    return json.dumps(results, indent=2)


def format_markdown_header() -> str:
    """Create markdown header."""
    return "# Video Inference Results\n\n"


def format_segment_markdown(result: Dict) -> str:
    """Format single segment as markdown."""
    output = f"## Segment {result['segment_number']}\n"
    output += f"**Time Range:** {result['time_range']}\n\n"
    output += f"**Inference Result:**\n{result['inference_text']}\n\n---\n\n"
    return output


def format_as_markdown(results: List[Dict[str, str]]) -> str:
    """Format results as markdown."""
    output = format_markdown_header()
    for result in results:
        output += format_segment_markdown(result)
    return output


def format_text_header() -> str:
    """Create text format header."""
    return "=" * 80 + "\n" + "VIDEO INFERENCE RESULTS\n" + "=" * 80 + "\n\n"


def format_segment_text(result: Dict) -> str:
    """Format single segment as text."""
    output = f"Segment {result['segment_number']}\n"
    output += f"Time Range: {result['time_range']}\n"
    output += f"{'-' * 40}\n{result['inference_text']}\n\n"
    return output + "=" * 80 + "\n\n"


def format_as_text(results: List[Dict[str, str]]) -> str:
    """Format results as plain text."""
    output = format_text_header()
    for result in results:
        output += format_segment_text(result)
    return output


def format_results(results: List[Dict[str, str]], output_format: str = "text") -> str:
    """Format inference results in a readable format."""
    if output_format == "json":
        return format_as_json(results)
    elif output_format == "markdown":
        return format_as_markdown(results)
    return format_as_text(results)


def write_to_file(output_file: str, content: str):
    """Write content to file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)


def save_results(
    results: List[Dict[str, str]], output_file: str, output_format: str = "text"
):
    """Save inference results to a file."""
    formatted_output = format_results(results, output_format)
    write_to_file(output_file, formatted_output)
    print(f"Results saved to: {output_file}")


def main():
    """Main function for command-line execution."""
    # Example usage
    # Use path relative to this script's location
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    results_base_dir = backend_dir / "results"
    results_base_dir.mkdir(exist_ok=True)
    
    video_file = str(script_dir / "VideoRecorded" / "2026.01.12-18.11.03.avi")
    segments = 4
    
    # Create a folder for this specific video's results
    video_name = Path(video_file).stem  # Get filename without extension
    results_dir = results_base_dir / video_name
    results_dir.mkdir(exist_ok=True)

    try:
        # Method 1: Traditional segmented analysis (existing approach)
        print("=== TRADITIONAL SEGMENTED ANALYSIS ===")
        results = process_video_with_inference(
            video_path=video_file,
            num_segments=segments,
            output_dir=str(results_dir / "segments_output"),
            cleanup_segments=False,
        )
        print("\n" + format_results(results, output_format="text"))

        # Method 2: Direct full video analysis (new approach)
        print("\n=== DIRECT FULL VIDEO ANALYSIS ===")
        model = initialize_model()
        full_analysis = analyze_full_video(video_file, model, num_segments=8)
        print(f"Full video analysis:\n{full_analysis}")

        # Method 3: Time-based segment analysis (new approach)
        print("\n=== TIME-BASED SEGMENT ANALYSIS ===")
        duration = get_video_duration(video_file)
        segment_duration = duration / 3  # Divide into 3 time segments
        time_segments = [
            (0, segment_duration),
            (segment_duration, 2 * segment_duration),
            (2 * segment_duration, duration)
        ]
        time_analyses = analyze_video_by_time_segments(video_file, time_segments, model)
        for i, analysis in enumerate(time_analyses):
            start, end = time_segments[i]
            print(f"\nSegment {i+1} ({start:.1f}s - {end:.1f}s):\n{analysis}")

        # Save all results under results/ folder
        save_results(results, str(results_dir / "inference_results.txt"), output_format="text")
        save_results(results, str(results_dir / "inference_results.md"), output_format="markdown")
        save_results(results, str(results_dir / "inference_results.json"), output_format="json")
        
        # Save new analysis results
        with open(results_dir / "full_video_analysis.txt", "w", encoding="utf-8") as f:
            f.write(f"Full Video Analysis:\n{full_analysis}\n\n")
            f.write("Time-based Segment Analysis:\n")
            for i, analysis in enumerate(time_analyses):
                start, end = time_segments[i]
                f.write(f"\nSegment {i+1} ({start:.1f}s - {end:.1f}s):\n{analysis}\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
