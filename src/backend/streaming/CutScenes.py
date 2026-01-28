import cv2
import os
from pathlib import Path


def split_video(
    video_path: str, num_segments: int, output_dir: str = "output_segments"
):
    """
    Split a video into multiple segments.

    Args:
        video_path (str): Path to the input video file
        num_segments (int): Number of segments to split the video into
        output_dir (str): Directory to save the split video segments

    Returns:
        list: List of paths to the created video segments
    """
    if num_segments <= 0:
        raise ValueError("Number of segments must be greater than 0")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Calculate frames per segment
    frames_per_segment = total_frames // num_segments

    if frames_per_segment == 0:
        raise ValueError(
            f"Video has only {total_frames} frames, cannot split into {num_segments} segments"
        )

    segment_paths = []
    video_name = Path(video_path).stem

    # Process each segment
    for segment_idx in range(num_segments):
        segment_filename = f"{video_name}_segment_{segment_idx + 1}.mp4"
        segment_path = output_path / segment_filename
        segment_paths.append(str(segment_path))

        # Create video writer for this segment
        out = cv2.VideoWriter(str(segment_path), fourcc, fps, (width, height))

        # Calculate start and end frames for this segment
        start_frame = segment_idx * frames_per_segment
        # For the last segment, include all remaining frames
        end_frame = (
            total_frames
            if segment_idx == num_segments - 1
            else (segment_idx + 1) * frames_per_segment
        )

        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Write frames to segment
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()
        print(f"Created segment {segment_idx + 1}/{num_segments}: {segment_path}")

    cap.release()
    return segment_paths


if __name__ == "__main__":
    # Example usage
    video_file = "input_video.mp4"
    segments = 4

    try:
        output_files = split_video(video_file, segments)
        print(f"\nSuccessfully split video into {len(output_files)} segments")
        for idx, file in enumerate(output_files, 1):
            print(f"  Segment {idx}: {file}")
    except Exception as e:
        print(f"Error: {e}")
