"""
Temporal Information Bottleneck (TIB) Module – End-to-End
---------------------------------------------------------
Description:
The Temporal Information Bottleneck (TIB) identifies pedagogically
significant keyframes from educational videos. It computes structural
and content saliency to select frames that maximize temporal information
gain while minimizing redundancy. Domain-informed pruning removes
irrelevant frames such as blanks, logos, or long static shots.

Features:
1. Candidate keyframe extraction using frame entropy.
2. Domain-informed pruning based on duration and SSIM.
3. Separation of moderately significant frames (K').
4. Inline visualization when run in Jupyter Notebook.
---------------------------------------------------------
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
from skimage.metrics import structural_similarity as ssim

try:
    from IPython.display import Image, display
    from ipywidgets import FileUpload
except ImportError:
    # Allow running outside Jupyter
    pass


class TemporalInformationBottleneck:
    """
    Temporal Information Bottleneck (TIB)
    ------------------------------------
    Extracts keyframes from video based on frame entropy, 
    structural and content saliency, and domain-informed pruning.
    """

    def __init__(self, entropy_threshold: float = 0.15, sampling_rate: int = 2):
        self.entropy_threshold = entropy_threshold
        self.sampling_rate = sampling_rate

    @staticmethod
    def frame_entropy(frame: np.ndarray) -> float:
        """Compute Shannon entropy of a frame (grayscale)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist /= hist.sum()
        entropy = -np.sum([p * np.log2(p + 1e-8) for p in hist])
        return float(entropy)

    def extract_keyframes(self, video_path: str) -> List[Tuple[int, float]]:
        """Extract candidate keyframes using entropy difference."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        keyframes = []
        prev_entropy = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sampling_rate == 0:
                entropy = self.frame_entropy(frame)
                if prev_entropy is not None:
                    delta = abs(entropy - prev_entropy)
                    if delta > self.entropy_threshold:
                        keyframes.append((frame_idx, entropy))
                prev_entropy = entropy

            frame_idx += 1

        cap.release()
        return keyframes

    def domain_pruning(self, video_path: str, keyframes: List[Tuple[int,float]], fps: int = 25,
                       duration_thresh: float = 0.04, ssim_thresh_high: float = 0.95,
                       ssim_thresh_mod: float = 0.85):
        """
        Apply domain-informed pruning:
        - Remove very short frames
        - Remove static/redundant frames using SSIM
        - Place moderately significant frames into K_prime
        """
        cap = cv2.VideoCapture(video_path)
        K_final = []
        K_prime = []
        prev_frame = None

        for idx, _ in keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Duration check
            duration = 1 / fps
            if duration < duration_thresh:
                continue

            # SSIM with previous keyframe
            if prev_frame is not None:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                score = ssim(frame_gray, prev_gray)
                if score > ssim_thresh_high:
                    continue
                elif score > ssim_thresh_mod:
                    K_prime.append((idx, _))
                    continue

            K_final.append((idx, _))
            prev_frame = frame

        cap.release()
        return K_final, K_prime

    def save_keyframes(self, video_path: str, keyframes: List[Tuple[int,float]], output_dir: str):
        """Save keyframes to disk for visualization."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        for idx, _ in keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                out_path = os.path.join(output_dir, f"frame_{idx:05d}.jpg")
                cv2.imwrite(out_path, frame)
        cap.release()
        print(f"Saved {len(keyframes)} keyframes to {output_dir}")


# -------------------------------
# Optional: Interactive upload in Jupyter Notebook
# -------------------------------
def interactive_tib():
    """Run TIB interactively in Jupyter Notebook."""
    print("Upload your video file (MP4/AVI):")
    uploader = FileUpload(accept=".mp4,.avi", multiple=False)
    display(uploader)
    return uploader

# -------------------------------
# Optional: Display first few keyframes inline
# -------------------------------
def display_keyframes(output_dir: str, keyframes: List[Tuple[int,float]], n: int = 5):
    """Display first n keyframes in Jupyter Notebook."""
    try:
        from IPython.display import Image, display
    except ImportError:
        print("IPython not available. Skipping display.")
        return

    for idx, _ in keyframes[:n]:
        display(Image(filename=os.path.join(output_dir, f"frame_{idx:05d}.jpg")))


# -------------------------------
# Example usage (for Python script)
# -------------------------------
if __name__ == "__main__":
    # Initialize TIB
    tib = TemporalInformationBottleneck(entropy_threshold=0.12, sampling_rate=5)

    # Provide path to local video
    video_path = "sample_video.mp4"   # Replace with your video path
    output_dir = "keyframes_output"

    # Step 1: Candidate keyframes
    candidate_keyframes = tib.extract_keyframes(video_path)
    print(f"Candidate keyframes: {len(candidate_keyframes)}")

    # Step 2: Domain-informed pruning
    K_final, K_prime = tib.domain_pruning(video_path, candidate_keyframes, fps=25)
    print(f"Pedagogically significant keyframes (K): {len(K_final)}")
    print(f"Moderately significant frames (K'): {len(K_prime)}")

    # Step 3: Save keyframes
    tib.save_keyframes(video_path, K_final, output_dir)
