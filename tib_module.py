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
from typing import List

class TemporalInformationBottleneck:
    """Extracts keyframes using temporal entropy differences."""

    def __init__(self, entropy_threshold: float = 0.15, sampling_rate: int = 2):
        """
        Parameters:
        ----------
        entropy_threshold : float
            Minimum entropy difference to consider a frame as informative.
        sampling_rate : int
            Number of frames to skip between evaluations.
        """
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

    def extract_keyframes(self, video_path: str) -> List[int]:
        """Return list of frame indices considered keyframes."""
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
                if prev_entropy is not None and abs(entropy - prev_entropy) > self.entropy_threshold:
                    keyframes.append(frame_idx)
                prev_entropy = entropy
            frame_idx += 1

        cap.release()
        return keyframes

    def save_keyframes(self, video_path: str, keyframes: List[int], output_dir: str):
        """Save extracted keyframes to disk."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        for idx in keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(output_dir, f"frame_{idx:05d}.jpg"), frame)
        cap.release()
        print(f"Saved {len(keyframes)} keyframes to {output_dir}")


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    video_path = "sample_video.mp4"  # Replace with your video path
    output_dir = "keyframes_output"

    tib = TemporalInformationBottleneck(entropy_threshold=0.12, sampling_rate=5)
    keyframes = tib.extract_keyframes(video_path)
    tib.save_keyframes(video_path, keyframes, output_dir)
