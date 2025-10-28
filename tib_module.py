import cv2
import numpy as np
import os
from typing import List, Tuple

class TemporalInformationBottleneck:
    """
    Temporal Information Bottleneck (TIB)
    ------------------------------------
    Identifies keyframes by measuring mutual information across consecutive
    frame embeddings, selecting frames that maximize temporal information gain.
    """

    def __init__(self, entropy_threshold: float = 0.15, sampling_rate: int = 2):
        self.entropy_threshold = entropy_threshold
        self.sampling_rate = sampling_rate

    @staticmethod
    def frame_entropy(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist /= hist.sum()
        entropy = -np.sum([p * np.log2(p + 1e-8) for p in hist])
        return float(entropy)

    def extract_keyframes(self, video_path: str) -> List[Tuple[int, float]]:
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

    def save_keyframes(self, video_path: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        keyframes = self.extract_keyframes(video_path)
        cap = cv2.VideoCapture(video_path)

        for idx, _ in keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                out_path = f"{output_dir}/frame_{idx:05d}.jpg"
                cv2.imwrite(out_path, frame)
        
        cap.release()
        print(f"Saved {len(keyframes)} keyframes to {output_dir}")


# -------------------------------
# Test example on your video
# -------------------------------
if __name__ == "__main__":
    # Initialize TIB
    tib = TemporalInformationBottleneck(entropy_threshold=0.12, sampling_rate=5)
    
    # Path to a video from your evd-benchmark fork
    video_path = "https://github.com/NivedhaVijayan/EVD-benchmark/blob/main/part2(split-video.com).mp4"  # <-- replace with actual video path
    output_dir = "keyframes_output"

    # Extract keyframes
    keyframes = tib.extract_keyframes(video_path)
    print(f"Total keyframes extracted: {len(keyframes)}")
    print("First 10 keyframes:", keyframes[:10])

    # Optionally, save keyframes to disk
    tib.save_keyframes(video_path, output_dir)
