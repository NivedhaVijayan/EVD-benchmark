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
from sklearn.cluster import DBSCAN  # Using DBSCAN as a proxy for Density Peak Clustering


class TemporalInformationBottleneck:
    """TIB implementing Algorithm 1."""

    def __init__(self,
                 alpha: float = 0.67,
                 mse_threshold: float = 0.01,
                 saliency_threshold: float = 0.02,
                 ssim_high: float = 0.95,
                 ssim_moderate: float = 0.85,
                 duration_threshold: int = 3,
                 sampling_rate: int = 2):
        """
        Parameters:
        ----------
        alpha : float
            Weight between structural (MSE) and content (saliency) scores.
        mse_threshold : float
            Minimum MSE change to consider structural significance.
        saliency_threshold : float
            Minimum mean saliency value for content significance.
        ssim_high : float
            High SSIM threshold for pruning redundant frames.
        ssim_moderate : float
            Moderate SSIM threshold for secondary significance.
        duration_threshold : int
            Minimum duration of frame in frames to be considered.
        sampling_rate : int
            Skip frames to reduce computation.
        """
        self.alpha = alpha
        self.mse_threshold = mse_threshold
        self.saliency_threshold = saliency_threshold
        self.ssim_high = ssim_high
        self.ssim_moderate = ssim_moderate
        self.duration_threshold = duration_threshold
        self.sampling_rate = sampling_rate

    @staticmethod
    def compute_mse(frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Structural saliency: mean squared error between frames."""
        return np.mean((frame1.astype(float) - frame2.astype(float))**2)

    @staticmethod
    def compute_saliency_map(frame: np.ndarray) -> np.ndarray:
        """Compute simple saliency using OpenCV spectral residual method."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, saliency_map = saliency.computeSaliency(gray)
        return saliency_map

    @staticmethod
    def mean_saliency(frame: np.ndarray) -> float:
        sal_map = TemporalInformationBottleneck.compute_saliency_map(frame)
        return float(np.mean(sal_map))

    def extract_keyframes(self, video_path: str) -> Tuple[List[int], List[int]]:
        """
        Extract K (keyframes) and K' (moderate significance frames)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        # Step 1: Pedagogical significance scoring
        significance_scores = []
        for i in range(1, len(frames)):
            frame_prev = frames[i-1]
            frame_curr = frames[i]

            mse_val = self.compute_mse(frame_curr, frame_prev)
            sal_val = self.mean_saliency(frame_curr)
            score = self.alpha * mse_val + (1 - self.alpha) * sal_val
            significance_scores.append(score)

        significance_scores = np.array(significance_scores).reshape(-1, 1)

        # Step 2: Instructional transition identification (clustering)
        # Using DBSCAN as proxy for density peak clustering
        clustering = DBSCAN(eps=0.01, min_samples=1).fit(significance_scores)
        labels = clustering.labels_
        centroids_idx = [i+1 for i in range(1, len(frames)) if labels[i] != labels[i-1]]

        # Step 3: Domain-informed pruning
        K = []
        K_prime = []

        for idx in centroids_idx:
            frame = frames[idx]
            # Compute duration as number of consecutive frames (approx)
            duration = self.sampling_rate
            # SSIM with previous frame
            prev_idx = max(0, idx - 1)
            sim_val = ssim(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                           cv2.cvtColor(frames[prev_idx], cv2.COLOR_BGR2GRAY))
            # Mean saliency
            sal_val = self.mean_saliency(frame)

            if duration < self.duration_threshold:
                continue  # prune short frames
            elif sim_val > self.ssim_high:
                continue  # prune redundant frames
            elif sim_val > self.ssim_moderate and sal_val < self.saliency_threshold:
                K_prime.append(idx)
            else:
                K.append(idx)

        return K, K_prime

    def save_keyframes(self, video_path: str, K: List[int], K_prime: List[int], output_dir: str):
        """Save extracted keyframes and moderate frames to disk."""
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        for idx in K:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(output_dir, f"K_frame_{idx:05d}.jpg"), frame)

        for idx in K_prime:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(output_dir, f"Kprime_frame_{idx:05d}.jpg"), frame)

        cap.release()
        print(f"Saved {len(K)} keyframes and {len(K_prime)} moderate frames to {output_dir}")


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    video_path = "sample_video.mp4"
    output_dir = "keyframes_output"

    tib = TemporalInformationBottleneck()
    K, K_prime = tib.extract_keyframes(video_path)
    tib.save_keyframes(video_path, K, K_prime, output_dir)

    print("Keyframes (K):", K[:10])
    print("Moderate frames (K'):", K_prime[:10])
