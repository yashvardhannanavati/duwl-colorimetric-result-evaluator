"""
Image Preprocessing Module

Enhances image quality before vial detection and color extraction.
Applies white balance correction, contrast enhancement, and noise reduction.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import yaml


class ImagePreprocessor:
    """Preprocesses images to improve color detection accuracy."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the image preprocessor.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get preprocessing settings or use defaults
        self.preprocessing_enabled = self.config.get('preprocessing', {}).get('enabled', True)
        self.white_balance_enabled = self.config.get('preprocessing', {}).get('white_balance', True)
        self.white_balance_method = self.config.get('preprocessing', {}).get('white_balance_method', 'gray_world')
        self.contrast_enhancement = self.config.get('preprocessing', {}).get('contrast_enhancement', True)
        self.noise_reduction = self.config.get('preprocessing', {}).get('noise_reduction', True)

        # Sharpening settings
        sharpening_config = self.config.get('preprocessing', {}).get('sharpening', {})
        self.sharpening_enabled = sharpening_config.get('enabled', False)
        self.sharpening_amount = sharpening_config.get('amount', 1.0)
        self.sharpening_kernel = sharpening_config.get('kernel_size', 2.0)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to image.

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed image
        """
        if not self.preprocessing_enabled:
            return image

        processed = image.copy()

        # Step 1: Denoise (slight blur to reduce sensor noise)
        if self.noise_reduction:
            processed = self._reduce_noise(processed)

        # Step 2: White balance correction (critical for color accuracy)
        if self.white_balance_enabled:
            if self.white_balance_method == 'background':
                processed = self._white_balance_from_background(processed)
            else:
                processed = self._white_balance(processed)

        # Step 3: Enhance contrast (improve color separation)
        if self.contrast_enhancement:
            processed = self._enhance_contrast(processed)

        # Step 4: Sharpen image (enhance edge definition)
        if self.sharpening_enabled:
            processed = self._sharpen(processed)

        return processed

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce image noise while preserving edges.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        # Use bilateral filter - reduces noise while keeping edges sharp
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)
        return denoised

    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction using Gray World algorithm.

        This is CRITICAL because gray backgrounds cause color shifts.
        The algorithm assumes the average color in the image should be gray,
        and adjusts color channels accordingly.

        Args:
            image: Input image (BGR)

        Returns:
            White-balanced image
        """
        # Gray World White Balance Algorithm
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])

        # Adjust color channels to make average neutral gray
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)

        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

        This helps separate subtle color differences (pink vs purple vs clear).

        Args:
            image: Input image

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel only (preserves color while enhancing brightness contrast)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return enhanced

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Sharpen image using unsharp mask technique.

        This enhances edges and fine details, which improves:
        - Vial detection accuracy (sharper edges = better contour detection)
        - Laplacian variance sharpness metrics
        - Visual clarity for analysis

        Args:
            image: Input image

        Returns:
            Sharpened image
        """
        # Create blurred version using Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), self.sharpening_kernel)

        # Unsharp mask: sharpened = original + amount * (original - blurred)
        sharpened = cv2.addWeighted(image, 1.0 + self.sharpening_amount,
                                     blurred, -self.sharpening_amount, 0)

        return sharpened

    def _white_balance_from_background(self, image: np.ndarray) -> np.ndarray:
        """
        White balance using known-white background areas from the white box.

        Samples pixels from image corners (which should be white background),
        filters to bright pixels, and scales channels so white becomes [255,255,255].

        Args:
            image: Input image (BGR)

        Returns:
            White-balanced image
        """
        h, w = image.shape[:2]
        margin_y = max(1, int(h * 0.05))
        margin_x = max(1, int(w * 0.05))

        # Sample from four corner regions
        corners = [
            image[0:margin_y, 0:margin_x],
            image[0:margin_y, w-margin_x:w],
            image[h-margin_y:h, 0:margin_x],
            image[h-margin_y:h, w-margin_x:w],
        ]
        samples = np.vstack([c.reshape(-1, 3) for c in corners])

        # Filter to bright pixels only (likely white background)
        bright_mask = np.all(samples > 150, axis=1)
        if np.sum(bright_mask) < 50:
            return image  # Not enough white pixels, skip correction

        white_samples = samples[bright_mask].astype(np.float64)
        mean_bgr = np.mean(white_samples, axis=0)

        # Scale so white reference becomes [255, 255, 255]
        scale = 255.0 / np.clip(mean_bgr, 1, 255)

        corrected = image.astype(np.float64) * scale
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _white_balance_simple_world(self, image: np.ndarray) -> np.ndarray:
        """
        Alternative white balance using simple gray world assumption.
        Scales each color channel so the average becomes 128 (neutral gray).

        Args:
            image: Input image (BGR)

        Returns:
            White-balanced image
        """
        result = image.copy().astype(np.float32)

        # Calculate average for each channel
        avg_b = np.average(result[:, :, 0])
        avg_g = np.average(result[:, :, 1])
        avg_r = np.average(result[:, :, 2])

        # Scale factors to make each channel average = 128
        scale_b = 128.0 / avg_b if avg_b > 0 else 1.0
        scale_g = 128.0 / avg_g if avg_g > 0 else 1.0
        scale_r = 128.0 / avg_r if avg_r > 0 else 1.0

        # Apply scaling
        result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * scale_g, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)

        return result.astype(np.uint8)

    def create_comparison(self, original: np.ndarray, processed: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of original and processed images.

        Args:
            original: Original image
            processed: Processed image

        Returns:
            Comparison image
        """
        h, w = original.shape[:2]

        # Create canvas
        comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        comparison.fill(255)  # White separator

        # Place images
        comparison[:h, :w] = original
        comparison[:h, w+20:] = processed

        # Add labels
        cv2.putText(comparison, "ORIGINAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "PREPROCESSED", (w + 30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return comparison
