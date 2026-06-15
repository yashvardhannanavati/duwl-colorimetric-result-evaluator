"""
Color Extraction Module

Extracts representative color from the liquid region of the test vial.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
import yaml


class ColorExtractor:
    """Extracts representative color from liquid region of test vial."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the color extractor with configuration parameters.

        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.extraction_config = self.config['color_extraction']
        self.quality_config = self.config['quality']

    def extract_color(self, image: np.ndarray, liquid_region: Tuple[int, int, int, int]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Extract representative color from the liquid region.

        Args:
            image: Input image (BGR format)
            liquid_region: (x, y, w, h) bounding box of liquid region

        Returns:
            Tuple of (success, color_info)
            color_info contains:
                - rgb: [R, G, B] average RGB values
                - hsv: [H, S, V] average HSV values
                - variance: color variance (uniformity metric)
                - sample_size: number of pixels sampled
        """
        if image is None or image.size == 0:
            return False, None

        # Extract liquid region from image
        x, y, w, h = liquid_region
        if w <= 0 or h <= 0:
            return False, None

        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        liquid_roi = image[y:y+h, x:x+w]

        if liquid_roi.size == 0:
            return False, None

        # Apply median blur to reduce noise
        kernel_size = self.extraction_config['median_blur_kernel']
        blurred = cv2.medianBlur(liquid_roi, kernel_size)

        # Sample from central region (avoid edges)
        sample_percentage = self.extraction_config['liquid_sample_percentage']
        sample_roi = self._get_central_sample(blurred, sample_percentage)

        if sample_roi.size == 0:
            return False, None

        # Check minimum sample size
        sample_pixels = sample_roi.shape[0] * sample_roi.shape[1]
        if sample_pixels < self.extraction_config['min_sample_pixels']:
            return False, None

        # Calculate average RGB with gamma linearization for accurate averaging
        # sRGB JPEGs have gamma ~2.2; averaging in gamma space is inaccurate
        linear_roi = self._linearize_srgb(sample_roi)
        avg_linear_bgr = np.mean(linear_roi, axis=(0, 1))
        avg_srgb_bgr = self._delinearize_to_srgb(avg_linear_bgr)
        avg_rgb = [int(avg_srgb_bgr[2]), int(avg_srgb_bgr[1]), int(avg_srgb_bgr[0])]  # BGR to RGB

        # Convert to HSV
        avg_rgb_uint8 = np.uint8([[avg_rgb[::-1]]])  # Back to BGR for OpenCV
        avg_hsv_bgr = cv2.cvtColor(avg_rgb_uint8, cv2.COLOR_BGR2HSV)[0][0]
        avg_hsv = [float(avg_hsv_bgr[0]), float(avg_hsv_bgr[1]) / 255.0, float(avg_hsv_bgr[2]) / 255.0]

        # Calculate color variance (uniformity metric)
        variance = self._calculate_color_variance(sample_roi)

        color_info = {
            'rgb': avg_rgb,
            'hsv': avg_hsv,
            'variance': variance,
            'sample_size': sample_pixels
        }

        return True, color_info

    def _linearize_srgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert sRGB uint8 image to linear-light float64 array.

        Removes the gamma curve (~2.2) applied by cameras to JPEG images,
        allowing physically correct color averaging.

        Args:
            image: sRGB uint8 image

        Returns:
            Linear-light float64 array (0-1 range)
        """
        normalized = image.astype(np.float64) / 255.0
        return np.power(normalized, 2.2)

    def _delinearize_to_srgb(self, linear_bgr: np.ndarray) -> np.ndarray:
        """
        Convert linear-light float values back to sRGB uint8.

        Args:
            linear_bgr: Linear-light BGR values (float, 0-1 range per channel)

        Returns:
            sRGB uint8 values as array of 3 values [B, G, R]
        """
        srgb = np.power(np.clip(linear_bgr, 0, 1), 1.0 / 2.2) * 255.0
        return np.clip(srgb, 0, 255).astype(np.uint8)

    def _get_central_sample(self, roi: np.ndarray, percentage: float) -> np.ndarray:
        """
        Extract central region from ROI.

        Args:
            roi: Region of interest
            percentage: Percentage of area to sample (0-1)

        Returns:
            Central sample region
        """
        h, w = roi.shape[:2]

        # Calculate margins
        margin_h = int(h * (1 - percentage) / 2)
        margin_w = int(w * (1 - percentage) / 2)

        # Extract central region
        y1 = max(0, margin_h)
        y2 = min(h, h - margin_h)
        x1 = max(0, margin_w)
        x2 = min(w, w - margin_w)

        if y2 <= y1 or x2 <= x1:
            return roi  # Fallback to full ROI if margins are too large

        return roi[y1:y2, x1:x2]

    def _calculate_color_variance(self, roi: np.ndarray) -> float:
        """
        Calculate color variance within the ROI (uniformity metric).

        Args:
            roi: Region of interest

        Returns:
            Combined variance across all channels
        """
        # Calculate variance for each channel
        b_var = np.var(roi[:, :, 0])
        g_var = np.var(roi[:, :, 1])
        r_var = np.var(roi[:, :, 2])

        # Combined variance
        total_variance = float(b_var + g_var + r_var)

        return total_variance

    def get_color_uniformity_score(self, variance: float) -> float:
        """
        Convert variance to a uniformity score (0-100).

        Args:
            variance: Color variance

        Returns:
            Uniformity score (higher = more uniform)
        """
        if variance < self.quality_config['high_uniformity_variance']:
            return 100.0
        elif variance > self.quality_config['low_uniformity_variance']:
            return 0.0
        else:
            # Linear interpolation
            range_variance = (self.quality_config['low_uniformity_variance'] -
                              self.quality_config['high_uniformity_variance'])
            normalized = (variance - self.quality_config['high_uniformity_variance']) / range_variance
            return max(0.0, min(100.0, 100.0 * (1 - normalized)))

    def visualize_color_sample(self, image: np.ndarray, liquid_region: Tuple[int, int, int, int],
                                color_info: Dict[str, Any], size: int = 100) -> np.ndarray:
        """
        Create a visualization showing the sampled color.

        Args:
            image: Original image
            liquid_region: Liquid region coordinates
            color_info: Extracted color information
            size: Size of color patch to display

        Returns:
            Image with color patch overlay
        """
        annotated = image.copy()

        # Draw liquid region boundary
        x, y, w, h = liquid_region
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Draw sampling region (central area)
        percentage = self.extraction_config['liquid_sample_percentage']
        margin_h = int(h * (1 - percentage) / 2)
        margin_w = int(w * (1 - percentage) / 2)

        sample_x = x + margin_w
        sample_y = y + margin_h
        sample_w = w - 2 * margin_w
        sample_h = h - 2 * margin_h

        cv2.rectangle(annotated, (sample_x, sample_y),
                      (sample_x + sample_w, sample_y + sample_h),
                      (0, 255, 255), 1)

        # Create color patch
        rgb = color_info['rgb']
        bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR
        color_patch = np.full((size, size, 3), bgr, dtype=np.uint8)

        # Add border to patch
        cv2.rectangle(color_patch, (0, 0), (size-1, size-1), (0, 0, 0), 2)

        # Position patch in top-right corner
        patch_x = annotated.shape[1] - size - 10
        patch_y = 10

        # Overlay patch
        annotated[patch_y:patch_y+size, patch_x:patch_x+size] = color_patch

        # Add RGB text
        rgb_text = f"RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}"
        cv2.putText(annotated, rgb_text, (patch_x, patch_y + size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Add HSV text
        hsv = color_info['hsv']
        hsv_text = f"HSV: {hsv[0]:.0f}, {hsv[1]:.2f}, {hsv[2]:.2f}"
        cv2.putText(annotated, hsv_text, (patch_x, patch_y + size + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def create_color_patch(self, rgb: list, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Create a solid color patch.

        Args:
            rgb: [R, G, B] color values
            size: (width, height) of patch

        Returns:
            Color patch image
        """
        bgr = (rgb[2], rgb[1], rgb[0])
        patch = np.full((size[1], size[0], 3), bgr, dtype=np.uint8)
        return patch
