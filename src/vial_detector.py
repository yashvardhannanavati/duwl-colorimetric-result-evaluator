"""
Vial Detection Module

Locates and detects the test vial in an image by identifying the blue cap
and cylindrical body (Aquavial Quick Check kit).
"""

from typing import Tuple, Optional, Dict, Any
import cv2
import numpy as np
import yaml


class VialDetector:
    """Detects test vials in images using blue cap detection and contour analysis."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the vial detector with configuration parameters.

        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.detection_config = self.config['detection']
        self.quality_config = self.config['quality']

    def detect_vial(self, image: np.ndarray) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect the test vial in an image.

        Tries blue cap detection first, falls back to direct vial body detection
        if blue cap is not found.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Tuple of (success, detection_info)
            detection_info contains:
                - bbox: (x, y, w, h) bounding box of vial
                - liquid_region: (x, y, w, h) region containing liquid
                - blue_cap_center: (x, y) center of cap (or estimated top center)
                - confidence: detection confidence score (0-100)
        """
        if image is None or image.size == 0:
            return False, None

        # Check image quality
        quality_score = self._check_image_quality(image)
        if quality_score < self.quality_config['min_detection_confidence']:
            return False, None

        # Try blue cap detection first (primary method)
        cap_detected, cap_info = self._detect_blue_cap(image)

        # If blue cap not found, try black cap (for reference images)
        if not cap_detected:
            cap_detected, cap_info = self._detect_black_cap(image)

        if cap_detected:
            # Blue cap found - use standard detection
            vial_detected, vial_bbox = self._detect_vial_body(image, cap_info)
            if not vial_detected:
                return False, None

            # Calculate liquid region (below cap, above bottom)
            liquid_region = self._calculate_liquid_region(vial_bbox, cap_info)

            detection_info = {
                'bbox': vial_bbox,
                'liquid_region': liquid_region,
                'blue_cap_center': cap_info['center'],
                'confidence': quality_score * 0.4 + cap_info['confidence'] * 0.6
            }

            return True, detection_info

        else:
            # Blue cap NOT found - try direct vial body detection (fallback)
            vial_detected, vial_info = self._detect_vial_direct(image)

            if not vial_detected:
                return False, None

            detection_info = {
                'bbox': vial_info['bbox'],
                'liquid_region': vial_info['liquid_region'],
                'blue_cap_center': vial_info['cap_center'],
                'confidence': quality_score * 0.4 + vial_info['confidence'] * 0.6
            }

            return True, detection_info

    def _check_image_quality(self, image: np.ndarray) -> float:
        """
        Check image quality based on brightness and clarity.

        Args:
            image: Input image

        Returns:
            Quality score (0-100)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check brightness
        mean_brightness = np.mean(gray)
        if mean_brightness < self.quality_config['min_brightness']:
            brightness_score = 30
        elif mean_brightness > self.quality_config['max_brightness']:
            brightness_score = 40
        elif (self.quality_config['optimal_brightness_min'] <= mean_brightness <=
              self.quality_config['optimal_brightness_max']):
            brightness_score = 100
        else:
            brightness_score = 70

        # Check sharpness (using Laplacian variance)
        # Use config thresholds calibrated for plastic vials
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        excellent = self.quality_config.get('excellent_sharpness', 50)
        acceptable = self.quality_config.get('acceptable_sharpness', 25)
        poor = self.quality_config.get('poor_sharpness', 10)

        if sharpness >= excellent:
            sharpness_score = 100.0
        elif sharpness >= acceptable:
            sharpness_score = 70.0 + 30.0 * (sharpness - acceptable) / (excellent - acceptable)
        elif sharpness >= poor:
            sharpness_score = 40.0 + 30.0 * (sharpness - poor) / (acceptable - poor)
        else:
            sharpness_score = max(0.0, 40.0 * sharpness / poor)

        # Combined quality score
        quality_score = (brightness_score * 0.6 + sharpness_score * 0.4)
        return quality_score

    def _detect_blue_cap(self, image: np.ndarray) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect the blue cap of the test vial (Aquavial Quick Check).

        Args:
            image: Input image

        Returns:
            Tuple of (success, cap_info)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for blue color
        lower = np.array(self.detection_config['blue_cap_hsv_lower'])
        upper = np.array(self.detection_config['blue_cap_hsv_upper'])

        blue_mask = cv2.inRange(hsv, lower, upper)

        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None

        # Find largest contour (likely the blue cap)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 100:  # Minimum area threshold
            return False, None

        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate confidence based on circularity and area
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return False, None

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        confidence = min(100, circularity * 100)

        cap_info = {
            'center': (center_x, center_y),
            'bbox': (x, y, w, h),
            'area': area,
            'confidence': confidence
        }

        return True, cap_info

    def _detect_black_cap(self, image: np.ndarray) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Detect the black cap of the test vial (for official reference images).

        Args:
            image: Input image

        Returns:
            Tuple of (success, cap_info)
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for black/dark color (low value/brightness)
        lower = np.array(self.detection_config['black_cap_hsv_lower'])
        upper = np.array(self.detection_config['black_cap_hsv_upper'])

        black_mask = cv2.inRange(hsv, lower, upper)

        # Apply morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None

        # Find largest contour (likely the black cap)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        if area < 100:  # Minimum area threshold
            return False, None

        # Get bounding box and center
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + w // 2
        center_y = y + h // 2

        # Calculate confidence based on circularity and area
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return False, None

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        confidence = min(100, circularity * 100)

        cap_info = {
            'center': (center_x, center_y),
            'bbox': (x, y, w, h),
            'area': area,
            'confidence': confidence
        }

        return True, cap_info

    def _detect_vial_body(self, image: np.ndarray, cap_info: Dict[str, Any]) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Detect the vial body using contour detection.

        Args:
            image: Input image
            cap_info: Information about detected red cap

        Returns:
            Tuple of (success, bbox)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cap_center = cap_info['center']
        cap_y = cap_info['bbox'][1]

        # Filter contours that could be the vial
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.detection_config['min_vial_area']:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Check if contour contains or is near the cap
            if not (x <= cap_center[0] <= x + w):
                continue

            # Check aspect ratio (vial should be taller than wide)
            aspect_ratio = w / h if h > 0 else 0
            if not (self.detection_config['min_vial_aspect_ratio'] <=
                    aspect_ratio <= self.detection_config['max_vial_aspect_ratio']):
                continue

            # Vial should extend below the cap
            if y > cap_y:
                continue

            valid_contours.append((contour, area, (x, y, w, h)))

        if not valid_contours:
            # Fallback: create bounding box based on cap position
            cap_x, cap_y, cap_w, cap_h = cap_info['bbox']
            # Estimate vial dimensions
            vial_w = int(cap_w * 1.5)
            vial_h = int(cap_w * 4)  # Assume vial is ~4x as tall as cap is wide
            vial_x = cap_x - (vial_w - cap_w) // 2
            vial_y = cap_y - int(cap_h * 0.5)  # Start slightly above cap

            # Ensure within image bounds
            vial_x = max(0, vial_x)
            vial_y = max(0, vial_y)
            vial_w = min(vial_w, image.shape[1] - vial_x)
            vial_h = min(vial_h, image.shape[0] - vial_y)

            return True, (vial_x, vial_y, vial_w, vial_h)

        # Select the largest valid contour
        best_contour = max(valid_contours, key=lambda x: x[1])
        bbox = best_contour[2]

        return True, bbox

    def _calculate_liquid_region(self, vial_bbox: Tuple[int, int, int, int],
                                  cap_info: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Calculate the region containing the liquid (below cap, above bottom).

        Focuses on the bottom portion of the vial where liquid settles due to gravity.

        Args:
            vial_bbox: Bounding box of entire vial (x, y, w, h)
            cap_info: Information about red cap

        Returns:
            Liquid region bounding box (x, y, w, h)
        """
        vial_x, vial_y, vial_w, vial_h = vial_bbox
        cap_x, cap_y, cap_w, cap_h = cap_info['bbox']

        # Calculate vial bottom
        vial_bottom = vial_y + vial_h

        # CALIBRATION MODE: For small reference images, use smaller rectangle at bottom
        # Smaller, more focused sampling area for better color accuracy
        liquid_height_ratio = 0.25  # 25% height (SMALL for reference images)
        liquid_h = int(vial_h * liquid_height_ratio)

        # Position very low where the actual colored liquid sits in reference images
        liquid_y = vial_bottom - liquid_h - int(vial_h * 0.15)  # 8% from bottom (LOWER for reference images)

        # Ensure liquid region is below the cap
        cap_bottom = cap_y + cap_h
        if liquid_y < cap_bottom + 20:  # Need at least 20px clearance from cap
            liquid_y = cap_bottom + 20
            # Recalculate height to fit
            liquid_h = min(liquid_h, vial_bottom - liquid_y - int(vial_h * 0.05))

        # Liquid region is very narrow central strip (avoiding vial walls and background)
        # Thinner sampling to capture only the core liquid, not edges
        margin = int(vial_w * 0.35)  # 35% margin on each side (thinner sampling, was 30%)
        liquid_x = vial_x + margin
        liquid_w = vial_w - 2 * margin

        # Ensure positive dimensions
        liquid_h = max(50, liquid_h)  # Minimum 50 pixels height
        liquid_w = max(10, liquid_w)

        return (liquid_x, liquid_y, liquid_w, liquid_h)

    def _detect_vial_direct(self, image: np.ndarray) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Direct vial body detection without relying on red cap.

        Uses color-based segmentation to find non-gray regions (the vial)
        and vertical rectangle finding. Works for any cap color.

        Args:
            image: Input image

        Returns:
            Tuple of (success, vial_info)
            vial_info contains: bbox, liquid_region, cap_center, confidence
        """
        h, w = image.shape[:2]

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Convert to grayscale for brightness analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create mask for non-gray regions (vial contains colored liquid or is brighter/darker than background)
        # Method 1: Find regions with ANY saturation (even very subtle)
        # Extract saturation channel
        _, saturation, _ = cv2.split(hsv)

        # Use Otsu's thresholding on saturation to automatically find threshold
        _, sat_mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Method 2: Find regions different from median gray (more robust than mean)
        median_gray = np.median(gray)
        brightness_diff = np.abs(gray.astype(float) - median_gray)
        bright_mask = (brightness_diff > 8).astype(np.uint8) * 255  # Lower threshold for subtle differences

        # Method 3: Use adaptive thresholding on saturation channel
        sat_adaptive = cv2.adaptiveThreshold(saturation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 151, -3)

        # Combine all three methods
        combined_mask = cv2.bitwise_or(sat_mask, bright_mask)
        combined_mask = cv2.bitwise_or(combined_mask, sat_adaptive)

        # Morphological operations to clean up and connect regions
        kernel_large = np.ones((15, 15), np.uint8)
        kernel_small = np.ones((5, 5), np.uint8)

        # Close gaps
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=5)
        # Fill holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel_small, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, None

        # Scale minimum area based on image size (for high-res images)
        img_area = h * w
        min_vial_area = img_area * 0.01  # Vial should be at least 1% of image (reduced for better detection)

        # Filter for vertical rectangular contours (vial-like shapes)
        valid_vials = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip if too small relative to image size
            if area < min_vial_area:
                continue

            # Get bounding rectangle
            x, y, vw, vh = cv2.boundingRect(contour)

            # Check aspect ratio (vial should be tall and narrow)
            aspect_ratio = vw / vh if vh > 0 else 0

            # Vial is vertical cylinder: width/height should be 0.15-0.7
            if not (0.15 <= aspect_ratio <= 0.7):
                continue

            # Check if relatively centered in image (vials are usually centered)
            x_center = x + vw // 2
            center_offset = abs(x_center - w // 2) / w

            # Prefer vials within central 70% of image width
            centrality_score = max(0, 1.0 - (center_offset / 0.35))

            # Calculate confidence based on aspect ratio, size, and centrality
            ideal_aspect = 0.3
            aspect_score = 100 * max(0, 1.0 - abs(aspect_ratio - ideal_aspect) / ideal_aspect)
            size_score = min(100, (area / (img_area * 0.1)) * 100)  # Ideal: 10% of image

            confidence = (aspect_score * 0.35 + size_score * 0.35 + centrality_score * 100 * 0.30)

            valid_vials.append({
                'bbox': (x, y, vw, vh),
                'area': area,
                'confidence': confidence,
                'centrality': centrality_score,
                'aspect_ratio': aspect_ratio
            })

        if not valid_vials:
            return False, None

        # Select best vial (highest confidence)
        best_vial = max(valid_vials, key=lambda v: v['confidence'])

        vial_x, vial_y, vial_w, vial_h = best_vial['bbox']

        # Estimate cap center (top center of vial)
        cap_center_x = vial_x + vial_w // 2
        cap_center_y = vial_y + int(vial_h * 0.1)  # ~10% from top

        # Create synthetic cap_info for liquid region calculation
        cap_info = {
            'center': (cap_center_x, cap_center_y),
            'bbox': (vial_x, vial_y, vial_w, int(vial_h * 0.15))  # Estimate cap height as 15% of vial
        }

        # Calculate liquid region using existing method
        liquid_region = self._calculate_liquid_region(best_vial['bbox'], cap_info)

        vial_info = {
            'bbox': best_vial['bbox'],
            'liquid_region': liquid_region,
            'cap_center': (cap_center_x, cap_center_y),
            'confidence': best_vial['confidence']
        }

        return True, vial_info

    def visualize_detection(self, image: np.ndarray, detection_info: Dict[str, Any]) -> np.ndarray:
        """
        Create a visualization of the detection results.

        Args:
            image: Original image
            detection_info: Detection information

        Returns:
            Annotated image
        """
        annotated = image.copy()

        # Draw vial bounding box (green)
        x, y, w, h = detection_info['bbox']
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw liquid region (blue)
        lx, ly, lw, lh = detection_info['liquid_region']
        cv2.rectangle(annotated, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)

        # Draw blue cap center (cyan)
        cap_x, cap_y = detection_info['blue_cap_center']
        cv2.circle(annotated, (cap_x, cap_y), 5, (255, 255, 0), -1)

        # Add confidence text
        conf_text = f"Confidence: {detection_info['confidence']:.1f}%"
        cv2.putText(annotated, conf_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated
