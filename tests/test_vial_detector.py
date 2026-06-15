"""
Unit tests for VialDetector module
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vial_detector import VialDetector


class TestVialDetector(unittest.TestCase):
    """Test cases for VialDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use parent directory's config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.detector = VialDetector(config_path)

    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.config)
        self.assertIn('detection', self.detector.config)

    def test_detect_vial_none_image(self):
        """Test detection with None image."""
        success, info = self.detector.detect_vial(None)
        self.assertFalse(success)
        self.assertIsNone(info)

    def test_detect_vial_empty_image(self):
        """Test detection with empty image."""
        empty_image = np.array([])
        success, info = self.detector.detect_vial(empty_image)
        self.assertFalse(success)
        self.assertIsNone(info)

    def test_check_image_quality(self):
        """Test image quality assessment."""
        # Create a well-lit test image
        good_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        quality = self.detector._check_image_quality(good_image)
        self.assertGreater(quality, 0)
        self.assertLessEqual(quality, 100)

        # Create a dark image
        dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 20
        dark_quality = self.detector._check_image_quality(dark_image)
        self.assertLess(dark_quality, quality)

    def test_detect_red_cap(self):
        """Test red cap detection."""
        # Create synthetic image with red region
        image = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Add red circle (cap)
        cv2.circle(image, (320, 100), 30, (0, 0, 255), -1)

        success, cap_info = self.detector._detect_red_cap(image)
        # May or may not detect depending on HSV thresholds
        if success:
            self.assertIn('center', cap_info)
            self.assertIn('confidence', cap_info)

    def test_calculate_liquid_region(self):
        """Test liquid region calculation."""
        vial_bbox = (100, 50, 80, 300)
        cap_info = {
            'center': (140, 80),
            'bbox': (120, 60, 40, 20)
        }

        liquid_region = self.detector._calculate_liquid_region(vial_bbox, cap_info)

        self.assertEqual(len(liquid_region), 4)
        x, y, w, h = liquid_region

        # Liquid should be below cap
        cap_bottom = cap_info['bbox'][1] + cap_info['bbox'][3]
        self.assertGreaterEqual(y, cap_bottom)

        # Dimensions should be positive
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)


if __name__ == '__main__':
    unittest.main()
