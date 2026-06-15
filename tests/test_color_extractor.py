"""
Unit tests for ColorExtractor module
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from color_extractor import ColorExtractor


class TestColorExtractor(unittest.TestCase):
    """Test cases for ColorExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.extractor = ColorExtractor(config_path)

    def test_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
        self.assertIsNotNone(self.extractor.config)

    def test_extract_color_none_image(self):
        """Test extraction with None image."""
        success, info = self.extractor.extract_color(None, (0, 0, 100, 100))
        self.assertFalse(success)
        self.assertIsNone(info)

    def test_extract_color_invalid_region(self):
        """Test extraction with invalid region."""
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        success, info = self.extractor.extract_color(image, (0, 0, -10, -10))
        self.assertFalse(success)
        self.assertIsNone(info)

    def test_extract_color_valid_region(self):
        """Test extraction with valid region."""
        # Create pink image
        image = np.ones((480, 640, 3), dtype=np.uint8)
        image[:, :] = (193, 182, 255)  # BGR format for pink

        liquid_region = (200, 200, 100, 100)
        success, color_info = self.extractor.extract_color(image, liquid_region)

        self.assertTrue(success)
        self.assertIsNotNone(color_info)
        self.assertIn('rgb', color_info)
        self.assertIn('hsv', color_info)
        self.assertIn('variance', color_info)

        # Check RGB is close to pink
        rgb = color_info['rgb']
        self.assertEqual(len(rgb), 3)
        self.assertAlmostEqual(rgb[0], 255, delta=5)  # R
        self.assertAlmostEqual(rgb[1], 182, delta=5)  # G
        self.assertAlmostEqual(rgb[2], 193, delta=5)  # B

    def test_get_central_sample(self):
        """Test central sample extraction."""
        roi = np.ones((100, 100, 3), dtype=np.uint8) * 128
        sample = self.extractor._get_central_sample(roi, 0.6)

        # Sample should be smaller than original
        self.assertLessEqual(sample.shape[0], roi.shape[0])
        self.assertLessEqual(sample.shape[1], roi.shape[1])

    def test_calculate_color_variance(self):
        """Test color variance calculation."""
        # Uniform color - low variance
        uniform_roi = np.ones((100, 100, 3), dtype=np.uint8) * 128
        variance_uniform = self.extractor._calculate_color_variance(uniform_roi)
        self.assertAlmostEqual(variance_uniform, 0.0, delta=0.1)

        # Random colors - high variance
        np.random.seed(42)
        random_roi = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        variance_random = self.extractor._calculate_color_variance(random_roi)
        self.assertGreater(variance_random, variance_uniform)

    def test_get_color_uniformity_score(self):
        """Test uniformity score calculation."""
        # Low variance = high uniformity
        low_variance = 50
        high_score = self.extractor.get_color_uniformity_score(low_variance)
        self.assertGreater(high_score, 80)

        # High variance = low uniformity
        high_variance = 1500
        low_score = self.extractor.get_color_uniformity_score(high_variance)
        self.assertLess(low_score, 20)

    def test_create_color_patch(self):
        """Test color patch creation."""
        rgb = [255, 0, 0]  # Red
        patch = self.extractor.create_color_patch(rgb, (50, 50))

        self.assertEqual(patch.shape, (50, 50, 3))
        # Check color (BGR format)
        self.assertEqual(patch[0, 0, 2], 255)  # R
        self.assertEqual(patch[0, 0, 1], 0)    # G
        self.assertEqual(patch[0, 0, 0], 0)    # B


if __name__ == '__main__':
    unittest.main()
