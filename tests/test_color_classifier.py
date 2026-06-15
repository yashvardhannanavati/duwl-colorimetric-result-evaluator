"""
Unit tests for ColorClassifier module
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from color_classifier import ColorClassifier


class TestColorClassifier(unittest.TestCase):
    """Test cases for ColorClassifier class."""

    def setUp(self):
        """Set up test fixtures."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        self.classifier = ColorClassifier(config_path)

    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier)
        self.assertIsNotNone(self.classifier.config)
        self.assertIn('reference_colors', self.classifier.config)

    def test_classify_pink(self):
        """Test classification of pink (clean) sample."""
        color_info = {
            'rgb': [255, 182, 193],  # Pink reference
            'hsv': [350, 0.5, 1.0],
            'variance': 100.0,
            'sample_size': 1000
        }

        result = self.classifier.classify(color_info)

        self.assertEqual(result['classification'], 'PINK')
        self.assertEqual(result['cfu_range'], '<500 CFU/mL')
        self.assertGreater(result['confidence_score'], 0)
        self.assertIn('color_metrics', result)

    def test_classify_purple(self):
        """Test classification of purple (contaminated) sample."""
        color_info = {
            'rgb': [147, 112, 219],  # Purple reference
            'hsv': [260, 0.5, 0.86],
            'variance': 150.0,
            'sample_size': 1000
        }

        result = self.classifier.classify(color_info)

        self.assertEqual(result['classification'], 'PURPLE')
        self.assertEqual(result['cfu_range'], '500-2000 CFU/mL')
        self.assertGreater(result['confidence_score'], 0)

    def test_classify_clear(self):
        """Test classification of clear (highly contaminated) sample."""
        color_info = {
            'rgb': [240, 240, 240],  # Clear reference
            'hsv': [0, 0.02, 0.94],  # Very low saturation
            'variance': 80.0,
            'sample_size': 1000
        }

        result = self.classifier.classify(color_info)

        self.assertEqual(result['classification'], 'CLEAR')
        self.assertEqual(result['cfu_range'], '>2000 CFU/mL')

    def test_weighted_rgb_distance(self):
        """Test weighted RGB distance calculation."""
        rgb1 = [255, 0, 0]
        rgb2 = [255, 0, 0]

        # Same color - distance should be 0
        distance_same = self.classifier._weighted_rgb_distance(rgb1, rgb2)
        self.assertEqual(distance_same, 0.0)

        # Different colors
        rgb3 = [0, 255, 0]
        distance_diff = self.classifier._weighted_rgb_distance(rgb1, rgb3)
        self.assertGreater(distance_diff, 0)

    def test_get_confidence_level(self):
        """Test confidence level categorization."""
        self.assertEqual(self.classifier.get_confidence_level(90), "High")
        self.assertEqual(self.classifier.get_confidence_level(65), "Medium")
        self.assertEqual(self.classifier.get_confidence_level(30), "Low")

    def test_get_all_distances(self):
        """Test distance calculation to all references."""
        rgb_sample = [200, 150, 180]
        distances = self.classifier.get_all_distances(rgb_sample)

        self.assertIn('pink', distances)
        self.assertIn('purple', distances)
        self.assertIn('clear', distances)

        for distance in distances.values():
            self.assertGreaterEqual(distance, 0)

    def test_get_reference_colors(self):
        """Test getting reference colors."""
        refs = self.classifier.get_reference_colors()

        self.assertIn('pink', refs)
        self.assertIn('purple', refs)
        self.assertIn('clear', refs)

        self.assertEqual(len(refs['pink']), 3)
        self.assertEqual(len(refs['purple']), 3)
        self.assertEqual(len(refs['clear']), 3)

    def test_get_reference_info(self):
        """Test getting reference information."""
        info = self.classifier.get_reference_info('PINK')

        self.assertIn('label', info)
        self.assertIn('cfu_range', info)
        self.assertIn('description', info)
        self.assertIn('rgb', info)


if __name__ == '__main__':
    unittest.main()
