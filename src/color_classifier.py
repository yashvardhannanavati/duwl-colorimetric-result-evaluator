"""
Color Classification Module

Compares extracted colors to reference standards and classifies water quality.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import yaml


class ColorClassifier:
    """Classifies water quality based on colorimetric comparison to reference standards."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the color classifier with configuration parameters.

        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.reference_colors = self.config['reference_colors']
        self.extraction_config = self.config['color_extraction']
        self.classification_config = self.config['classification']

    def classify(self, color_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify water quality based on extracted color.

        Args:
            color_info: Color information from ColorExtractor

        Returns:
            Classification result with metrics
        """
        rgb_sample = color_info['rgb']
        hsv_sample = color_info['hsv']

        # ALWAYS calculate distances to all references first
        dist_pink = self._weighted_rgb_distance(rgb_sample, self.reference_colors['pink']['rgb'])
        dist_purple = self._weighted_rgb_distance(rgb_sample, self.reference_colors['purple']['rgb'])
        dist_clear = self._weighted_rgb_distance(rgb_sample, self.reference_colors['clear']['rgb'])

        distances = {
            'pink': dist_pink,
            'purple': dist_purple,
            'clear': dist_clear
        }

        # Find closest color by distance
        min_class = min(distances, key=distances.get)
        min_dist = distances[min_class]

        # Check saturation to validate CLEAR classification
        saturation = hsv_sample[1]
        saturation_threshold = self.extraction_config['clear_saturation_threshold']

        # Classification strategy: Use distance-based classification
        # The Aquavial kit's PURPLE result has very low saturation naturally,
        # so we rely on distance to references rather than saturation thresholds
        classification = min_class.upper()
        cfu_range = self.reference_colors[min_class]['cfu_range']
        description = self.reference_colors[min_class]['description']
        confidence = self._calculate_confidence(distances, min_class)

        # Calculate saturation score (0-100)
        saturation_score = min(100, saturation * 100)

        result = {
            'classification': classification,
            'cfu_range': cfu_range,
            'description': description,
            'confidence_score': confidence,
            'color_metrics': {
                'sample_rgb': rgb_sample,
                'sample_hsv': hsv_sample,
                'distance_to_pink': dist_pink,
                'distance_to_purple': dist_purple,
                'distance_to_clear': dist_clear,
                'saturation_score': saturation_score
            }
        }

        return result

    def _weighted_rgb_distance(self, rgb1: List[int], rgb2: List[int]) -> float:
        """
        Calculate weighted Euclidean distance between two RGB colors.

        Uses formula: Distance = sqrt(2*(R1-R2)² + 4*(G1-G2)² + 3*(B1-B2)²)

        Args:
            rgb1: First RGB color [R, G, B]
            rgb2: Second RGB color [R, G, B]

        Returns:
            Weighted distance
        """
        r_weight = self.classification_config['rgb_weight_r']
        g_weight = self.classification_config['rgb_weight_g']
        b_weight = self.classification_config['rgb_weight_b']

        r_diff = rgb1[0] - rgb2[0]
        g_diff = rgb1[1] - rgb2[1]
        b_diff = rgb1[2] - rgb2[2]

        distance = np.sqrt(
            r_weight * r_diff**2 +
            g_weight * g_diff**2 +
            b_weight * b_diff**2
        )

        return float(distance)

    def _calculate_confidence(self, distances: Dict[str, float], min_class: str) -> float:
        """
        Calculate confidence score based on distance separation.

        Args:
            distances: Dictionary of distances to each reference
            min_class: The class with minimum distance

        Returns:
            Confidence score (0-100)
        """
        min_dist = distances[min_class]

        # Find second-best distance
        other_distances = [d for k, d in distances.items() if k != min_class]
        if not other_distances:
            return 100.0

        second_min_dist = min(other_distances)

        if second_min_dist == 0:
            return 50.0  # Ambiguous case

        # Calculate separation ratio
        if min_dist == 0:
            return 100.0

        separation = (second_min_dist - min_dist) / second_min_dist

        # Check if ambiguous
        ambiguity_threshold = self.classification_config['ambiguity_threshold']
        if separation < ambiguity_threshold:
            # Low separation = low confidence
            confidence = 50.0
        else:
            # Map separation to confidence (0.15-1.0 -> 50-100)
            normalized = (separation - ambiguity_threshold) / (1 - ambiguity_threshold)
            confidence = 50.0 + (normalized * 50.0)

        return min(100.0, max(0.0, confidence))

    def _calculate_confidence_clear(self, saturation: float, threshold: float) -> float:
        """
        Calculate confidence for CLEAR classification based on saturation.

        Args:
            saturation: Measured saturation (0-1)
            threshold: Saturation threshold for CLEAR

        Returns:
            Confidence score (0-100)
        """
        if saturation >= threshold:
            return 50.0  # Borderline case

        # Map saturation to confidence (0-threshold -> 100-50)
        normalized = saturation / threshold
        confidence = 100.0 - (normalized * 50.0)

        return min(100.0, max(50.0, confidence))

    def get_confidence_level(self, confidence: float) -> str:
        """
        Get confidence level as text.

        Args:
            confidence: Confidence score (0-100)

        Returns:
            Confidence level string
        """
        high_threshold = self.classification_config['high_confidence_threshold']
        low_threshold = self.classification_config['low_confidence_threshold']

        if confidence >= high_threshold:
            return "High"
        elif confidence >= low_threshold:
            return "Medium"
        else:
            return "Low"

    def is_ambiguous(self, color_metrics: Dict[str, Any]) -> bool:
        """
        Check if the classification is ambiguous.

        Args:
            color_metrics: Color metrics from classification

        Returns:
            True if ambiguous, False otherwise
        """
        distances = [
            color_metrics['distance_to_pink'],
            color_metrics['distance_to_purple'],
            color_metrics['distance_to_clear']
        ]

        sorted_distances = sorted(distances)
        min_dist = sorted_distances[0]
        second_min_dist = sorted_distances[1]

        if second_min_dist == 0:
            return True

        separation = (second_min_dist - min_dist) / second_min_dist
        ambiguity_threshold = self.classification_config['ambiguity_threshold']

        return separation < ambiguity_threshold

    def get_all_distances(self, rgb_sample: List[int]) -> Dict[str, float]:
        """
        Calculate distances to all reference colors.

        Args:
            rgb_sample: Sample RGB color

        Returns:
            Dictionary of distances
        """
        return {
            'pink': self._weighted_rgb_distance(rgb_sample, self.reference_colors['pink']['rgb']),
            'purple': self._weighted_rgb_distance(rgb_sample, self.reference_colors['purple']['rgb']),
            'clear': self._weighted_rgb_distance(rgb_sample, self.reference_colors['clear']['rgb'])
        }

    def get_reference_colors(self) -> Dict[str, List[int]]:
        """
        Get reference RGB colors.

        Returns:
            Dictionary of reference colors
        """
        return {
            'pink': self.reference_colors['pink']['rgb'],
            'purple': self.reference_colors['purple']['rgb'],
            'clear': self.reference_colors['clear']['rgb']
        }

    def get_reference_info(self, classification: str) -> Dict[str, str]:
        """
        Get detailed information about a classification.

        Args:
            classification: Classification label (PINK, PURPLE, CLEAR)

        Returns:
            Reference information
        """
        key = classification.lower()
        if key not in self.reference_colors:
            return {}

        return {
            'label': self.reference_colors[key]['label'],
            'cfu_range': self.reference_colors[key]['cfu_range'],
            'description': self.reference_colors[key]['description'],
            'rgb': self.reference_colors[key]['rgb']
        }
