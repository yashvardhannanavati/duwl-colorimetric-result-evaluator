"""
Results Reporter Module

Formats and exports test results with visualizations.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os
import csv
import json
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class ResultsReporter:
    """Formats and exports waterline test results."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the results reporter with configuration parameters.

        Args:
            config_path: Path to the configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.output_config = self.config['output']
        self.reference_colors = self.config['reference_colors']

        # Ensure output directory exists
        os.makedirs(self.output_config['output_dir'], exist_ok=True)

    def generate_report(self, image_path: str, detection_info: Dict[str, Any],
                        color_info: Dict[str, Any], classification_result: Dict[str, Any],
                        quality_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a complete test report.

        Args:
            image_path: Path to original test image
            detection_info: Vial detection information
            color_info: Color extraction information
            classification_result: Classification result
            quality_indicators: Quality metrics

        Returns:
            Complete report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'classification': classification_result['classification'],
            'cfu_range': classification_result['cfu_range'],
            'description': classification_result['description'],
            'confidence_score': classification_result['confidence_score'],
            'color_metrics': classification_result['color_metrics'],
            'quality_indicators': quality_indicators
        }

        return report

    def format_json(self, report: Dict[str, Any]) -> str:
        """
        Format report as JSON.

        Args:
            report: Report dictionary

        Returns:
            JSON string
        """
        return json.dumps(report, indent=2)

    def format_text(self, report: Dict[str, Any]) -> str:
        """
        Format report as human-readable text.

        Args:
            report: Report dictionary

        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CleanStream Waterline Test Results")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {report['timestamp']}")
        lines.append(f"Image: {os.path.basename(report['image_path'])}")
        lines.append("")
        lines.append(f"RESULT: {report['classification']}")
        lines.append(f"Status: {report['description']}")
        lines.append(f"CFU Range: {report['cfu_range']}")
        lines.append(f"Confidence: {report['confidence_score']:.1f}%")
        lines.append("")
        lines.append("Color Metrics:")
        lines.append(f"  Sample RGB: {report['color_metrics']['sample_rgb']}")
        lines.append(f"  Sample HSV: [{report['color_metrics']['sample_hsv'][0]:.0f}, "
                     f"{report['color_metrics']['sample_hsv'][1]:.2f}, "
                     f"{report['color_metrics']['sample_hsv'][2]:.2f}]")
        lines.append(f"  Saturation: {report['color_metrics']['saturation_score']:.1f}%")
        lines.append("")
        lines.append("Distance to References:")
        lines.append(f"  Pink: {report['color_metrics']['distance_to_pink']:.2f}")
        lines.append(f"  Purple: {report['color_metrics']['distance_to_purple']:.2f}")
        lines.append(f"  Clear: {report['color_metrics']['distance_to_clear']:.2f}")
        lines.append("")
        lines.append("Quality Indicators:")
        lines.append(f"  Color Uniformity: {report['quality_indicators']['color_uniformity']:.1f}%")
        lines.append(f"  Image Quality: {report['quality_indicators']['image_quality']}")
        lines.append(f"  Detection Confidence: {report['quality_indicators']['detection_confidence']:.1f}%")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save_to_csv(self, reports: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save multiple reports to CSV file.

        Args:
            reports: List of report dictionaries
            filename: Output filename (optional)

        Returns:
            Path to saved CSV file
        """
        if filename is None:
            filename = self.output_config['csv_filename']

        filepath = os.path.join(self.output_config['output_dir'], filename)

        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'timestamp',
                'image_path',
                'classification',
                'cfu_range',
                'confidence_score',
                'sample_rgb_r',
                'sample_rgb_g',
                'sample_rgb_b',
                'sample_hsv_h',
                'sample_hsv_s',
                'sample_hsv_v',
                'saturation_score',
                'distance_to_pink',
                'distance_to_purple',
                'distance_to_clear',
                'color_uniformity',
                'image_quality',
                'detection_confidence'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for report in reports:
                row = {
                    'timestamp': report['timestamp'],
                    'image_path': report['image_path'],
                    'classification': report['classification'],
                    'cfu_range': report['cfu_range'],
                    'confidence_score': report['confidence_score'],
                    'sample_rgb_r': report['color_metrics']['sample_rgb'][0],
                    'sample_rgb_g': report['color_metrics']['sample_rgb'][1],
                    'sample_rgb_b': report['color_metrics']['sample_rgb'][2],
                    'sample_hsv_h': report['color_metrics']['sample_hsv'][0],
                    'sample_hsv_s': report['color_metrics']['sample_hsv'][1],
                    'sample_hsv_v': report['color_metrics']['sample_hsv'][2],
                    'saturation_score': report['color_metrics']['saturation_score'],
                    'distance_to_pink': report['color_metrics']['distance_to_pink'],
                    'distance_to_purple': report['color_metrics']['distance_to_purple'],
                    'distance_to_clear': report['color_metrics']['distance_to_clear'],
                    'color_uniformity': report['quality_indicators']['color_uniformity'],
                    'image_quality': report['quality_indicators']['image_quality'],
                    'detection_confidence': report['quality_indicators']['detection_confidence']
                }
                writer.writerow(row)

        return filepath

    def create_visualization(self, image: np.ndarray, detection_info: Dict[str, Any],
                             color_info: Dict[str, Any], classification_result: Dict[str, Any]) -> np.ndarray:
        """
        Create comprehensive visualization of results.

        Args:
            image: Original image
            detection_info: Detection information
            color_info: Color information
            classification_result: Classification result

        Returns:
            Annotated image with visualization
        """
        # Create a larger canvas for visualization
        height, width = image.shape[:2]
        vis_width = width + 400  # Extra space for info panel
        canvas = np.ones((height, vis_width, 3), dtype=np.uint8) * 255

        # Place original image with annotations
        annotated = image.copy()

        # Draw vial bounding box
        if self.output_config['show_bounding_box']:
            x, y, w, h = detection_info['bbox']
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "Vial", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw liquid region
        lx, ly, lw, lh = detection_info['liquid_region']
        cv2.rectangle(annotated, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
        cv2.putText(annotated, "Liquid", (lx, ly - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Place annotated image on canvas
        canvas[:height, :width] = annotated

        # Create info panel
        panel_x = width + 10
        y_offset = 30

        # Title
        cv2.putText(canvas, "TEST RESULTS", (panel_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
        y_offset += 60

        # Classification
        classification = classification_result['classification']
        color_map = {
            'PINK': (255, 182, 193),
            'PURPLE': (147, 112, 219),
            'CLEAR': (240, 240, 240)
        }
        result_color = color_map.get(classification, (128, 128, 128))

        cv2.putText(canvas, classification, (panel_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, result_color[::-1], 4)  # BGR
        y_offset += 55

        # CFU Range
        cv2.putText(canvas, classification_result['cfu_range'], (panel_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        y_offset += 40

        # Confidence
        conf_text = f"Confidence: {classification_result['confidence_score']:.1f}%"
        cv2.putText(canvas, conf_text, (panel_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
        y_offset += 45

        # Sample color patch
        if self.output_config['show_color_sample']:
            sample_rgb = color_info['rgb']
            patch_size = 90

            # Ensure patch fits within canvas height
            available_height = canvas.shape[0] - y_offset
            if available_height < patch_size:
                patch_size = max(40, available_height - 10)  # Minimum size or available space

            patch = np.full((patch_size, patch_size, 3),
                            (sample_rgb[2], sample_rgb[1], sample_rgb[0]), dtype=np.uint8)
            cv2.rectangle(patch, (0, 0), (patch_size-1, patch_size-1), (0, 0, 0), 2)

            canvas[y_offset:y_offset+patch_size, panel_x:panel_x+patch_size] = patch
            cv2.putText(canvas, "Sample", (panel_x, y_offset - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            y_offset += patch_size + 30

        # Reference colors
        if self.output_config['show_reference_comparison']:
            # Check if we have enough space for references
            if y_offset < canvas.shape[0] - 50:
                cv2.putText(canvas, "References:", (panel_x, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
                y_offset += 35

                for ref_name in ['pink', 'purple', 'clear']:
                    # Check if we have space for this reference
                    if y_offset + 50 > canvas.shape[0]:
                        break

                    ref_rgb = self.reference_colors[ref_name]['rgb']
                    patch_h = min(50, canvas.shape[0] - y_offset - 5)
                    patch_w = 140
                    patch = np.full((patch_h, patch_w, 3),
                                    (ref_rgb[2], ref_rgb[1], ref_rgb[0]), dtype=np.uint8)
                    cv2.rectangle(patch, (0, 0), (patch_w-1, patch_h-1), (0, 0, 0), 2)

                    canvas[y_offset:y_offset+patch_h, panel_x:panel_x+patch_w] = patch

                    label = ref_name.upper()
                    cv2.putText(canvas, label, (panel_x + patch_w + 10, y_offset + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                    y_offset += patch_h + 15

        return canvas

    def create_distance_chart(self, classification_result: Dict[str, Any],
                               size: Tuple[int, int] = (400, 300)) -> np.ndarray:
        """
        Create a bar chart showing distances to reference colors.

        Args:
            classification_result: Classification result
            size: (width, height) of chart

        Returns:
            Chart image as numpy array
        """
        metrics = classification_result['color_metrics']

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=100)

        categories = ['Pink', 'Purple', 'Clear']
        distances = [
            metrics['distance_to_pink'],
            metrics['distance_to_purple'],
            metrics['distance_to_clear']
        ]

        colors_bar = ['#FFB6C1', '#9370DB', '#F0F0F0']
        bars = ax.bar(categories, distances, color=colors_bar, edgecolor='black')

        # Highlight the minimum (selected classification)
        min_idx = distances.index(min(distances))
        bars[min_idx].set_edgecolor('red')
        bars[min_idx].set_linewidth(3)

        ax.set_ylabel('Distance', fontsize=10)
        ax.set_title('Distance to Reference Colors', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(distances) * 1.2)

        # Add value labels on bars
        for i, (bar, dist) in enumerate(zip(bars, distances)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{dist:.1f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # Convert to numpy array
        fig.canvas.draw()
        chart_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        chart_array = chart_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)

        # Convert RGB to BGR for OpenCV
        chart_array = cv2.cvtColor(chart_array, cv2.COLOR_RGB2BGR)

        return chart_array

    def save_visualization(self, visualization: np.ndarray, image_path: str,
                           suffix: str = "_result") -> str:
        """
        Save visualization image.

        Args:
            visualization: Visualization image
            image_path: Original image path
            suffix: Suffix for output filename

        Returns:
            Path to saved image
        """
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        output_filename = f"{name}{suffix}{ext}"
        output_path = os.path.join(self.output_config['output_dir'], output_filename)

        cv2.imwrite(output_path, visualization)
        return output_path

    def save_mathematical_values(self, report: Dict[str, Any], image_path: str) -> str:
        """
        Save mathematical values to a separate JSON file.

        Args:
            report: Report dictionary containing all analysis results
            image_path: Original image path

        Returns:
            Path to saved JSON file
        """
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        output_filename = f"{name}_values.json"
        output_path = os.path.join(self.output_config['output_dir'], output_filename)

        # Extract mathematical values
        math_values = {
            'timestamp': report['timestamp'],
            'image_path': report['image_path'],
            'classification': report['classification'],
            'cfu_range': report['cfu_range'],
            'confidence_score': report['confidence_score'],
            'color_metrics': {
                'sample_rgb': report['color_metrics']['sample_rgb'],
                'sample_hsv': report['color_metrics']['sample_hsv'],
                'saturation_score': report['color_metrics']['saturation_score'],
                'distance_to_pink': report['color_metrics']['distance_to_pink'],
                'distance_to_purple': report['color_metrics']['distance_to_purple'],
                'distance_to_clear': report['color_metrics']['distance_to_clear']
            },
            'quality_indicators': {
                'color_uniformity': report['quality_indicators']['color_uniformity'],
                'image_quality': report['quality_indicators']['image_quality'],
                'detection_confidence': report['quality_indicators']['detection_confidence']
            }
        }

        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(math_values, f, indent=2)

        return output_path

    def print_summary(self, report: Dict[str, Any]) -> None:
        """
        Print a summary of the report to console.

        Args:
            report: Report dictionary
        """
        print("\n" + "=" * 60)
        print(f"RESULT: {report['classification']} - {report['description']}")
        print(f"CFU Range: {report['cfu_range']}")
        print(f"Confidence: {report['confidence_score']:.1f}%")
        print("=" * 60 + "\n")
