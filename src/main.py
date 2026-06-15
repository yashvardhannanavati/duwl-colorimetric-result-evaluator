#!/usr/bin/env python3
"""
CleanStream Waterline Test Analyzer - Main CLI

Analyzes CleanStream Waterline Test Kit vial images and classifies water quality.
"""

import argparse
import sys
import os
from typing import List, Dict, Any
import cv2

from vial_detector import VialDetector
from color_extractor import ColorExtractor
from color_classifier import ColorClassifier
from results_reporter import ResultsReporter
from image_preprocessor import ImagePreprocessor


class WaterlineAnalyzer:
    """Main analyzer class orchestrating the analysis pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the analyzer.

        Args:
            config_path: Path to configuration file
        """
        self.preprocessor = ImagePreprocessor(config_path)
        self.detector = VialDetector(config_path)
        self.extractor = ColorExtractor(config_path)
        self.classifier = ColorClassifier(config_path)
        self.reporter = ResultsReporter(config_path)

    def analyze_image(self, image_path: str, show_visualization: bool = False,
                      save_output: bool = True) -> Dict[str, Any]:
        """
        Analyze a single test vial image.

        Args:
            image_path: Path to test image
            show_visualization: Whether to display visualization
            save_output: Whether to save output images

        Returns:
            Analysis report dictionary
        """
        # Load image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        print(f"\nAnalyzing: {os.path.basename(image_path)}")
        print("-" * 60)

        # Step 1: Preprocess image
        print("1. Preprocessing image...")
        image = self.preprocessor.preprocess(original_image)
        print("   ✓ Image preprocessed (white balance, contrast enhancement)")

        # Step 2: Detect vial
        print("2. Detecting vial...")
        success, detection_info = self.detector.detect_vial(image)

        if not success or detection_info is None:
            print("   ERROR: Could not detect vial in image")
            return self._create_error_report(image_path, "Vial detection failed")

        print(f"   ✓ Vial detected (confidence: {detection_info['confidence']:.1f}%)")

        # Step 3: Extract color
        print("3. Extracting liquid color...")
        success, color_info = self.extractor.extract_color(image, detection_info['liquid_region'])

        if not success or color_info is None:
            print("   ERROR: Could not extract color from liquid region")
            return self._create_error_report(image_path, "Color extraction failed")

        rgb = color_info['rgb']
        print(f"   ✓ Color extracted: RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")

        # Step 4: Classify
        print("4. Classifying water quality...")
        classification_result = self.classifier.classify(color_info)

        print(f"   ✓ Classification: {classification_result['classification']}")
        print(f"     CFU Range: {classification_result['cfu_range']}")
        print(f"     Confidence: {classification_result['confidence_score']:.1f}%")

        # Step 4: Generate quality indicators
        quality_indicators = self._generate_quality_indicators(
            detection_info, color_info, classification_result
        )

        # Check for warnings
        self._print_warnings(quality_indicators, classification_result)

        # Step 5: Generate report
        report = self.reporter.generate_report(
            image_path, detection_info, color_info,
            classification_result, quality_indicators
        )

        # Step 6: Create and save visualization
        if save_output or show_visualization:
            visualization = self.reporter.create_visualization(
                image, detection_info, color_info, classification_result
            )

            if save_output:
                output_path = self.reporter.save_visualization(visualization, image_path)
                print(f"\n✓ Results saved to: {output_path}")

                # Save mathematical values to separate JSON file
                json_path = self.reporter.save_mathematical_values(report, image_path)
                print(f"✓ Mathematical values saved to: {json_path}")

            if show_visualization:
                cv2.imshow("Analysis Result", visualization)
                print("\nPress any key to close visualization...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return report

    def analyze_batch(self, image_paths: List[str], save_csv: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze multiple test vial images.

        Args:
            image_paths: List of image paths
            save_csv: Whether to save results to CSV

        Returns:
            List of analysis reports
        """
        print(f"\nBatch Processing: {len(image_paths)} images")
        print("=" * 60)

        reports = []

        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}]")
            try:
                report = self.analyze_image(image_path, show_visualization=False, save_output=True)
                reports.append(report)
            except Exception as e:
                print(f"   ERROR: {str(e)}")
                error_report = self._create_error_report(image_path, str(e))
                reports.append(error_report)

        # Save to CSV
        if save_csv and reports:
            csv_path = self.reporter.save_to_csv(reports)
            print(f"\n{'=' * 60}")
            print(f"✓ Batch results saved to: {csv_path}")
            print(f"{'=' * 60}")

        # Print summary
        self._print_batch_summary(reports)

        return reports

    def _generate_quality_indicators(self, detection_info: Dict[str, Any],
                                      color_info: Dict[str, Any],
                                      classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality indicator metrics."""

        # Color uniformity
        color_uniformity = self.extractor.get_color_uniformity_score(color_info['variance'])

        # Image quality assessment
        detection_confidence = detection_info['confidence']
        if detection_confidence >= 80:
            image_quality = "good"
        elif detection_confidence >= 60:
            image_quality = "fair"
        else:
            image_quality = "poor"

        return {
            'color_uniformity': color_uniformity,
            'image_quality': image_quality,
            'detection_confidence': detection_confidence
        }

    def _print_warnings(self, quality_indicators: Dict[str, Any],
                        classification_result: Dict[str, Any]) -> None:
        """Print warnings about quality or ambiguity."""

        warnings = []

        # Check image quality
        if quality_indicators['image_quality'] == 'poor':
            warnings.append("Poor image quality detected - results may be unreliable")

        # Check color uniformity
        if quality_indicators['color_uniformity'] < 50:
            warnings.append("Low color uniformity - liquid may not be well-mixed")

        # Check for ambiguous classification
        if self.classifier.is_ambiguous(classification_result['color_metrics']):
            warnings.append("Ambiguous result - color is equidistant from multiple references")

        # Check confidence
        if classification_result['confidence_score'] < 60:
            warnings.append("Low confidence score - consider retesting")

        if warnings:
            print("\n⚠ WARNINGS:")
            for warning in warnings:
                print(f"  • {warning}")

    def _print_batch_summary(self, reports: List[Dict[str, Any]]) -> None:
        """Print summary statistics for batch processing."""

        total = len(reports)
        successful = sum(1 for r in reports if 'error' not in r)
        failed = total - successful

        if successful == 0:
            return

        # Count classifications
        classifications = {}
        for report in reports:
            if 'error' not in report:
                cls = report['classification']
                classifications[cls] = classifications.get(cls, 0) + 1

        print(f"\nBATCH SUMMARY:")
        print(f"  Total processed: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"\n  Classifications:")
        for cls, count in sorted(classifications.items()):
            percentage = (count / successful) * 100
            print(f"    {cls}: {count} ({percentage:.1f}%)")

    def _create_error_report(self, image_path: str, error_msg: str) -> Dict[str, Any]:
        """Create an error report."""
        from datetime import datetime
        return {
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path,
            'error': error_msg,
            'classification': 'ERROR',
            'cfu_range': 'N/A',
            'description': 'Detection failed',
            'confidence_score': 0.0,
            'color_metrics': {
                'sample_rgb': [0, 0, 0],
                'sample_hsv': [0, 0, 0],
                'saturation_score': 0,
                'distance_to_pink': 0,
                'distance_to_purple': 0,
                'distance_to_clear': 0
            },
            'quality_indicators': {
                'color_uniformity': 0,
                'image_quality': 'poor',
                'detection_confidence': 0
            }
        }


def main():
    """Main entry point for CLI."""

    parser = argparse.ArgumentParser(
        description='CleanStream Waterline Test Analyzer - Analyze test vial images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Analyze a single image:
    python main.py image.jpg

  Analyze with visualization:
    python main.py image.jpg --show

  Batch process multiple images:
    python main.py image1.jpg image2.jpg image3.jpg --batch

  Process all images in a directory:
    python main.py sample_images/*.jpg --batch

  Custom configuration:
    python main.py image.jpg --config custom_config.yaml

  Output formats:
    python main.py image.jpg --format json
    python main.py image.jpg --format text
        '''
    )

    parser.add_argument('images', nargs='+', help='Path(s) to test vial image(s)')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Display visualization window')
    parser.add_argument('--batch', '-b', action='store_true',
                        help='Batch processing mode (saves CSV)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save output images')
    parser.add_argument('--format', '-f', choices=['text', 'json'],
                        default='text', help='Output format for single image (default: text)')

    args = parser.parse_args()

    # Resolve config path relative to this script's directory if not found as-is
    config_path = args.config
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, '..', args.config)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    args.config = config_path

    # Initialize analyzer
    try:
        analyzer = WaterlineAnalyzer(args.config)
    except Exception as e:
        print(f"Error initializing analyzer: {e}")
        sys.exit(1)

    # Process images
    try:
        if args.batch or len(args.images) > 1:
            # Batch mode
            reports = analyzer.analyze_batch(args.images, save_csv=True)

        else:
            # Single image mode
            image_path = args.images[0]

            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                sys.exit(1)

            report = analyzer.analyze_image(
                image_path,
                show_visualization=args.show,
                save_output=not args.no_save
            )

            # Print report
            print("\n" + "=" * 60)
            if args.format == 'json':
                print(analyzer.reporter.format_json(report))
            else:
                print(analyzer.reporter.format_text(report))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
