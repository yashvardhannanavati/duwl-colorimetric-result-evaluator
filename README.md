# Dental Unit Waterline (DUWL) Colorimetric Result Evaluator

A Python tool for automated analysis of dental unit waterline (DUWL) microbial contamination test results using the **Aquavial Quick Check** test kit. It classifies colorimetric vial results objectively — replacing subjective visual reading with quantitative, reproducible measurements.

## Overview

The Aquavial Quick Check test produces a color change in a reagent vial after a 15-minute reaction at room temperature (20–30°C / 68–90°F):

| Color | Classification | CFU/mL | Status |
|-------|---------------|--------|--------|
| Pink | Clean | < 500 | Acceptable |
| Purple | Contaminated | 500–2000 | Requires treatment |
| Clear | Highly Contaminated | > 2000 | Requires immediate action |

This tool accepts a photograph of a test vial and automatically:
1. Detects the vial and isolates the liquid region
2. Preprocesses the image to compensate for lighting variation
3. Extracts the representative liquid color (RGB + HSV)
4. Classifies the contamination level via weighted color distance
5. Reports results with a confidence score and quality indicators

## Features

- Distance-based classification with a perceptually-weighted RGB formula
- Image preprocessing pipeline (white balance, contrast enhancement, sharpening)
- Batch processing with CSV export for multiple samples
- Quality indicators: color uniformity, image quality, detection confidence
- Ambiguity warnings for borderline results
- Configurable thresholds via `config.yaml` — no code changes needed
- Annotated output images and per-sample JSON files

## Installation

**Requirements:** Python 3.8+

```bash
git clone https://github.com/yashvardhannanavati/duwl-colorimetric-result-evaluator.git
cd duwl-colorimetric-result-evaluator

python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

## Usage

### Single image

```bash
cd src
python main.py path/to/image.jpg
```

Show visualization:
```bash
python main.py path/to/image.jpg --show
```

Get JSON output:
```bash
python main.py path/to/image.jpg --format json
```

### Batch processing

```bash
python main.py image1.jpg image2.jpg image3.jpg --batch
python main.py ../sample_images/*.jpg --batch
```

### All options

```
positional arguments:
  images           Path(s) to test vial image(s)

optional arguments:
  -h, --help       Show help message
  --config, -c     Path to config file (default: config.yaml)
  --show, -s       Display visualization window
  --batch, -b      Batch mode — saves a CSV summary
  --no-save        Do not save output images
  --format, -f     Output format: text or json (default: text)
```

## Configuration

Edit `config.yaml` to adjust:

- **Reference colors** — RGB values for pink/purple/clear standards
- **Preprocessing** — white balance, contrast enhancement, sharpening, noise reduction
- **Detection** — cap HSV ranges, vial dimensions, Hough circle parameters
- **Classification** — RGB distance weights, confidence thresholds, ambiguity threshold
- **Quality** — uniformity variance, brightness and sharpness thresholds
- **Output** — directories, CSV filename, visualization options

```yaml
preprocessing:
  enabled: true
  white_balance: false          # Enable for non-white backgrounds
  contrast_enhancement: true    # Improves pink/purple/clear separation
  sharpening:
    enabled: true
    amount: 1.0
```

## How It Works

### 1. Image Preprocessing
- **Noise reduction** via bilateral filtering (preserves edges)
- **White balance correction** using Gray World / background sampling
- **Contrast enhancement** via CLAHE on the luminance channel only
- **Sharpening** via unsharp mask

### 2. Vial Detection
- Detects the vial cap using HSV color thresholding
- Identifies the vial body via Canny edge detection and contour filtering
- Calculates the liquid region in the lower portion of the vial using narrow central margins to avoid wall reflections

### 3. Color Extraction
- Samples the central 60% of the liquid region after median blur
- Computes mean RGB and HSV values
- Measures color variance as a uniformity metric

### 4. Classification

Weighted Euclidean distance to each reference color:

```
D = sqrt(2*(R1-R2)² + 4*(G1-G2)² + 3*(B1-B2)²)
```

Green is weighted 2× more than red to maximize pink/purple separation (the two colors differ most in their green channel). The class with the minimum distance wins, with a saturation crosscheck to validate or override CLEAR classifications.

Confidence score:

```
confidence = 100 × (1 - d_min / d_max)
```

### 5. Output

- **Annotated image** — bounding boxes, color patch, reference comparison, confidence score
- **JSON file** — complete color metrics, distances, quality indicators, timestamp
- **CSV** (batch mode) — one row per sample, importable into R, Python, or Excel

## Output Example

```json
{
  "classification": "PURPLE",
  "cfu_range": "500-2000 CFU/mL",
  "confidence_score": 61.9,
  "color_metrics": {
    "sample_rgb": [136, 130, 141],
    "sample_hsv": [136.0, 0.078, 0.55],
    "distance_to_pink": 217.37,
    "distance_to_purple": 140.68,
    "distance_to_clear": 315.33,
    "saturation_score": 7.8
  },
  "quality_indicators": {
    "color_uniformity": 81.8,
    "image_quality": "fair",
    "detection_confidence": 60.8
  }
}
```

## Sample Images

A synthetic reference image (`sample_images/sample_pink_clean.jpg`) is included to test the pipeline without a physical test kit. See `sample_images/README.md` for a script to generate synthetic images for all three classification categories.

## Troubleshooting

| Error | Likely cause | Fix |
|-------|-------------|-----|
| "Could not detect vial" | Cap not visible or poor focus | Ensure cap is clearly visible and image is in focus |
| "Could not extract color" | Liquid region obscured | Check vial orientation and image resolution |
| Low confidence score | Lighting or mixing issue | Improve lighting; invert vial before capturing |
| Misclassification | Gray background color bleed | Enable `white_balance: true` in config |

## Project Structure

```
.
├── src/
│   ├── main.py                # CLI and pipeline orchestration
│   ├── image_preprocessor.py  # White balance, contrast, sharpening
│   ├── vial_detector.py       # Cap and vial body detection
│   ├── color_extractor.py     # ROI sampling and color metrics
│   ├── color_classifier.py    # Distance-based classification
│   └── results_reporter.py    # Output generation (images, JSON, CSV)
├── tests/                     # Unit tests
├── sample_images/             # Reference images for testing
├── config.yaml                # All tunable parameters
├── requirements.txt
└── LICENSE
```

## Testing

```bash
cd tests
python -m pytest
python -m pytest --cov=src      # with coverage
```

## Contributing

Bug reports and pull requests are welcome. Please open an issue first for significant changes.

## License

MIT — see [LICENSE](LICENSE) for details.

## Citation

If you use this tool in research, please cite:

```
DUWL Colorimetric Result Evaluator (2026)
https://github.com/yashvardhannanavati/duwl-colorimetric-result-evaluator
```

## References

- Aquavial Quick Check Waterline Test Kit Instructions, Exact Blue Technologies Inc.
- Zuiderveld (1994), Contrast Limited Adaptive Histogram Equalization
- Tomasi & Manduchi (1998), Bilateral Filtering for Gray and Color Images
- van de Weijer et al. (2007), Learning-Based Color Constancy
