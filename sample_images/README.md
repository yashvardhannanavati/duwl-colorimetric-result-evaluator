# Sample Images

This directory contains example images for testing and demonstrating the analysis pipeline.

## Included Files

| File | Description |
|------|-------------|
| `sample_input.jpg` | Raw vial photograph (input to the analyzer) |
| `sample_output.jpg` | Annotated result image produced by the analyzer |
| `sample_output_values.json` | Full JSON output with color metrics and quality indicators |
| `sample_pink_clean.jpg` | Synthetic vial image for pipeline testing (no test kit required) |

## Image Requirements

For best results, input images should meet the following criteria:

- **Format**: JPG or PNG
- **Resolution**: Minimum 640×480 pixels (higher recommended)
- **Background**: White background strongly recommended
- **Lighting**: Indirect natural daylight or daylight-balanced LED (5000–6500 K)
- **Timing**: Capture at exactly 15 minutes after reagent addition
- **Temperature**: 20–30°C (68–90°F) during reaction
- **Framing**: Vial upright, centered, cap clearly visible, filling 30–50% of frame

## Generating Synthetic Test Images

If you don't have a physical test kit, you can generate synthetic vial images for each classification:

```python
import cv2
import numpy as np

def create_synthetic_vial(liquid_bgr, output_path):
    img = np.ones((640, 480, 3), dtype=np.uint8) * 255  # white background
    vial_x, vial_y, vial_w, vial_h = 180, 120, 120, 380
    cv2.rectangle(img, (vial_x, vial_y), (vial_x + vial_w, vial_y + vial_h), (200, 200, 200), -1)
    cv2.rectangle(img, (vial_x, vial_y), (vial_x + vial_w, vial_y + vial_h), (150, 150, 150), 2)
    cap_h = 50
    cv2.rectangle(img, (vial_x - 5, vial_y - 5), (vial_x + vial_w + 5, vial_y + cap_h), (30, 30, 180), -1)
    liquid_y = vial_y + int(vial_h * 0.60)
    liquid_h = int(vial_h * 0.35)
    cv2.rectangle(img, (vial_x + 5, liquid_y), (vial_x + vial_w - 5, liquid_y + liquid_h), liquid_bgr, -1)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])

# Pink  — Clean          (<500 CFU/mL)
create_synthetic_vial((193, 182, 255), 'sample_pink_clean.jpg')
# Purple — Contaminated  (500–2000 CFU/mL)
create_synthetic_vial((219, 112, 147), 'sample_purple_contaminated.jpg')
# Clear  — Highly cont.  (>2000 CFU/mL)
create_synthetic_vial((240, 240, 240), 'sample_clear_highly_contaminated.jpg')
```

## Running the Analyzer on Samples

```bash
cd ../src

# Single image
python main.py ../sample_images/sample_input.jpg --show

# Batch — all samples
python main.py ../sample_images/sample_input.jpg ../sample_images/sample_pink_clean.jpg --batch
```
