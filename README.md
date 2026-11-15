# CV Segmentation & Morphology Suite

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-11557C)

A single command-line tool that runs classical computer-vision techniques on any image and shows the
results side by side. It covers six topics: morphology, thresholding, edge detection, region-based
segmentation, K-Means clustering and HSV colour segregation. A seventh mode puts them to work and
counts objects in a photo, separating ones that touch. No deep learning is involved.

It started as a set of separate lab scripts for a computer engineering elective, each with a
hard-coded image path, and was rebuilt into one tested tool.

## Installation

```bash
git clone https://github.com/10yukine/cv-segmentation-morphology-suite.git
cd cv-segmentation-morphology-suite
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python cv_suite.py <image> --mode <mode> [options]
```

Add `--save result.png` to any command to write the figure to a file instead of opening a window.
Images larger than 1280 px on their longest side are downscaled first.

## Modes

### `morphology`

Binarises the image with Otsu's threshold, then applies erosion, dilation, opening, closing,
morphological gradient and top-hat with a square structuring element.

```bash
python cv_suite.py coins.jpg --mode morphology
python cv_suite.py coins.jpg --mode morphology --kernel 9
```

### `threshold`

Compares a fixed global threshold (127), adaptive mean, adaptive Gaussian and Otsu's method. The
adaptive methods handle uneven lighting that defeats a single global value. The chosen Otsu
threshold is shown in the panel title.

```bash
python cv_suite.py document.jpg --mode threshold
```

### `edge`

Compares Sobel and Prewitt gradient magnitude, Canny, and Laplacian of Gaussian.

```bash
python cv_suite.py building.jpg --mode edge
```

### `region`

Runs two region-based methods:

- **Region growing** from a seed pixel, taking in every connected pixel within a grey-level
  tolerance of the seed. The seed is marked on the original.
- **Marker-based watershed**, with markers taken from the distance transform. Boundaries are drawn
  in red.

```bash
python cv_suite.py cells.jpg --mode region
python cv_suite.py cells.jpg --mode region --seed 200 150 --tolerance 25
```

### `kmeans`

Clusters pixels by colour into K groups and repaints each pixel with its cluster's mean colour.

```bash
python cv_suite.py landscape.jpg --mode kmeans --k 5
```

### `hsv`

Finds red, green and blue regions by hue range and draws a labelled box around each one larger than
500 px². Red uses two hue bands because it wraps around the ends of OpenCV's 0–179 hue scale.

```bash
python cv_suite.py balls.jpg --mode hsv
```

### `count`

Counts separate objects such as coins, cells or pills, and numbers each one. Thresholding alone
merges objects that touch into one blob. This mode chains four of the techniques above to split
them:

1. Otsu's threshold separates objects from background, and works for dark-on-light or
   light-on-dark images.
2. Morphological opening removes specks.
3. The distance transform turns each object into a hill whose peak is its centre. Two touching
   coins give two peaks with a saddle between them.
4. Watershed floods outward from the peaks and draws the boundary along the saddle.

```bash
python cv_suite.py coins.jpg --mode count
python cv_suite.py coins.jpg --mode count --peak 0.4
```

Lower `--peak` if small objects are missed. Raise it if touching objects are counted as one.

## Options

| Option | Used by | Default | Meaning |
|---|---|---|---|
| `--mode` | all | `edge` | Pipeline to run |
| `--save PATH` | all | none | Save the figure instead of displaying it |
| `--kernel N` | `morphology` | 5 | Structuring element size |
| `--seed X Y` | `region` | image centre | Region-growing seed pixel |
| `--tolerance N` | `region` | 15 | Grey-level tolerance for region growing |
| `--k N` | `kmeans` | 3 | Number of clusters |
| `--peak F` | `count` | 0.5 | How close to an object's centre a pixel must be to seed it (0–1) |

## Design

`cv_suite.py` has one function per mode. Each takes a BGR image and returns a list of
`(title, image, cmap)` panels, and `plot_grid` is the only place that touches Matplotlib. That keeps
the algorithms importable and testable without a display:

```python
import cv2, cv_suite
mask = cv_suite.region_growing(cv2.imread("cells.jpg", cv2.IMREAD_GRAYSCALE), seed=(200, 150))
```

## Tests

```bash
python test_cv_suite.py
```

The tests build a synthetic image with known shapes and check that region growing recovers the
seeded shape, that colour segregation labels each colour correctly, that counting separates two
overlapping discs on both light and dark backgrounds, and that every pipeline returns valid panels.

## License

MIT
