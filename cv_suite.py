"""Classical computer-vision toolkit: one CLI, six comparison pipelines and an object counter.

Example: python cv_suite.py photo.jpg --mode edge
         python cv_suite.py coins.jpg --mode count
         python cv_suite.py photo.jpg --mode kmeans --k 5 --save out.png
"""

import argparse
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

Panel = Tuple[str, np.ndarray, Optional[str]]  # title, image, matplotlib cmap (None = RGB)

MAX_SIDE = 1280  # larger inputs are downscaled: keeps K-Means fast and kernel sizes meaningful
MIN_BLOB_AREA = 500


def _rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _magnitude_u8(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(cv2.magnitude(gx, gy), 0, 255))


def morphology(img_bgr: np.ndarray, kernel_size: int = 5) -> List[Panel]:
    _, binary = cv2.threshold(_gray(img_bgr), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    ops = [
        ("Opening", cv2.MORPH_OPEN),
        ("Closing", cv2.MORPH_CLOSE),
        ("Gradient", cv2.MORPH_GRADIENT),
        ("Top-Hat", cv2.MORPH_TOPHAT),
    ]
    return [
        ("Original", _rgb(img_bgr), None),
        ("Binary (Otsu)", binary, "gray"),
        ("Erosion", cv2.erode(binary, kernel), "gray"),
        ("Dilation", cv2.dilate(binary, kernel), "gray"),
        *[(name, cv2.morphologyEx(binary, op, kernel), "gray") for name, op in ops],
    ]


def thresholding(img_bgr: np.ndarray) -> List[Panel]:
    gray = _gray(img_bgr)
    _, global_t = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    otsu_value, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = lambda method: cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, 11, 2)  # noqa: E731
    return [
        ("Grayscale", gray, "gray"),
        ("Global (T=127)", global_t, "gray"),
        ("Adaptive Mean", adaptive(cv2.ADAPTIVE_THRESH_MEAN_C), "gray"),
        ("Adaptive Gaussian", adaptive(cv2.ADAPTIVE_THRESH_GAUSSIAN_C), "gray"),
        (f"Otsu (T={otsu_value:.0f})", otsu, "gray"),
    ]


def edges(img_bgr: np.ndarray) -> List[Panel]:
    gray = _gray(img_bgr)
    sobel = _magnitude_u8(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    prewitt = _magnitude_u8(
        cv2.filter2D(gray, cv2.CV_64F, prewitt_x), cv2.filter2D(gray, cv2.CV_64F, prewitt_x.T)
    )

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    log = np.uint8(np.clip(np.abs(cv2.Laplacian(blurred, cv2.CV_64F)), 0, 255))
    return [
        ("Grayscale", gray, "gray"),
        ("Sobel Magnitude", sobel, "gray"),
        ("Prewitt Magnitude", prewitt, "gray"),
        ("Canny (100, 200)", cv2.Canny(gray, 100, 200), "gray"),
        ("Laplacian of Gaussian", log, "gray"),
    ]


def region_growing(gray: np.ndarray, seed: Tuple[int, int], threshold: int = 15) -> np.ndarray:
    """Grow an 8-connected region of pixels within `threshold` grey levels of the seed pixel.

    This is BFS region growing, done by cv2.floodFill in C instead of a Python queue
    (FIXED_RANGE compares each pixel to the seed, not to its neighbour).
    """
    h, w = gray.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)  # floodFill wants a 1 px border
    flags = 8 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(gray, mask, seed, 0, loDiff=threshold, upDiff=threshold, flags=flags)
    return mask[1:-1, 1:-1]


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu-threshold objects from background, whichever of the two is darker."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:  # the frame edge came out as "object": polarity is backwards
        mask = cv2.bitwise_not(mask)
    return mask


def watershed_labels(img_bgr: np.ndarray, peak: float = 0.7) -> np.ndarray:
    """Marker-based watershed. Returns labels: -1 boundary, 1 background, 2+ one per object.

    `peak` is the fraction of the largest distance-transform value a pixel needs to seed
    an object. Lower it when small objects are missed, raise it when touching objects merge.
    """
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(foreground_mask(_gray(img_bgr)), cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    sure_fg = np.uint8(dist > peak * dist.max()) * 255
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1  # background becomes 1 so that 0 can mean "unknown, let watershed decide"
    markers[unknown == 255] = 0
    return cv2.watershed(img_bgr, markers)


def count_objects(img_bgr: np.ndarray, peak: float = 0.5) -> List[Panel]:
    """Count separate objects (coins, cells, pills), splitting ones that touch."""
    labels = watershed_labels(img_bgr, peak)
    object_ids = [i for i in np.unique(labels) if i > 1]
    out = img_bgr.copy()
    for number, object_id in enumerate(object_ids, 1):
        mask = np.uint8(labels == object_id)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
        m = cv2.moments(mask, binaryImage=True)
        centre = (int(m["m10"] / m["m00"]) - 8, int(m["m01"] / m["m00"]) + 8)
        cv2.putText(out, str(number), centre, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return [
        ("Original", _rgb(img_bgr), None),
        (f"Counted: {len(object_ids)} objects", _rgb(out), None),
    ]


def region(img_bgr: np.ndarray, seed: Optional[Tuple[int, int]] = None, threshold: int = 15) -> List[Panel]:
    gray = _gray(img_bgr)
    h, w = gray.shape
    seed = seed or (w // 2, h // 2)
    if not (0 <= seed[0] < w and 0 <= seed[1] < h):
        raise ValueError(f"Seed {seed} is outside the {w}x{h} image.")

    seeded = _rgb(img_bgr)
    cv2.drawMarker(seeded, seed, (255, 0, 0), cv2.MARKER_CROSS, 24, 3)
    boundaries = img_bgr.copy()
    boundaries[watershed_labels(img_bgr) == -1] = (0, 0, 255)
    return [
        (f"Original (seed at {seed[0]}, {seed[1]})", seeded, None),
        (f"Region Growing (±{threshold})", region_growing(gray, seed, threshold), "gray"),
        ("Watershed Boundaries", _rgb(boundaries), None),
    ]


def kmeans(img_bgr: np.ndarray, k: int = 3) -> List[Panel]:
    pixels = np.float32(img_bgr.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    segmented = np.uint8(centers)[labels.flatten()].reshape(img_bgr.shape)
    return [
        ("Original", _rgb(img_bgr), None),
        (f"K-Means Segmentation (K={k})", _rgb(segmented), None),
    ]


def hsv_segregation(img_bgr: np.ndarray) -> List[Panel]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # OpenCV hue runs 0-179. Red straddles the wrap-around, so it needs two bands.
    bands = {
        "RED": ((0, 0, 255), [((0, 50, 50), (10, 255, 255)), ((170, 50, 50), (179, 255, 255))]),
        "GREEN": ((0, 255, 0), [((35, 50, 50), (85, 255, 255))]),
        "BLUE": ((255, 0, 0), [((86, 50, 50), (140, 255, 255))]),
    }
    out = img_bgr.copy()
    for label, (color, ranges) in bands.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_BLOB_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                cv2.putText(out, label, (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return [
        ("Original", _rgb(img_bgr), None),
        ("HSV Color Segregation", _rgb(out), None),
    ]


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    scale = MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def plot_grid(panels: List[Panel], cols: int = 4, save_path: Optional[str] = None) -> None:
    cols = min(cols, len(panels))
    rows = (len(panels) + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, (title, img, cmap) in enumerate(panels, 1):
        plt.subplot(rows, cols, idx)
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Classical CV Segmentation & Morphology Suite")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--mode",
        choices=["morphology", "threshold", "edge", "region", "kmeans", "hsv", "count"],
        default="edge",
        help="Pipeline to execute",
    )
    parser.add_argument("--k", type=int, default=3, help="Clusters for K-Means")
    parser.add_argument("--kernel", type=int, default=5, help="Structuring element size for morphology")
    parser.add_argument("--seed", type=int, nargs=2, metavar=("X", "Y"), help="Region-growing seed (default: centre)")
    parser.add_argument("--tolerance", type=int, default=15, help="Region-growing grey-level tolerance")
    parser.add_argument("--peak", type=float, default=0.5, help="Seed threshold for count mode (0-1)")
    parser.add_argument("--save", metavar="PATH", help="Write the figure to a file instead of opening a window")
    args = parser.parse_args()

    try:
        if args.mode == "kmeans" and args.k < 1:
            raise ValueError("--k must be at least 1")
        if args.mode == "morphology" and args.kernel < 1:
            raise ValueError("--kernel must be at least 1")
        if args.mode == "region" and args.tolerance < 0:
            raise ValueError("--tolerance must be zero or greater")
        if args.mode == "count" and not 0.0 <= args.peak <= 1.0:
            raise ValueError("--peak must be between 0 and 1")

        image = load_image(args.image)
        if args.mode == "kmeans" and args.k > image.shape[0] * image.shape[1]:
            raise ValueError("--k cannot exceed the number of image pixels")
        panels = {
            "morphology": lambda: morphology(image, args.kernel),
            "threshold": lambda: thresholding(image),
            "edge": lambda: edges(image),
            "region": lambda: region(image, tuple(args.seed) if args.seed else None, args.tolerance),
            "kmeans": lambda: kmeans(image, args.k),
            "hsv": lambda: hsv_segregation(image),
            "count": lambda: count_objects(image, args.peak),
        }[args.mode]()
    except (FileNotFoundError, ValueError) as e:
        parser.error(str(e))
    plot_grid(panels, save_path=args.save)
