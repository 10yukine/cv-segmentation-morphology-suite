"""Run with: python test_cv_suite.py"""

import cv2
import numpy as np

import cv_suite


def synthetic_image() -> np.ndarray:
    """Grey background with one red, one green and one blue square, plus a bright disc."""
    img = np.full((240, 320, 3), 60, np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), -1)  # BGR red
    cv2.rectangle(img, (120, 20), (180, 80), (0, 255, 0), -1)
    cv2.rectangle(img, (220, 20), (280, 80), (255, 0, 0), -1)
    cv2.circle(img, (160, 170), 40, (230, 230, 230), -1)
    return img


def test_region_growing_selects_only_the_seeded_blob():
    gray = cv2.cvtColor(synthetic_image(), cv2.COLOR_BGR2GRAY)
    mask = cv_suite.region_growing(gray, seed=(160, 170), threshold=15)
    assert mask.shape == gray.shape
    assert mask[170, 160] == 255 and mask[5, 5] == 0
    disc_area = np.pi * 40**2
    assert abs(np.count_nonzero(mask) - disc_area) / disc_area < 0.05


def test_hsv_labels_each_colour_once():
    out = cv_suite.hsv_segregation(synthetic_image())[1][1]  # RGB panel with boxes drawn
    # Each square should have a box in its own colour just outside its top-left corner.
    assert tuple(out[20, 20]) == (255, 0, 0)  # red box on the red square
    assert tuple(out[20, 120]) == (0, 255, 0)
    assert tuple(out[20, 220]) == (0, 0, 255)


def test_count_splits_touching_objects_on_either_background():
    centres = [(50, 60), (96, 60), (200, 60), (100, 170), (230, 170)]  # first two overlap
    for background, disc in ((220, 40), (40, 220)):
        img = np.full((240, 320, 3), background, np.uint8)
        for centre in centres:
            cv2.circle(img, centre, 25, (disc, disc, disc), -1)
        title = cv_suite.count_objects(img)[1][0]
        assert title == "Counted: 5 objects", f"{title} (background={background})"


def test_every_pipeline_returns_drawable_panels():
    img = synthetic_image()
    pipelines = [
        cv_suite.morphology,
        cv_suite.thresholding,
        cv_suite.edges,
        cv_suite.region,
        cv_suite.kmeans,
        cv_suite.hsv_segregation,
        cv_suite.count_objects,
    ]
    for pipeline in pipelines:
        panels = pipeline(img)
        assert panels, pipeline.__name__
        for title, panel, cmap in panels:
            assert panel.dtype == np.uint8 and panel.shape[:2] == img.shape[:2], (pipeline.__name__, title)
            assert (panel.ndim == 2) == (cmap == "gray"), (pipeline.__name__, title)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
