"""Numerically equivalent sparse median evaluation for the frozen risk rule.

Only held-out median outputs are needed for reconstruction risk. Compute those
3x3 medians directly instead of filtering all other pixels and discarding them.
The gate, loss values, actual recognition views, and selector remain unchanged.
"""
import numpy as np
from PIL import Image
from .data import preprocess_image


def masked_risks_fast(image, views):
    observed = np.asarray(image, dtype=np.float32) / 255.0
    h, w = observed.shape[:2]
    if h < 8 or w < 8:
        raise ValueError("masked-risk routing requires images of at least 8x8")
    replacement = (observed[:-2, 1:-1] + observed[2:, 1:-1]
                   + observed[1:-1, :-2] + observed[1:-1, 2:]) / 4.0
    losses = []
    for offset in (0, 2):
        mask = np.zeros((h, w), dtype=bool)
        mask[1 + offset:h - 1:4, 1 + offset:w - 1:4] = True
        hidden = observed.copy()
        hidden[1:-1, 1:-1][mask[1:-1, 1:-1]] = replacement[mask[1:-1, 1:-1]]
        hidden_u8 = np.rint(hidden * 255).astype(np.uint8)
        hidden_image = Image.fromarray(hidden_u8)
        yy, xx = np.nonzero(mask)
        by_view = []
        for name in views:
            if name == "median_3":
                # Every target is one pixel inside the boundary; no pad is used.
                neighbors = np.stack([hidden_u8[yy + dy, xx + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
                estimate = np.partition(neighbors, 4, axis=1)[:, 4].astype(np.float32) / 255.0
            else:
                estimate = np.asarray(preprocess_image(hidden_image, name), dtype=np.float32)[mask] / 255.0
            residual = estimate - observed[mask]
            by_view.append([float(np.mean(residual ** 2)), float(np.mean(np.abs(residual)))])
        losses.append(by_view)
    return np.mean(losses, axis=0).astype(np.float32)
