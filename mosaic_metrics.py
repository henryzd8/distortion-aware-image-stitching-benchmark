"""Full-reference mosaic and seam metrics for the benchmark."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from skimage.metrics import structural_similarity

from distortcorrect_stitcher_gpu import find_overlap_coords

DATA_RANGE = 65535.0  # uint16 tiles


def assemble_exact(
    tiles: np.ndarray,
    positions: np.ndarray,
    border: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste tiles in their local coordinate frame and average overlaps."""
    tiles = np.asarray(tiles)
    if tiles.ndim != 3:
        raise ValueError("assemble_exact expects single-channel tiles (t, y, x)")
    h, w = tiles.shape[-2:]
    pos = np.round(np.asarray(positions)).astype(int)
    y0, x0 = int(pos[:, 0].min()), int(pos[:, 1].min())
    H = int(pos[:, 0].max() - y0 + h)
    W = int(pos[:, 1].max() - x0 + w)
    canvas = np.zeros((H, W), dtype=np.float64)
    cover = np.zeros((H, W), dtype=np.float64)
    for idx in range(len(tiles)):
        tile = tiles[idx]
        yt = int(pos[idx, 0] - y0)
        xt = int(pos[idx, 1] - x0)
        canvas[yt + border:yt + h - border,
               xt + border:xt + w - border] += tile[border:-border,
                                                    border:-border]
        cover[yt + border:yt + h - border,
              xt + border:xt + w - border] += 1.0
    valid = cover > 0.5
    canvas[valid] /= cover[valid]
    return canvas, valid


def assemble_on_reference(
    tiles: np.ndarray,
    positions: np.ndarray,
    output_shape: Sequence[int],
    border: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Paste tiles into a fixed source frame and return coverage."""
    tiles = np.asarray(tiles)
    if tiles.ndim != 3:
        raise ValueError("assemble_on_reference expects (tile, y, x) data")
    out_h, out_w = (int(v) for v in output_shape)
    h, w = tiles.shape[-2:]
    pos = np.round(np.asarray(positions)).astype(int)
    canvas = np.zeros((out_h, out_w), dtype=np.float64)
    cover = np.zeros((out_h, out_w), dtype=np.float64)

    for tile, (y, x) in zip(tiles, pos):
        y0, y1 = max(y + border, 0), min(y + h - border, out_h)
        x0, x1 = max(x + border, 0), min(x + w - border, out_w)
        if y1 <= y0 or x1 <= x0:
            continue
        ty0, tx0 = y0 - y, x0 - x
        ty1, tx1 = ty0 + (y1 - y0), tx0 + (x1 - x0)
        canvas[y0:y1, x0:x1] += tile[ty0:ty1, tx0:tx1]
        cover[y0:y1, x0:x1] += 1.0

    valid = cover > 0.5
    canvas[valid] /= cover[valid]
    return canvas, valid


def _masked_ssim(
    gt: np.ndarray,
    mosaic: np.ndarray,
    valid: np.ndarray,
    data_range: float,
) -> float:
    """Average SSIM over windows fully contained in the valid mask."""
    from scipy.ndimage import uniform_filter

    ssim_map = structural_similarity(
        gt, mosaic, data_range=data_range, full=True)[1]
    win_ok = uniform_filter(valid.astype(np.float64), size=7,
                            mode="constant") > 0.999
    ok = win_ok[:ssim_map.shape[0], :ssim_map.shape[1]]
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean(ssim_map[ok]))


def metrics_vs_source(
    tiles_corrected: np.ndarray,
    positions: np.ndarray,
    source_crop: np.ndarray,
    border: int = 64,
    evaluation_positions: np.ndarray | None = None,
) -> dict[str, float]:
    """Calculate image-quality metrics against a source crop."""
    tiles_corrected = np.asarray(tiles_corrected)
    src = np.asarray(source_crop)
    mosaic, covered = assemble_on_reference(
        tiles_corrected, positions, src.shape[-2:], border)
    if evaluation_positions is None:
        valid = covered
    else:
        marker_tiles = np.ones_like(tiles_corrected, dtype=np.uint8)
        _, valid = assemble_on_reference(
            marker_tiles, evaluation_positions, src.shape[-2:], border)
    gt = src.astype(np.float64)
    if not np.any(valid):
        return {"psnr": float("nan"), "ssim": float("nan"),
                "ncc": float("nan"), "mse": float("nan"),
                "valid_fraction": 0.0, "coverage_fraction": 0.0}
    # rescaled only if tiles are normalised floats; uint16 tiles are left alone
    if not np.issubdtype(tiles_corrected.dtype, np.integer) \
            and np.nanmax(mosaic) <= 1.0:
        mosaic = mosaic * DATA_RANGE
    if not np.issubdtype(src.dtype, np.integer) and np.nanmax(gt) <= 1.0:
        gt = gt * DATA_RANGE
    coverage_fraction = float(np.mean(covered[valid]))
    mse = float(np.mean((mosaic[valid] - gt[valid]) ** 2))
    psnr = float("inf") if mse == 0 else \
        float(10.0 * np.log10(DATA_RANGE ** 2 / mse))
    # Crop the expensive SSIM calculation to the fixed evaluation footprint.
    yy, xx = np.where(valid)
    sl = (slice(int(yy.min()), int(yy.max()) + 1),
          slice(int(xx.min()), int(xx.max()) + 1))
    ssim = _masked_ssim(gt[sl], mosaic[sl], valid[sl], DATA_RANGE)

    a = gt[valid].ravel()
    b = mosaic[valid].ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan")

    return {"psnr": psnr, "ssim": ssim, "ncc": corr, "mse": mse,
            "valid_fraction": float(np.mean(valid)),
            "coverage_fraction": coverage_fraction}


def seam_error(
    tiles: np.ndarray,
    positions: np.ndarray,
    border: int = 64,
    adjacent_only: bool = True,
) -> float:
    """Measure normalized absolute intensity mismatch across tile seams."""
    tiles = np.asarray(tiles)
    if tiles.ndim != 3:
        raise ValueError("seam_error expects single-channel tiles (t, y, x)")
    h, w = tiles.shape[-2:]
    weighted_sum = 0.0
    total_weight = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            yx1, yx2 = find_overlap_coords(
                positions[i, 0], positions[i, 1],
                positions[j, 0], positions[j, 1], h, w)
            if yx1 is None:
                continue
            overlap_h = yx1[1] - yx1[0]
            overlap_w = yx1[3] - yx1[2]
            # Match the stitching graph: corner-only diagonal overlaps do not
            # provide a primary grid-adjacency constraint.
            if adjacent_only and overlap_h < 0.4 * h and overlap_w < 0.4 * w:
                continue
            b = min(border, (yx1[1] - yx1[0]) // 2, (yx1[3] - yx1[2]) // 2)
            if yx1[1] - yx1[0] <= 2 * b or yx1[3] - yx1[2] <= 2 * b:
                continue  # trimmed overlap is empty (e.g. small tiles)
            a = np.asarray(tiles[i], dtype=np.float64)[
                yx1[0] + b:yx1[1] - b, yx1[2] + b:yx1[3] - b]
            d = np.asarray(tiles[j], dtype=np.float64)[
                yx2[0] + b:yx2[1] - b, yx2[2] + b:yx2[3] - b]
            weight = int(a.size)
            weighted_sum += float(np.sum(np.abs(a - d)))
            total_weight += weight
    return weighted_sum / total_weight / DATA_RANGE \
        if total_weight else float("nan")
