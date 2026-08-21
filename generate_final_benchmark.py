"""Generate deterministic tiled benchmarks with known geometry and distortion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

DEFAULT_SOURCE = (
    "datasets_mouse_brain_map_BrainReceptorShowcase_Slice2_Replicate1_images_"
    "mosaic_DAPI_z0.tif"
)
DEFAULT_K1 = (-0.008, 0.0, 0.008)
DEFAULT_NOISE = (0, 10)
DEFAULT_CROPS = [(26000, 6000), (38000, 6000),
                 (6000, 40000), (50000, 44000)]
DEFAULT_SEEDS = (2026, 2027)


def apply_radial_distortion(
    image: np.ndarray,
    k1: float,
    k2: float = 0.0,
    order: int = 1,
) -> np.ndarray:
    """Apply the tile-centred radial distortion model to one image."""
    image = np.asarray(image)
    h, w = image.shape[-2:]
    y_idx, x_idx = np.indices((h, w), dtype=np.float64)
    x_d = (x_idx - w / 2.0) / (w / 2.0)
    y_d = (y_idx - h / 2.0) / (h / 2.0)
    r_d = np.sqrt(x_d * x_d + y_d * y_d)
    if abs(k1) < 1e-15 and abs(k2) < 1e-15:
        return image.copy()

    r_u = r_d.copy()
    for _ in range(12):
        f = r_u + k1 * r_u ** 3 + k2 * r_u ** 5 - r_d
        df = 1.0 + 3.0 * k1 * r_u ** 2 + 5.0 * k2 * r_u ** 4
        r_u -= f / df

    scale = np.divide(r_u, r_d, out=np.ones_like(r_d), where=r_d > 1e-12)
    x_u = x_d * scale * (w / 2.0) + w / 2.0
    y_u = y_d * scale * (h / 2.0) + h / 2.0
    return map_coordinates(image, [y_u, x_u], order=order, mode="nearest")


def read_window(
    path: Path,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Read and validate a bounded window from a YX TIFF image."""
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if series.axes != "YX":
            raise ValueError(f"expected one YX image, found {series.axes!r}")
        image_h, image_w = series.shape
        if x < 0 or y < 0 or x + width > image_w or y + height > image_h:
            raise ValueError(
                f"crop ({x}, {y}, {width}, {height}) exceeds "
                f"image bounds ({image_w}, {image_h})"
            )

    mapped = tifffile.memmap(path, series=0, mode="r")
    try:
        return np.asarray(mapped[y:y + height, x:x + width]).copy()
    finally:
        del mapped


def case_name(
    k1: float,
    noise: int,
    crop_id: int,
    seed: int,
    k2: float = 0.0,
) -> str:
    """Build the stable filename stem for one benchmark condition."""
    k = f"k1_{k1:+.4f}_noise_{noise:02d}".replace("+", "p").replace("-", "m")
    if abs(k2) > 1e-15:
        q = f"_k2_{k2:+.4f}".replace("+", "p").replace("-", "m")
        k += q
    return f"c{crop_id}_s{seed}_{k}"


def position_case_seed(crop_id: int, seed: int, noise: int) -> int:
    """Return the reproducible seed for a crop, replicate, and noise level."""
    return int(seed + 1_000_003 * crop_id + 1009 * noise)


def generate_benchmark(
    source: Path,
    output: Path,
    grid: int = 3,
    tile_size: int = 1024,
    overlap: float = 0.15,
    crops: Sequence[tuple[int, int]] = DEFAULT_CROPS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    k1_values: Sequence[float] = DEFAULT_K1,
    k2_values: Sequence[float] = (0.0,),
    noise_values: Sequence[int] = DEFAULT_NOISE,
    interp_order: int = 5,
    crop_size: int = 6144,
) -> dict[str, Any]:
    """Write crop images, case NPZ files, and the corresponding manifest."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    step = int(round(tile_size * (1.0 - overlap)))
    extent = (grid - 1) * step + tile_size
    if crop_size < extent:
        raise ValueError(f"crop_size {crop_size} < needed extent {extent}")

    offsets = [i * step for i in range(grid)]
    positions_true = np.array(
        [[y, x] for y in offsets for x in offsets], dtype=np.int64
    )

    manifest_cases = []
    crops_meta = []

    for cid, (cx, cy) in enumerate(crops):
        source_crop = read_window(source, cx, cy, crop_size, crop_size)
        if not np.all(np.isfinite(source_crop)):
            raise ValueError(
                f"source crop {cid} at ({cx}, {cy}) contains non-finite values")
        crop_min = float(np.min(source_crop))
        crop_max = float(np.max(source_crop))
        if crop_max <= crop_min:
            raise ValueError(
                f"source crop {cid} at ({cx}, {cy}) is constant "
                f"(value {crop_min}); choose an informative region")
        crop_file = output / f"source_crop_{cid}.tif"
        tifffile.imwrite(crop_file, source_crop)
        crops_meta.append({
            "id": cid, "xy": [cx, cy],
            "file": crop_file.name,
            "shape": list(source_crop.shape),
            "dtype": np.dtype(source_crop.dtype).name,
            "intensity_min": crop_min,
            "intensity_max": crop_max,
        })

        base_tiles = np.stack(
            [source_crop[y:y + tile_size, x:x + tile_size]
             for y, x in positions_true],
            axis=0,
        )

        for seed_idx, seed in enumerate(seeds):
            for k1 in k1_values:
                for k2 in k2_values:
                    distorted = np.stack(
                        [apply_radial_distortion(t, k1, k2=k2, order=interp_order)
                         for t in base_tiles],
                        axis=0,
                    )
                    if np.issubdtype(source_crop.dtype, np.integer):
                        distorted = np.clip(
                            np.rint(distorted),
                            np.iinfo(source_crop.dtype).min,
                            np.iinfo(source_crop.dtype).max,
                        ).astype(source_crop.dtype)
                    else:
                        distorted = distorted.astype(np.float32)

                    for noise in noise_values:
                        # At zero stage noise every seed produces the same initial
                        # positions.  Keeping duplicate files would inflate the
                        # apparent sample size without adding information.
                        if noise == 0 and seed_idx > 0:
                            continue

                        # Common-random-number design: the position perturbation
                        # depends on content, replicate seed, and noise level, but
                        # not on k1.  Thus every k1 level is tested with the exact
                        # same stage-error realization.
                        case_seed = position_case_seed(cid, seed, noise)
                        rng = np.random.default_rng(case_seed)
                        if noise == 0:
                            position_noise = np.zeros((grid * grid, 2),
                                                      dtype=np.int64)
                        else:
                            position_noise = rng.integers(
                                -noise, noise + 1, size=(grid * grid, 2)
                            )
                        positions_ini = positions_true + position_noise

                        name = case_name(k1, noise, cid, seed, k2=k2)
                        case_path = output / "cases" / f"{name}.npz"
                        case_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            case_path,
                            tiles=distorted,
                            positions_true=positions_true,
                            positions_ini=positions_ini,
                            true_k1=np.array(k1, dtype=np.float64),
                            true_k2=np.array(k2, dtype=np.float64),
                            position_noise=position_noise,
                        )
                        manifest_cases.append({
                            "case": name,
                            "file": case_path.relative_to(output).as_posix(),
                            "crop_id": cid,
                            "seed": seed,
                            "k1_true": k1,
                            "k2_true": k2,
                            "position_noise_max_px": noise,
                            "case_seed": case_seed,
                            "tiles_shape": list(distorted.shape),
                            "tiles_dtype": np.dtype(distorted.dtype).name,
                            "positions_true": positions_true.tolist(),
                        })

    metadata = {
        "source_file": str(source),
        "grid": [grid, grid],
        "tile_size": tile_size,
        "overlap": overlap,
        "step": step,
        "extent": extent,
        "interpolation_order": interp_order,
        "distortion_model": "shared k1/k2 radial model per tile, local "
        "tile-centered coordinates",
        "position_noise_distribution": "uniform integer in [-level, level] "
        "independently for y and x",
        "position_noise_pairing": "same realization reused across k1 levels "
        "within (crop, seed, noise)",
        "zero_noise_seed_policy": "only the first seed is retained because all "
        "zero-noise seeds are identical",
        "k1_values": list(k1_values),
        "k2_values": list(k2_values),
        "noise_values": list(noise_values),
        "seeds": list(seeds),
        "crops": crops_meta,
        "case_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    return metadata


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    """Parse semicolon-separated integer coordinate pairs."""
    out = []
    for part in text.split(";"):
        x, y = (int(v) for v in part.split(","))
        out.append((x, y))
    return out


def main() -> None:
    """Parse CLI options and generate the requested benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, default=6144)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--crops", type=str,
                        default=";".join(f"{x},{y}" for x, y in DEFAULT_CROPS))
    parser.add_argument("--seeds", type=str, default="2026,2027")
    parser.add_argument("--k1", type=str,
                        default=",".join(f"{v:.4f}" for v in DEFAULT_K1))
    parser.add_argument("--k2", type=str, default="0.0000",
                        help="generation-only higher-order radial coefficients; "
                        "estimators remain k1-only")
    parser.add_argument("--noise", type=str,
                        default=",".join(str(v) for v in DEFAULT_NOISE))
    parser.add_argument("--interp-order", type=int, default=5)
    args = parser.parse_args()

    k1_values = tuple(float(v) for v in args.k1.split(","))
    k2_values = tuple(float(v) for v in args.k2.split(","))
    noise_values = tuple(int(v) for v in args.noise.split(","))
    seeds = tuple(int(v) for v in args.seeds.split(","))
    crops = _parse_pairs(args.crops)

    metadata = generate_benchmark(
        source=args.source,
        output=args.output,
        grid=args.grid,
        tile_size=args.tile_size,
        crop_size=args.crop_size,
        overlap=args.overlap,
        crops=crops,
        seeds=seeds,
        k1_values=k1_values,
        k2_values=k2_values,
        noise_values=noise_values,
        interp_order=args.interp_order,
    )
    print(json.dumps({
        "output": str(args.output),
        "grid": metadata["grid"],
        "tile_size": metadata["tile_size"],
        "case_count": metadata["case_count"],
        "interpolation_order": metadata["interpolation_order"],
        "crops": len(metadata["crops"]),
        "seeds": metadata["seeds"],
    }, indent=2))


if __name__ == "__main__":
    main()
