# Sequential baseline: estimate k1 once, correct once, then stitch.
# Uses the same primitives as the joint method; the only difference is
# that the loop is run a single time with no feedback.

from collections import namedtuple

import numpy as np

from distortcorrect_stitcher_gpu import (
    _gpu_backend,
    _integer_positions,
    _require_cuda,
    assemble,
    correct_tiles_once,
    estimate_k1_once,
    pr_stitching,
    sharpen_tiles,
)


SequentialResult = namedtuple(
    "SequentialResult",
    ["positions", "k1", "k1_history", "tiles", "mosaic", "positions_ini"],
)


class SequentialStitcher:
    def __init__(self, k1_bounds=(-0.015, 0.015), interpolation_order=3,
                 gamma_schedule=None, ncc_threshold=0.0, sharpen=False,
                 boundary_frac=0.25, boundary_px=None, local_search=5, border=64,
                 skip_diagonal=True):
        self.k1_bounds = k1_bounds
        self.order = interpolation_order
        self.gamma_schedule = gamma_schedule
        self.ncc_threshold = ncc_threshold
        self.sharpen = sharpen
        self.boundary_frac = boundary_frac
        self.boundary_px = boundary_px
        self.local_search = local_search
        self.border = border
        self.skip_diagonal = skip_diagonal

    def estimate_k1(self, tiles, positions, outlier_tiles=None):
        return estimate_k1_once(
            tiles,
            positions,
            k1_bounds=self.k1_bounds,
            order=self.order,
            boundary_frac=self.boundary_frac,
            boundary_px=self.boundary_px,
            local_search=self.local_search,
            outlier_tiles=outlier_tiles,
            skip_diagonal=self.skip_diagonal,
        )

    def run(self, tiles, positions_ini, outlier_tiles=None,
            do_sharpen=True, verbose=False):
        tiles_proc = np.asarray(tiles).copy()
        if tiles_proc.ndim not in (3, 4):
            raise ValueError("tiles must have shape (t, y, x) or (t, c, y, x)")
        positions_ini = _integer_positions(positions_ini, tiles_proc.shape[0])

        if do_sharpen:
            tiles_proc = sharpen_tiles(tiles_proc)

        k1 = self.estimate_k1(
            tiles_proc,
            positions_ini,
            outlier_tiles=outlier_tiles,
        )
        tiles_corrected = correct_tiles_once(
            tiles_proc,
            k1,
            order=self.order,
        )

        positions = pr_stitching(
            tiles_corrected,
            positions_ini,
            gamma_schedule=self.gamma_schedule,
            ncc_threshold=self.ncc_threshold,
            sharpen=self.sharpen,
            skip_diagonal=self.skip_diagonal,
            verbose=verbose,
        )
        mosaic = assemble(tiles_corrected, positions, border=self.border)

        return SequentialResult(
            positions=positions,
            k1=k1,
            k1_history=k1.reshape(1, -1),
            tiles=tiles_corrected,
            mosaic=mosaic,
            positions_ini=positions_ini,
        )


class SequentialStitcherGPU(SequentialStitcher):
    def __init__(self, *args, device=None, **kwargs):
        self.device = str(_require_cuda(device))
        super().__init__(*args, **kwargs)

    def run(self, tiles, positions_ini, outlier_tiles=None,
            do_sharpen=True, verbose=False):
        with _gpu_backend(self.device):
            return super().run(
                tiles,
                positions_ini,
                outlier_tiles=outlier_tiles,
                do_sharpen=do_sharpen,
                verbose=verbose,
            )
