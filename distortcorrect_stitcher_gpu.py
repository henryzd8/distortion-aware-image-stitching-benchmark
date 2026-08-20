"""CPU stitching primitives with an optional CUDA execution adapter."""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize_scalar
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import networkx as nx
from skimage.registration import phase_cross_correlation
from skimage.filters import unsharp_mask
from tqdm import tqdm


# Radial distortion model: r_d = r_u * (1 + k1*r_u^2 + k2*r_u^4 + k3*r_u^6)

def radial_distortion_map(
    coords: np.ndarray,
    h: int,
    w: int,
    k1: float,
    k2: float = 0.0,
    k3: float = 0.0,
) -> np.ndarray:
    """Map undistorted pixel coordinates through the radial model."""
    y_u, x_u = coords
    x_c = (x_u - w / 2.0) / (w / 2.0)
    y_c = (y_u - h / 2.0) / (h / 2.0)
    r2 = x_c ** 2 + y_c ** 2
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = x_c * radial * (w / 2.0) + w / 2.0
    y_d = y_c * radial * (h / 2.0) + h / 2.0
    return np.vstack((y_d, x_d))


def _saturate(arr, dtype):
    # spline overshoot wraps on integer dtypes, so clip first
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(arr), info.min, info.max).astype(dtype)
    return arr.astype(dtype)


def undistort_image(
    image: np.ndarray,
    k1: float,
    k2: float = 0.0,
    k3: float = 0.0,
    order: int = 3,
    mode: str = "nearest",
) -> np.ndarray:
    """Correct one grayscale or channel-first image for radial distortion."""
    h, w = image.shape[-2:]
    y_idx, x_idx = np.indices((h, w), dtype=np.float64)
    coords = np.stack((y_idx.ravel(), x_idx.ravel()))
    coords_d = radial_distortion_map(coords, h, w, k1, k2, k3)
    y_d = coords_d[0].reshape(h, w)
    x_d = coords_d[1].reshape(h, w)

    if image.ndim == 2:
        return _saturate(
            map_coordinates(image, [y_d, x_d], order=order, mode=mode),
            image.dtype,
        )
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        out[c] = _saturate(
            map_coordinates(image[c], [y_d, x_d], order=order, mode=mode),
            image.dtype,
        )
    return out


def undistort_tiles(
    tiles: np.ndarray,
    k1s: Sequence[float],
    order: int = 3,
    mode: str = "nearest",
) -> np.ndarray:
    """Apply one or more cumulative distortion corrections to tile data."""
    corrected = tiles.copy()
    for k1 in k1s:
        corrected = np.array([
            undistort_image(tile, k1, order=order, mode=mode)
            for tile in corrected
        ])
    return corrected


# NCC (normalized cross-correlation)

def ncc(image1: np.ndarray, image2: np.ndarray) -> float:
    """Return normalized cross-correlation, or zero for constant inputs."""
    a = image1.ravel().astype(np.float64)
    b = image2.ravel().astype(np.float64)
    a_ms = a - a.mean()
    b_ms = b - b.mean()
    denom = np.linalg.norm(a_ms) * np.linalg.norm(b_ms)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_ms, b_ms) / denom)


# Overlap geometry: bounding-box intersection of two same-sized tiles

def find_overlap_coords(
    y1: int,
    x1: int,
    y2: int,
    x2: int,
    h: int,
    w: int,
) -> tuple[Any, Any]:
    """Return local overlap slices for two same-sized positioned tiles."""
    ox_start = max(x1, x2)
    ox_end = min(x1 + w, x2 + w)
    oy_start = max(y1, y2)
    oy_end = min(y1 + h, y2 + h)

    if ox_start >= ox_end or oy_start >= oy_end:
        return None, None

    return (
        (oy_start - y1, oy_end - y1, ox_start - x1, ox_end - x1),
        (oy_start - y2, oy_end - y2, ox_start - x2, ox_end - x2),
    )


# Pairwise shift estimation

def _normalize_overlap(ref, tar, sharpen):
    # Normalize in float32. A blank or constant overlap has no translation
    # information, so return None instead of sending it to MI alignment.
    ref = np.asarray(ref, dtype=np.float32).copy()
    tar = np.asarray(tar, dtype=np.float32).copy()

    def to_uint8(image):
        if not np.all(np.isfinite(image)):
            return None
        lo = float(image.min())
        hi = float(image.max())
        if hi <= 0 or hi - lo <= 1e-12:
            return None
        return np.uint8(np.clip(image / hi, 0, 1) * 255)

    if ref.ndim == 3:  # (c, h, w)
        for c in range(ref.shape[0]):
            mx = ref[c].max()
            if mx > 0:
                ref[c] = ref[c] / mx
            mx = tar[c].max()
            if mx > 0:
                tar[c] = tar[c] / mx
        if sharpen:
            ref = np.array([unsharp_mask(r, radius=1, amount=1) for r in ref])
            tar = np.array([unsharp_mask(t, radius=1, amount=1) for t in tar])
        ref8 = [to_uint8(r) for r in ref]
        tar8 = [to_uint8(t) for t in tar]
        if any(r is None for r in ref8) or any(t is None for t in tar8):
            return None, None
        ref8 = np.stack(ref8)
        tar8 = np.stack(tar8)
    else:  # (h, w)
        mx = ref.max()
        if mx > 0:
            ref = ref / mx
        mx = tar.max()
        if mx > 0:
            tar = tar / mx
        if sharpen:
            ref = unsharp_mask(ref, radius=1, amount=1)
            tar = unsharp_mask(tar, radius=1, amount=1)
        ref8 = to_uint8(ref)
        tar8 = to_uint8(tar)
        if ref8 is None or tar8 is None:
            return None, None
    return ref8, tar8


def cal_shift_ij(
    i: int,
    j: int,
    tiles: np.ndarray,
    positions0: np.ndarray,
    h: int,
    w: int,
    gamma: float = 0.1,
    sharpen: bool = True,
) -> np.ndarray | None:
    """Estimate a pairwise integer shift from the current tile overlap."""
    positions = positions0.copy()
    pos_i = positions[i]
    pos_j = positions[j]
    dydx_out = np.array([0, 0])
    aligned = False

    for _ in range(2):  # two sub-iterations of refinement
        yx1, yx2 = find_overlap_coords(
            pos_i[0], pos_i[1], pos_j[0], pos_j[1], h, w)
        if yx1 is not None and yx2 is not None:
            ref = tiles[i, ..., yx1[0]:yx1[1], yx1[2]:yx1[3]].copy()
            tar = tiles[j, ..., yx2[0]:yx2[1], yx2[2]:yx2[3]].copy()

            ref, tar = _normalize_overlap(ref, tar, sharpen)
            if ref is None or tar is None:
                break

            dydx = _align_pairwise(ref, tar, gamma)
            aligned = True
            pos_j = pos_j - dydx
        else:
            dydx = np.array([0, 0])
        dydx_out += dydx

    return dydx_out if aligned else None


def _align_pairwise(ref, tar, gamma):
    if ref.ndim == 2:
        shift = _mi_align_cpu(ref, tar)
    else:
        shifts = []
        for c in range(ref.shape[0]):
            shift_c = _mi_align_cpu(ref[c], tar[c])
            shifts.append(shift_c)
        shifts = np.array(shifts)
        shift = np.median(shifts, axis=0)
    return np.round(shift * gamma).astype(int)


# Mutual-information alignment (port of mutualinformation_single.py)

def _mi_align_cpu(ref, tar):
    try:
        import torch
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        # fallback: phase cross-correlation
        shift, _, _ = phase_cross_correlation(ref, tar, upsample_factor=10)
        return np.array([shift[0], shift[1]])

    h, w = ref.shape
    Q = 16

    ref8 = ref.copy().astype(np.float32)
    tar8 = tar.copy().astype(np.float32)

    ref_q = image2cat_kmeans(ref8, Q)
    tar_q = image2cat_kmeans(tar8, Q)
    M = np.ones((h, w), dtype='bool')

    param, _ = _align_translation_cpu(ref_q, tar_q, M, M, Q, Q,
                                       overlap=0.5)
    return np.array([param[1], param[2]])


def image2cat_kmeans(
    I: np.ndarray,
    k: int,
    batch_size: int = 10000,
    max_iter: int = 300,
    random_seed: int = 1000,
) -> np.ndarray:
    """Quantize an image into integer MiniBatchKMeans labels."""
    # The benchmark-wide accelerated profile is explicit and provenance-tracked.
    # The author's 100/1000 profile remains available through explicit arguments
    # for limited source-faithful calibration runs.
    from sklearn.cluster import MiniBatchKMeans
    if k == 1:
        return np.zeros(I.shape, dtype=np.int32)
    I_lin = I.reshape(-1, 1)
    km = MiniBatchKMeans(n_clusters=k, max_iter=max_iter,
                         batch_size=batch_size,
                         random_state=random_seed).fit(I_lin)
    return km.labels_.reshape(I.shape)


def _align_translation_cpu(A, B, M_A, M_B, Q_A, Q_B, overlap=0.5):
    # MI-based translation alignment via FFT (Oefverstedt 2021, MIT).
    import torch
    import torch.nn.functional as F

    eps = 1e-7
    VALUE_TYPE = torch.float32

    def compute_entropy(C, N, eps_loc=1e-7):
        p = C / N
        return p * torch.log2(torch.clamp(p, min=eps_loc))

    def float_compare(A, c):
        return torch.clamp(1 - torch.abs(A - c), 0.0)

    def fft(A):
        return torch.fft.rfft2(A)

    def ifft(Afft):
        return torch.fft.irfft2(Afft)

    def fftconv(A, B):
        return A * B

    def corr_target_setup(A):
        return fft(A)

    def corr_template_setup(B):
        return torch.conj(fft(B))

    def corr_apply(A, B, sz, do_rounding=True):
        C = fftconv(A, B)
        C = ifft(C)
        C = C[:sz[0], :sz[1], :sz[2], :sz[3]]
        if do_rounding:
            C = torch.round(C)
        return C

    def create_float_tensor(shape, fill_value=None):
        if isinstance(shape, torch.Tensor):
            shape = tuple(shape.tolist())
        if fill_value is not None:
            res = np.full(shape, fill_value=fill_value, dtype='float32')
        else:
            res = np.zeros(shape, dtype='float32')
        return torch.tensor(res, dtype=torch.float32)

    def to_tensor(A):
        if torch.is_tensor(A):
            A_tensor = A
            if A_tensor.ndim == 2:
                A_tensor = torch.reshape(A_tensor, (1, 1,
                                         A_tensor.shape[0],
                                         A_tensor.shape[1]))
            elif A_tensor.ndim == 3:
                A_tensor = torch.reshape(A_tensor, (1,
                                         A_tensor.shape[0],
                                         A_tensor.shape[1],
                                         A_tensor.shape[2]))
            return A_tensor
        else:
            return to_tensor(torch.tensor(A, dtype=VALUE_TYPE))

    def fft_of_levelsets(A, Q, packing, setup_fn):
        fft_list = []
        for a_start in range(0, Q, packing):
            a_end = min(a_start + packing, Q)
            levelsets = []
            for a in range(a_start, a_end):
                levelsets.append(float_compare(A, a))
            A_cat = torch.cat(levelsets, 0)
            del levelsets
            ffts = setup_fn(A_cat)
            del A_cat
            fft_list.append((ffts, a_start, a_end))
        return fft_list

    A_tensor = to_tensor(A)
    B_tensor = to_tensor(B)

    if A_tensor.shape[-1] < 1024:
        packing = min(Q_B, 64)
    elif A_tensor.shape[-1] <= 2048:
        packing = min(Q_B, 8)
    elif A_tensor.shape[-1] <= 4096:
        packing = min(Q_B, 4)
    else:
        packing = min(Q_B, 1)

    if M_A is None:
        M_A = create_float_tensor(A_tensor.shape, 1.0)
    else:
        M_A = to_tensor(M_A)
        A_tensor = torch.round(M_A * A_tensor + (1 - M_A) * (Q_A + 1))
    if M_B is None:
        M_B = create_float_tensor(B_tensor.shape, 1.0)
    else:
        M_B = to_tensor(M_B)

    partial_overlap_pad_sz = (round(B.shape[-1] * (1.0 - overlap)),
                               round(B.shape[-2] * (1.0 - overlap)))
    A_tensor = F.pad(A_tensor, (partial_overlap_pad_sz[0],
                                partial_overlap_pad_sz[0],
                                partial_overlap_pad_sz[1],
                                partial_overlap_pad_sz[1]),
                     mode='constant', value=Q_A + 1)
    M_A = F.pad(M_A, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0],
                       partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]),
                mode='constant', value=0)

    ext_ashape = A_tensor.shape
    ext_bshape = B_tensor.shape
    b_pad_shape = (torch.tensor(A_tensor.shape, dtype=torch.long)
                   - torch.tensor(B_tensor.shape, dtype=torch.long))
    ext_valid_shape = b_pad_shape + 1
    batched_valid_shape = ext_valid_shape + torch.tensor([packing - 1,
                                                           0, 0, 0])

    M_A_FFT = corr_target_setup(M_A)

    A_ffts = []
    for a in range(Q_A):
        A_ffts.append(corr_target_setup(float_compare(A_tensor, a)))
    del A_tensor, M_A

    MI = create_float_tensor(ext_valid_shape, 0.0)

    B_tensor_padded = F.pad(B_tensor,
                            (0, ext_ashape[-1] - ext_bshape[-1],
                             0, ext_ashape[-2] - ext_bshape[-2],
                             0, 0, 0, 0),
                            mode='constant', value=Q_B + 1)
    M_B_padded = F.pad(M_B,
                       (0, ext_ashape[-1] - ext_bshape[-1],
                        0, ext_ashape[-2] - ext_bshape[-2],
                        0, 0, 0, 0),
                       mode='constant', value=0)
    B_tensor_padded = torch.round(M_B_padded * B_tensor_padded
                                  + (1 - M_B_padded) * (Q_B + 1))

    M_B_FFT = corr_template_setup(M_B_padded)
    N = torch.clamp(corr_apply(M_A_FFT, M_B_FFT, ext_valid_shape),
                    min=eps)

    b_ffts = fft_of_levelsets(B_tensor_padded, Q_B, packing,
                              corr_template_setup)

    for bext in range(len(b_ffts)):
        b_fft = b_ffts[bext]
        E_M = torch.sum(compute_entropy(
            corr_apply(M_A_FFT, b_fft[0], batched_valid_shape),
            N, eps), dim=0)
        MI = torch.sub(MI, E_M)
        del E_M

        for a in range(Q_A):
            A_fft_curr = A_ffts[a]
            if bext == 0:
                E_M = compute_entropy(
                    corr_apply(A_fft_curr, M_B_FFT, ext_valid_shape),
                    N, eps)
                MI = torch.sub(MI, E_M)
                del E_M
            E_J = torch.sum(compute_entropy(
                corr_apply(A_fft_curr, b_fft[0], batched_valid_shape),
                N, eps), dim=0)
            MI = torch.add(MI, E_J)
            del E_J, A_fft_curr
        del b_fft
        if bext == 0:
            del M_B_FFT

    del B_tensor_padded

    (max_n, _) = torch.max(torch.reshape(N, (-1,)), 0)
    N_filt = torch.lt(N, overlap * max_n)
    MI[N_filt] = 0.0
    del N_filt, N

    MI_vec = torch.reshape(MI, (-1,))
    (val, ind) = torch.max(MI_vec, -1)

    sz_x = int(ext_valid_shape[3])
    y = ind // sz_x
    x = ind % sz_x

    translation_y = -(y - partial_overlap_pad_sz[1])
    translation_x = -(x - partial_overlap_pad_sz[0])

    val = val.item()
    translation_y = translation_y.item()
    translation_x = translation_x.item()

    return (val, translation_y, translation_x), None


# k1 estimation / correction (shared by the joint loop and the sequential
# baseline; the only difference between the two methods is the loop)

def _integer_positions(positions, n_tiles):
    arr = np.asarray(positions)
    if arr.ndim != 2 or arr.shape != (n_tiles, 2):
        raise ValueError("positions must have shape (number_of_tiles, 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("positions must contain finite values")
    if not np.allclose(arr, np.round(arr)):
        raise ValueError("positions must contain integer pixel coordinates")
    return np.round(arr).astype(int)


def estimate_k1_once(
    tiles: np.ndarray,
    positions: np.ndarray,
    k1_bounds: tuple[float, float] = (-0.015, 0.015),
    order: int = 3,
    boundary_frac: float = 0.25,
    boundary_px: int | None = None,
    local_search: int = 5,
    outlier_tiles: Sequence[int] | None = None,
    k1_fixed: float = 0.0,
    gamma: float = 1.0,
    skip_diagonal: bool = True,
) -> np.ndarray:
    """Estimate a bounded distortion increment from overlap-boundary NCC."""
    # per-channel bounded search for the best k1 increment around
    # k1_fixed, damped by gamma (gamma=1, k1_fixed=0 = one-shot)
    arr = np.asarray(tiles)
    if arr.ndim not in (3, 4):
        raise ValueError("tiles must have shape (t, y, x) or (t, c, y, x)")
    positions = _integer_positions(positions, arr.shape[0])
    if k1_bounds[0] >= k1_bounds[1]:
        raise ValueError("k1_bounds must be an increasing (low, high) pair")

    if outlier_tiles is None:
        good = np.arange(arr.shape[0])
    else:
        excluded = set(outlier_tiles)
        if any(i < 0 or i >= arr.shape[0] for i in excluded):
            raise ValueError("outlier_tiles contains an invalid tile index")
        good = np.array([i for i in range(arr.shape[0]) if i not in excluded])
    if len(good) < 2:
        raise ValueError("at least two non-outlier tiles are required")

    if arr.ndim == 3:
        channels = [arr[good]]
    else:
        channels = [arr[good, c] for c in range(arr.shape[1])]

    estimates = []
    for channel_tiles in channels:
        channel_positions = positions[good]

        def objective(k):
            return cost_boundary_ncc(
                float(k1_fixed) + k,
                channel_tiles,
                channel_positions,
                order=order,
                boundary_frac=boundary_frac,
                boundary_px=boundary_px,
                local_search=local_search, skip_diagonal=skip_diagonal,
            )

        result = minimize_scalar(
            objective,
            bounds=k1_bounds,
            method="bounded",
            options={"xatol": 1e-7},
        )
        estimates.append(float(result.x) * gamma)

    return np.asarray(estimates, dtype=float)


def correct_tiles_once(
    tiles: np.ndarray,
    k1: Sequence[float] | np.ndarray,
    order: int = 3,
    mode: str = "nearest",
) -> np.ndarray:
    """Correct grayscale or multichannel tiles using supplied coefficients."""
    arr = np.asarray(tiles)
    estimates = np.asarray(k1, dtype=float).reshape(-1)

    if arr.ndim == 3:
        if estimates.size != 1:
            raise ValueError("grayscale tiles require exactly one k1 estimate")
        return undistort_tiles(arr, [float(estimates[0])], order=order, mode=mode)

    if arr.ndim != 4:
        raise ValueError("tiles must have shape (t, y, x) or (t, c, y, x)")
    if estimates.size != arr.shape[1]:
        raise ValueError("multichannel tiles require one k1 estimate per channel")

    corrected = np.empty_like(arr)
    for c, value in enumerate(estimates):
        corrected[:, c] = undistort_tiles(
            arr[:, c], [float(value)], order=order, mode=mode
        )
    return corrected


# Graph construction & position optimization

def construct_matrix_colors(
    tiles: np.ndarray,
    positions: np.ndarray,
    gamma: float = 1.0,
    ncc_threshold: float = 0.0,
    sharpen: bool = True,
    skip_diagonal: bool = True,
) -> nx.Graph:
    """Build a weighted overlap graph from the current tile layout."""
    G = nx.Graph()
    num_tiles = len(tiles)
    h, w = tiles.shape[-2:]

    for i in range(num_tiles):
        G.add_node(i)

    for i in range(num_tiles):
        for j in range(i + 1, num_tiles):
            yx1, yx2 = find_overlap_coords(
                positions[i, 0], positions[i, 1],
                positions[j, 0], positions[j, 1], h, w)

            if yx1 is None:
                continue

            area = (yx1[1] - yx1[0]) * (yx1[3] - yx1[2])

            # skip diagonal corner overlaps
            ov_h = yx1[1] - yx1[0]
            ov_w = yx1[3] - yx1[2]
            if skip_diagonal and ov_h < h * 0.4 and ov_w < w * 0.4:
                continue

            # NCC edge filter (0 = off)
            if ncc_threshold > 0:
                ref = tiles[i, ..., yx1[0]:yx1[1], yx1[2]:yx1[3]]
                tar = tiles[j, ..., yx2[0]:yx2[1], yx2[2]:yx2[3]]
                if ref.ndim == 3:
                    corr = np.mean([ncc(ref[c], tar[c])
                                    for c in range(ref.shape[0])])
                else:
                    corr = ncc(ref, tar)
                if corr < ncc_threshold:
                    continue

            dydx = cal_shift_ij(i, j, tiles, positions, h, w,
                                gamma=gamma, sharpen=sharpen)
            if dydx is None:
                continue
            G.add_edge(j, i, weight=area, shift=dydx)

    return G


def extract_data_from_graph(
    G: nx.Graph,
) -> tuple[
    int,
    list[tuple[int, int]],
    list[np.ndarray],
    list[float],
    dict[int, int],
    dict[int, int],
]:
    """Extract pairwise shifts and node-index mappings from an overlap graph."""
    nodes = list(G.nodes())
    num_tiles = len(nodes)

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    pairs = []
    shifts_ij = []
    weights = []

    for i, j, data in G.edges(data=True):
        idx_i = node_to_idx[i]
        idx_j = node_to_idx[j]
        pairs.append((idx_i, idx_j))
        shifts_ij.append(np.array(data['shift']))
        weights.append(data['weight'])

    return num_tiles, pairs, shifts_ij, weights, node_to_idx, idx_to_node


def optimize_shifts_with_graph(
    G: nx.Graph,
    positions: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Solve weighted graph constraints and return corrected positions."""
    # weighted least squares over the pairwise-shift graph
    num_tiles, pairs, shifts_ij, weights, node_to_idx, idx_to_node = \
        extract_data_from_graph(G)
    fixed_tile_idx = node_to_idx[0] if 0 in node_to_idx else 0  # fix tile 0

    variable_indices = [idx for idx in range(num_tiles)
                        if idx != fixed_tile_idx]
    num_variables = 2 * len(variable_indices)
    num_equations = 2 * len(pairs)

    if num_equations == 0:
        return positions.copy(), {}

    tile_to_var = {idx: var_idx * 2
                   for var_idx, idx in enumerate(variable_indices)}

    A = lil_matrix((num_equations, num_variables))
    b = np.zeros(num_equations)

    eq = 0
    for (idx_i, idx_j), shift_ij, w_ij in zip(pairs, shifts_ij, weights):
        sqrt_w = np.sqrt(w_ij)
        i_var = idx_i != fixed_tile_idx
        j_var = idx_j != fixed_tile_idx

        # y-equation:  pos_j_y - pos_i_y = shift_ij_y
        if i_var:
            A[eq, tile_to_var[idx_i]] = -sqrt_w
        if j_var:
            A[eq, tile_to_var[idx_j]] = sqrt_w
        b[eq] = sqrt_w * shift_ij[0]
        eq += 1

        # x-equation
        if i_var:
            A[eq, tile_to_var[idx_i] + 1] = -sqrt_w
        if j_var:
            A[eq, tile_to_var[idx_j] + 1] = sqrt_w
        b[eq] = sqrt_w * shift_ij[1]
        eq += 1

    A = A.tocsr()
    result = lsqr(A, b)
    x = result[0]

    shifts = np.zeros((num_tiles, 2))
    for idx in range(num_tiles):
        if idx == fixed_tile_idx:
            shifts[idx] = [0.0, 0.0]
        else:
            vi = tile_to_var[idx]
            shifts[idx] = x[vi:vi + 2]

    shifts_ordered = {idx_to_node[idx]: shifts[idx] for idx in range(num_tiles)}

    positions_corrected = positions.copy()
    for node in G.nodes():
        positions_corrected[node] -= np.round(
            shifts_ordered[node]).astype(int)

    return positions_corrected, shifts_ordered


# Pre-stitching (coarse-to-fine gamma schedule)

_DEFAULT_GAMMA_SCHEDULE = [0.1] * 4 + [0.2] * 3 + [0.3] * 2 + \
    [0.4, 0.5, 0.6, 0.8, 0.9, 1]


def pr_stitching(
    tiles: np.ndarray,
    positions: np.ndarray,
    gamma_schedule: Sequence[float] | None = None,
    ncc_threshold: float = 0.0,
    sharpen: bool = True,
    skip_diagonal: bool = True,
    verbose: bool = False,
) -> np.ndarray:
    """Refine positions with a coarse-to-fine overlap-registration schedule."""
    if gamma_schedule is None:
        gamma_schedule = _DEFAULT_GAMMA_SCHEDULE

    positions = positions.copy()
    iterator = tqdm(gamma_schedule, desc="Pre-stitching") \
        if verbose else gamma_schedule

    for gamma in iterator:
        G = construct_matrix_colors(
            tiles, positions, gamma=gamma,
            ncc_threshold=ncc_threshold, sharpen=sharpen,
            skip_diagonal=skip_diagonal)
        positions, _ = optimize_shifts_with_graph(G, positions)

    return positions


# Boundary NCC cost function

def cost_boundary_ncc(
    k1: float,
    tiles: np.ndarray,
    positions: np.ndarray,
    order: int = 3,
    boundary_frac: float = 0.25,
    boundary_px: int | None = None,
    local_search: int = 5,
    skip_diagonal: bool = True,
) -> float:
    """Score a candidate coefficient using boundary-strip NCC."""
    tiles_corr = undistort_tiles(tiles, [k1], order=order)

    total_ncc = 0.0
    num_tiles = len(tiles_corr)
    h, w = tiles_corr.shape[-2:]

    for i in range(num_tiles):
        for j in range(i + 1, num_tiles):
            yx1, yx2 = find_overlap_coords(
                positions[i, 0], positions[i, 1],
                positions[j, 0], positions[j, 1], h, w)

            if yx1 is None:
                continue

            overlap_h = yx1[1] - yx1[0]
            overlap_w = yx1[3] - yx1[2]

            if skip_diagonal and overlap_h < h * 0.4 and overlap_w < w * 0.4:
                continue

            ov_i = tiles_corr[i, yx1[0]:yx1[1], yx1[2]:yx1[3]]
            ov_j = tiles_corr[j, yx2[0]:yx2[1], yx2[2]:yx2[3]]

            if overlap_w < overlap_h:
                # lateral overlap -> top/bottom boundary strips
                bh = (max(1, min(int(boundary_px), overlap_h // 2))
                      if boundary_px is not None else
                      max(1, min(int(overlap_h * boundary_frac),
                                 overlap_h // 2)))
                si = np.vstack((ov_i[:bh, :], ov_i[-bh:, :]))
                sj = np.vstack((ov_j[:bh, :], ov_j[-bh:, :]))
            else:
                # stacked overlap -> left/right boundary strips
                bw = (max(1, min(int(boundary_px), overlap_w // 2))
                      if boundary_px is not None else
                      max(1, min(int(overlap_w * boundary_frac),
                                 overlap_w // 2)))
                si = np.hstack((ov_i[:, :bw], ov_i[:, -bw:]))
                sj = np.hstack((ov_j[:, :bw], ov_j[:, -bw:]))

            total_ncc += _local_ncc_max(si, sj, local_search)

    return -total_ncc


def _local_ncc_max(ref, tar, local_search=5):
    # best NCC within a +/-local_search window (0 = rigid NCC)
    sh, sw = ref.shape
    margin = local_search

    if margin <= 0:
        return ncc(ref, tar)

    if sh < 2 * margin + 3 or sw < 2 * margin + 3:
        return ncc(ref, tar)

    ref_crop = ref[margin:-margin, margin:-margin]
    ch, cw = ref_crop.shape

    ref_n = ref_crop.ravel().astype(np.float64)
    ref_ms = ref_n - ref_n.mean()
    norm_ref = np.linalg.norm(ref_ms)
    if norm_ref < 1e-12:
        return 0.0

    best_ncc = -1.0

    for dy in range(-margin, margin + 1):
        y_start = margin + dy
        y_end = y_start + ch
        if y_end > sh:
            continue
        tar_y = tar[y_start:y_end, :]

        for dx in range(-margin, margin + 1):
            x_start = margin + dx
            x_end = x_start + cw
            if x_end > sw:
                continue
            tar_crop = tar_y[:, x_start:x_end]

            tar_n = tar_crop.ravel().astype(np.float64)
            tar_ms = tar_n - tar_n.mean()
            norm_tar = np.linalg.norm(tar_ms)
            if norm_tar < 1e-12:
                continue
            val = float(np.dot(ref_ms, tar_ms) / (norm_ref * norm_tar))
            if val > best_ncc:
                best_ncc = val

    return best_ncc


# Joint optimization loop

def compute_k1_recursive_colors(
    tiles: np.ndarray,
    positions: np.ndarray,
    outlier_tiles: Sequence[int] | None = None,
    k1_bounds: tuple[float, float] = (-0.005, 0.005),
    gamma: float = 0.5,
    n_iterations: int = 25,
    k1_tol: float = 0.0,
    order: int = 3,
    ncc_threshold: float = 0.0,
    sharpen: bool = True,
    boundary_frac: float = 0.25,
    boundary_px: int | None = None,
    position_damping: float = 0.2,
    local_search: int = 5,
    update_order: str = "positions_first",
    correction_mode: str = "pristine_cumulative",
    skip_diagonal: bool = True,
    oracle_k1: float | Sequence[float] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Alternate graph-based position refinement and radial updates."""
    # positions_first: stitch -> estimate k1 -> re-correct (paper order).
    # distortion_first: estimate k1 -> correct -> stitch (stabilised).
    # k1_tol: stop once every increment stays below it for two iterations.
    # correction_mode selects the benchmark adaptation (fresh correction from
    # pristine tiles) or the paper-style incremental resampling path.
    # oracle_k1: diagnostic control; inject the exact residual each iteration
    # instead of estimating it, so the cumulative correction equals oracle_k1.
    was_3d = tiles.ndim == 3
    if was_3d:
        tiles = tiles[:, np.newaxis, :, :]

    ncolors = tiles.shape[1]
    tiles_original = tiles.copy()  # kept pristine for fresh correction
    k1_cumul = np.zeros(ncolors)
    if oracle_k1 is None:
        oracle = None
    else:
        oracle = np.broadcast_to(
            np.asarray(oracle_k1, dtype=float), (ncolors,)).copy()

    if outlier_tiles is None:
        outlier_tiles = []
    good_ids = list(set(range(len(tiles))) - set(outlier_tiles))

    k1_history = []
    pos_history = []
    iterator = tqdm(range(n_iterations), desc="Joint opt") \
        if verbose else range(n_iterations)

    if update_order not in ("positions_first", "distortion_first"):
        raise ValueError(
            "update_order must be 'positions_first' or 'distortion_first'")
    if correction_mode not in ("pristine_cumulative", "incremental"):
        raise ValueError(
            "correction_mode must be 'pristine_cumulative' or 'incremental'")

    def _stitch_once(positions_in):
        G = construct_matrix_colors(
            tiles_stage, positions_in, gamma=gamma,
            ncc_threshold=ncc_threshold, sharpen=sharpen,
            skip_diagonal=skip_diagonal)
        positions_new, _ = optimize_shifts_with_graph(G, positions_in)
        delta = positions_new.astype(float) - positions_in.astype(float)
        return positions_in + np.round(delta * position_damping).astype(int)

    def _estimate_k1_once():
        k1color = []
        for c in range(ncolors):
            if oracle is not None:
                inc = float(oracle[c]) - k1_cumul[c]
            else:
                if correction_mode == "pristine_cumulative":
                    estimate_tiles = tiles_original[good_ids, c]
                    k1_fixed = float(k1_cumul[c])
                else:
                    estimate_tiles = tiles_stage[good_ids, c]
                    k1_fixed = 0.0
                inc = estimate_k1_once(
                    estimate_tiles, positions[good_ids],
                    k1_bounds=k1_bounds, order=order,
                    boundary_frac=boundary_frac, boundary_px=boundary_px,
                    local_search=local_search, k1_fixed=k1_fixed,
                    gamma=gamma, skip_diagonal=skip_diagonal)[0]
            k1_cumul[c] += inc
            k1color.append(inc)
            if verbose:
                print(f"  channel {c},  k1_inc: {inc:.6f},  cumul: {k1_cumul[c]:.6f}")
        return k1color

    def _correct_stage():
        # fresh re-correction from pristine tiles each iteration; skip
        # near-zero k1 (spline resampling at identity is not bit-exact), so
        # this stays a guarded loop rather than correct_tiles_once
        if correction_mode == "incremental":
            stage = tiles_stage.copy()
            for c, inc in enumerate(k1color):
                if abs(inc) > 1e-12:
                    stage[:, c] = undistort_tiles(
                        stage[:, c], [inc], order=order)
            return stage
        stage = tiles_original.copy()
        for c in range(ncolors):
            if abs(k1_cumul[c]) > 1e-12:
                stage[:, c] = undistort_tiles(
                    stage[:, c], [k1_cumul[c]], order=order)
        return stage

    tiles_stage = tiles_original.copy()

    small_streak = 0
    for _ in iterator:
        if update_order == "positions_first":
            positions = _stitch_once(positions)
            k1color = _estimate_k1_once()
            tiles_stage = _correct_stage()
        else:
            k1color = _estimate_k1_once()
            tiles_stage = _correct_stage()
            positions = _stitch_once(positions)
        k1_history.append(k1color)
        pos_history.append(positions.copy())
        if k1_tol > 0 and ncolors > 0:
            if all(abs(inc) < k1_tol for inc in k1color):
                small_streak += 1
                if small_streak >= 2:
                    break
            else:
                small_streak = 0

    if correction_mode == "incremental":
        tiles_corrected = tiles_stage
    else:
        tiles_corrected = tiles_original.copy()
        for c in range(ncolors):
            if abs(k1_cumul[c]) > 1e-12:
                tiles_corrected[:, c] = undistort_tiles(
                    tiles_corrected[:, c], [k1_cumul[c]], order=order)

    if was_3d:
        tiles_corrected = tiles_corrected[:, 0, :, :]

    return positions, np.array(k1_history), tiles_corrected, pos_history


# Utilities

def sharpen_tiles(tiles: np.ndarray) -> np.ndarray:
    """Normalize and unsharp-mask each tile before registration."""
    out = np.copy(tiles).astype(np.float32)
    multichannel = out.ndim == 4

    if multichannel:
        for ch in range(out.shape[1]):
            mx = out[:, ch].max()
            if mx > 0:
                out[:, ch] /= mx
            out[:, ch] = np.array([
                unsharp_mask(t, radius=1, amount=1) for t in out[:, ch]
            ])
    else:
        mx = out.max()
        if mx > 0:
            out /= mx
        out = np.array([
            unsharp_mask(t, radius=1, amount=1) for t in out
        ])

    return out.astype(np.float32)


def assemble(
    tiles: np.ndarray,
    positions: np.ndarray,
    border: int = 64,
) -> np.ndarray:
    """Paste corrected tiles into a non-blended local mosaic canvas."""
    h, w = tiles.shape[-2:]
    pos = positions.copy()

    y_c = pos[:, 0] - pos[:, 0].min()
    x_c = pos[:, 1] - pos[:, 1].min()

    H = int((y_c + h).max() + h * 1.2)
    W = int((x_c + w).max() + w * 1.2)
    oy, ox = h // 4, w // 4

    single = tiles.ndim == 3
    if single:
        canvas = np.zeros((H, W), dtype=tiles.dtype)
        for idx in range(len(tiles)):
            yt = int(round(y_c[idx] + oy))
            xt = int(round(x_c[idx] + ox))
            canvas[yt + border:yt + h - border,
                   xt + border:xt + w - border] = \
                tiles[idx, border:-border, border:-border]
    else:
        ch = tiles.shape[1]
        canvas = np.zeros((ch, H, W), dtype=tiles.dtype)
        for idx in range(tiles.shape[0]):
            yt = int(round(y_c[idx] + oy))
            xt = int(round(x_c[idx] + ox))
            canvas[:, yt + border:yt + h - border,
                   xt + border:xt + w - border] = \
                tiles[idx, :, border:-border, border:-border]

    return canvas


StitcherResult = namedtuple(
    "StitcherResult",
    ["positions", "k1_history", "tiles", "mosaic", "positions_pre",
     "position_history", "iterations_used"],
    defaults=[None, None],
)


class DistortCorrectStitcher:
    """Iteratively alternate position refinement and distortion updates."""

    def __init__(
        self,
        k1_bounds: tuple[float, float] = (-0.005, 0.005),
        gamma: float = 0.5,
        n_iterations: int = 25,
        k1_tol: float = 0.0,
        interpolation_order: int = 3,
        pre_stitch_gamma_schedule: Sequence[float] | None = None,
        ncc_threshold: float = 0.0,
        sharpen: bool = False,
        boundary_frac: float = 0.25,
        boundary_px: int | None = None,
        position_damping: float = 0.2,
        local_search: int = 5,
        update_order: str = "positions_first",
        correction_mode: str = "pristine_cumulative",
        border: int = 64,
        skip_diagonal: bool = True,
        oracle_k1: float | Sequence[float] | None = None,
    ) -> None:
        self.k1_bounds = k1_bounds
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.k1_tol = k1_tol
        self.order = interpolation_order
        self.pre_stitch_gamma_schedule = pre_stitch_gamma_schedule
        self.ncc_threshold = ncc_threshold
        self.sharpen = sharpen
        self.boundary_frac = boundary_frac
        self.boundary_px = boundary_px
        self.position_damping = position_damping
        self.local_search = local_search
        self.update_order = update_order
        self.correction_mode = correction_mode
        self.border = border
        self.skip_diagonal = skip_diagonal
        self.oracle_k1 = oracle_k1

    def pre_stitch(
        self,
        tiles: np.ndarray,
        positions: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """Run the configured position-only warm-start schedule."""
        return pr_stitching(
            tiles, positions,
            gamma_schedule=self.pre_stitch_gamma_schedule,
            ncc_threshold=self.ncc_threshold,
            sharpen=self.sharpen,
            skip_diagonal=self.skip_diagonal,
            verbose=verbose,
        )

    def joint_optimize(
        self,
        tiles: np.ndarray,
        positions: np.ndarray,
        outlier_tiles: Sequence[int] | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
        """Run the configured alternating position/distortion updates."""
        return compute_k1_recursive_colors(
            tiles, positions,
            outlier_tiles=outlier_tiles,
            k1_bounds=self.k1_bounds,
            gamma=self.gamma,
            n_iterations=self.n_iterations,
            k1_tol=self.k1_tol,
            order=self.order,
            ncc_threshold=self.ncc_threshold,
            sharpen=self.sharpen,
            boundary_frac=self.boundary_frac,
            boundary_px=self.boundary_px,
            position_damping=self.position_damping,
            local_search=self.local_search,
            update_order=self.update_order,
            correction_mode=self.correction_mode,
            skip_diagonal=self.skip_diagonal,
            oracle_k1=self.oracle_k1,
            verbose=verbose,
        )

    def run(
        self,
        tiles: np.ndarray,
        positions_ini: np.ndarray,
        outlier_tiles: Sequence[int] | None = None,
        do_sharpen: bool = True,
        verbose: bool = False,
    ) -> StitcherResult:
        """Execute preprocessing, warm start, iterative optimization, and assembly."""
        tiles_proc = tiles.copy()

        if do_sharpen:
            tiles_proc = sharpen_tiles(tiles_proc)

        positions_pre = self.pre_stitch(
            tiles_proc, positions_ini, verbose=verbose)

        (positions, k1_history, tiles_corrected,
         position_history) = self.joint_optimize(
            tiles_proc, positions_pre,
            outlier_tiles=outlier_tiles, verbose=verbose)

        mosaic = assemble(tiles_corrected, positions, border=self.border)

        return StitcherResult(
            positions=positions,
            k1_history=k1_history,
            tiles=tiles_corrected,
            mosaic=mosaic,
            positions_pre=positions_pre,
            position_history=position_history,
            iterations_used=len(k1_history),
        )


# CUDA acceleration for the consolidated stitching implementation.

from contextlib import contextmanager


@contextmanager
def _gpu_backend(device=None):
    global undistort_tiles, _mi_align_cpu
    import torch

    dev = _require_cuda(device)
    old_warp = undistort_tiles
    old_mi = _mi_align_cpu
    undistort_tiles = undistort_tiles_gpu
    _mi_align_cpu = _mi_align_gpu
    try:
        with torch.cuda.device(dev):
            yield
    finally:
        undistort_tiles = old_warp
        _mi_align_cpu = old_mi


def _require_cuda(device=None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU pipeline requested, but CUDA is unavailable. Install a "
            "CUDA-enabled PyTorch build and a compatible NVIDIA driver."
        )
    return torch.device(device or "cuda")


def _gpu_grid(h, w, k1, k2, k3, device):
    # grid_sample grid matching the CPU radial model
    import torch

    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    xc = (x - w / 2.0) / (w / 2.0)
    yc = (y - h / 2.0) / (h / 2.0)
    r2 = xc.square() + yc.square()
    radial = 1.0 + k1 * r2 + k2 * r2.square() + k3 * r2 * r2.square()
    xd = xc * radial * (w / 2.0) + w / 2.0
    yd = yc * radial * (h / 2.0) + h / 2.0

    # grid_sample expects (x, y) normalized to [-1, 1]
    gx = 2.0 * xd / max(w - 1, 1) - 1.0
    gy = 2.0 * yd / max(h - 1, 1) - 1.0
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


def undistort_image_gpu(
    image: np.ndarray,
    k1: float,
    k2: float = 0.0,
    k3: float = 0.0,
    order: int = 3,
    mode: str = "nearest",
    device: str | None = None,
) -> np.ndarray:
    """Correct one image with PyTorch grid sampling on CUDA."""
    import torch
    import torch.nn.functional as F

    dev = _require_cuda(device)
    original_dtype = image.dtype
    arr = np.asarray(image)
    if arr.ndim == 2:
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=dev)[None, None]
        channels = False
    elif arr.ndim == 3:
        tensor = torch.as_tensor(arr, dtype=torch.float32, device=dev)[None]
        channels = True
    else:
        raise ValueError("image must have shape (y, x) or (channel, y, x)")

    h, w = arr.shape[-2:]
    grid = _gpu_grid(h, w, k1, k2, k3, dev)
    interpolation = "bilinear" if order <= 1 else "bicubic"
    padding = "zeros" if mode in {"constant", "grid-constant"} else "border"
    out = F.grid_sample(
        tensor, grid, mode=interpolation,
        padding_mode=padding, align_corners=True,
    )
    result = out[0] if channels else out[0, 0]
    result = result.detach().cpu().numpy()
    return _saturate(result, original_dtype)


def undistort_tiles_gpu(
    tiles: np.ndarray,
    k1s: Sequence[float],
    order: int = 3,
    mode: str = "nearest",
    device: str | None = None,
) -> np.ndarray:
    """Batch-correct tiles with the CUDA warp backend."""
    import torch
    import torch.nn.functional as F

    dev = _require_cuda(device)
    corrected = np.asarray(tiles).copy()
    if corrected.ndim not in (3, 4):
        raise ValueError("tiles must have shape (t, y, x) or (t, c, y, x)")

    channels_first = corrected.ndim == 4
    for k1 in k1s:
        original_dtype = corrected.dtype
        if channels_first:
            batch = torch.as_tensor(
                corrected, dtype=torch.float32, device=dev
            )
        else:
            batch = torch.as_tensor(
                corrected, dtype=torch.float32, device=dev
            ).unsqueeze(1)
        h, w = corrected.shape[-2:]
        grid = _gpu_grid(h, w, float(k1), 0.0, 0.0, dev)
        grid = grid.expand(batch.shape[0], -1, -1, -1)
        interpolation = "bilinear" if order <= 1 else "bicubic"
        padding = "zeros" if mode in {"constant", "grid-constant"} else "border"
        batch = F.grid_sample(
            batch, grid, mode=interpolation,
            padding_mode=padding, align_corners=True,
        )
        corrected = _saturate(
            batch.detach().cpu().numpy(), original_dtype
        )
        if not channels_first:
            corrected = corrected[:, 0]
    return corrected


def _mi_align_gpu(ref, tar):
    # author's CUDA MI alignment after CPU k-means quantization
    _require_cuda()
    import mutualinformation_single as mi

    q = 16
    ref8 = ref.copy().astype(np.float32)
    tar8 = tar.copy().astype(np.float32)
    ref_q = image2cat_kmeans(ref8, q)
    tar_q = image2cat_kmeans(tar8, q)
    mask = np.ones(ref.shape, dtype=np.float32)
    param, _ = mi.align_translation(
        ref_q, tar_q, mask, mask, q, q,
        overlap=0.5,
        enable_partial_overlap=True,
        normalize_mi=False,
        on_gpu=True,
        save_maps=False,
    )
    return np.asarray([param[1], param[2]], dtype=float)


class DistortCorrectStitcherGPU(DistortCorrectStitcher):
    """CUDA-backed joint stitcher using the shared CPU orchestration."""

    def __init__(
        self,
        *args: Any,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.device = str(_require_cuda(device))
        super().__init__(*args, **kwargs)

    def run(
        self,
        tiles: np.ndarray,
        positions_ini: np.ndarray,
        outlier_tiles: Sequence[int] | None = None,
        do_sharpen: bool = True,
        verbose: bool = False,
    ) -> StitcherResult:
        """Run the joint pipeline with temporary CUDA primitive patches."""
        # Shared graph/scoring code calls the patched GPU primitives.
        with _gpu_backend(self.device):
            return super().run(
                tiles, positions_ini,
                outlier_tiles=outlier_tiles,
                do_sharpen=do_sharpen,
                verbose=verbose,
            )
