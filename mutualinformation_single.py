"""GPU mutual-information alignment adapted from Öfverstedt (2021)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# Adapted from code originally authored by Johan Öfverstedt (2021), MIT License.
# Modified by Hu Cang (2024).

VALUE_TYPE = torch.float32


def compute_entropy(
    C: torch.Tensor,
    N: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute the negative entropy contribution for quantized counts."""
    p = C / N
    return p * torch.log2(torch.clamp(p, min=eps))


def float_compare(A: torch.Tensor, c: int) -> torch.Tensor:
    """Build a soft level-set mask for one quantization value."""
    return torch.clamp(1 - torch.abs(A - c), 0.0)


def fft_of_levelsets(
    A: torch.Tensor,
    Q: int,
    packing: int,
    setup_fn: Callable[[torch.Tensor], Any],
) -> list[tuple[Any, int, int]]:
    """Batch level-set FFTs to limit temporary GPU memory use."""
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

def fft(A: torch.Tensor) -> torch.Tensor:
    """Return the two-dimensional real FFT of a tensor."""
    spectrum = torch.fft.rfft2(A)
    return spectrum


def ifft(Afft: torch.Tensor) -> torch.Tensor:
    """Return the inverse two-dimensional real FFT."""
    res = torch.fft.irfft2(Afft)
    return res


def fftconv(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Multiply Fourier-domain tensors for circular convolution."""
    C = A * B
    return C


def corr_target_setup(A: torch.Tensor) -> torch.Tensor:
    """Prepare a target tensor for Fourier-domain correlation."""
    B = fft(A)
    return B


def corr_template_setup(B: torch.Tensor) -> torch.Tensor:
    """Prepare a template tensor using the conjugate Fourier spectrum."""
    B_FFT = torch.conj(fft(B))
    return B_FFT


def corr_apply(
    A: torch.Tensor,
    B: torch.Tensor,
    sz: Sequence[int],
    do_rounding: bool = True,
) -> torch.Tensor:
    """Apply Fourier correlation and crop it to the valid shape."""
    C = fftconv(A, B)
    C = ifft(C)
    C = C[:sz[0], :sz[1], :sz[2], :sz[3]]
    if do_rounding:
        C = torch.round(C)
    return C

def create_float_tensor(
    shape: Sequence[int],
    on_gpu: bool,
    fill_value: float | None = None,
) -> torch.Tensor:
    """Create a float tensor on CPU or the active CUDA device."""
    if on_gpu:
        # Equivalent to the legacy torch.cuda.FloatTensor constructor, without
        # its deprecation warning on current PyTorch releases.
        res = torch.empty(tuple(shape), dtype=torch.float32, device="cuda")
        if fill_value is not None:
            res.fill_(fill_value)
        return res
    else:
        if fill_value is not None:
            res = np.full((shape[0], shape[1], shape[2], shape[3]), fill_value=fill_value, dtype='float32')
        else:
            res = np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype='float32')
        return torch.tensor(res, dtype=torch.float32)

def to_tensor(A: Any, on_gpu: bool = True) -> torch.Tensor:
    """Convert an array-like input to the four-dimensional tensor layout."""
    if torch.is_tensor(A):
        A_tensor = A.cuda(non_blocking=True) if on_gpu else A
        if A_tensor.ndim == 2:
            A_tensor = torch.reshape(A_tensor, (1, 1, A_tensor.shape[0], A_tensor.shape[1]))
        elif A_tensor.ndim == 3:
            A_tensor = torch.reshape(A_tensor, (1, A_tensor.shape[0], A_tensor.shape[1], A_tensor.shape[2]))
        return A_tensor
    else:
        return to_tensor(torch.tensor(A, dtype=VALUE_TYPE), on_gpu=on_gpu)

def align_translation(
    A: Any,
    B: Any,
    M_A: Any,
    M_B: Any,
    Q_A: int,
    Q_B: int,
    overlap: float = 0.5,
    enable_partial_overlap: bool = True,
    normalize_mi: bool = False,
    on_gpu: bool = True,
    save_maps: bool = False,
) -> tuple[tuple[float, int, int], list[np.ndarray] | None]:
    """Estimate translation by maximizing quantized mutual information."""
    eps = 1e-7
    maps = []

    A_tensor = to_tensor(A, on_gpu=on_gpu)
    B_tensor = to_tensor(B, on_gpu=on_gpu)

    if A_tensor.shape[-1] < 1024:
        packing = min(Q_B, 64)
    elif A_tensor.shape[-1] <= 2048:
        packing = min(Q_B, 8)
    elif A_tensor.shape[-1] <= 4096:
        packing = min(Q_B, 4)
    else:
        packing = min(Q_B, 1)

    # Create all constant masks if not provided
    if M_A is None:
        M_A = create_float_tensor(A_tensor.shape, on_gpu, 1.0)
    else:
        M_A = to_tensor(M_A, on_gpu)
        A_tensor = torch.round(M_A * A_tensor + (1 - M_A) * (Q_A + 1))
    if M_B is None:
        M_B = create_float_tensor(B_tensor.shape, on_gpu, 1.0)
    else:
        M_B = to_tensor(M_B, on_gpu)

    # Pad for overlap
    if enable_partial_overlap:
        partial_overlap_pad_sz = (round(B.shape[-1] * (1.0 - overlap)), round(B.shape[-2] * (1.0 - overlap)))
        A_tensor = F.pad(A_tensor, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0],
                                    partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]), mode='constant', value=Q_A + 1)
        M_A = F.pad(M_A, (partial_overlap_pad_sz[0], partial_overlap_pad_sz[0],
                          partial_overlap_pad_sz[1], partial_overlap_pad_sz[1]), mode='constant', value=0)
    else:
        partial_overlap_pad_sz = (0, 0)

    ext_ashape = A_tensor.shape
    ext_bshape = B_tensor.shape
    b_pad_shape = torch.tensor(A_tensor.shape, dtype=torch.long) - torch.tensor(B_tensor.shape, dtype=torch.long)
    ext_valid_shape = b_pad_shape + 1
    batched_valid_shape = ext_valid_shape + torch.tensor([packing - 1, 0, 0, 0])

    # Precompute FFTs of A and M_A
    M_A_FFT = corr_target_setup(M_A)

    A_ffts = []
    for a in range(Q_A):
        A_ffts.append(corr_target_setup(float_compare(A_tensor, a)))

    del A_tensor
    del M_A

    if normalize_mi:
        H_MARG = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
        H_AB = create_float_tensor(ext_valid_shape, on_gpu, 0.0)
    else:
        MI = create_float_tensor(ext_valid_shape, on_gpu, 0.0)

    # Prepare B (no rotation)
    B_tensor_padded = F.pad(B_tensor, (0, ext_ashape[-1] - ext_bshape[-1],
                                       0, ext_ashape[-2] - ext_bshape[-2],
                                       0, 0, 0, 0), mode='constant', value=Q_B + 1)
    M_B_padded = F.pad(M_B, (0, ext_ashape[-1] - ext_bshape[-1],
                             0, ext_ashape[-2] - ext_bshape[-2],
                             0, 0, 0, 0), mode='constant', value=0)
    B_tensor_padded = torch.round(M_B_padded * B_tensor_padded + (1 - M_B_padded) * (Q_B + 1))

    M_B_FFT = corr_template_setup(M_B_padded)
    N = torch.clamp(corr_apply(M_A_FFT, M_B_FFT, ext_valid_shape), min=eps)

    b_ffts = fft_of_levelsets(B_tensor_padded, Q_B, packing, corr_template_setup)

    for bext in range(len(b_ffts)):
        b_fft = b_ffts[bext]
        E_M = torch.sum(compute_entropy(corr_apply(M_A_FFT, b_fft[0], batched_valid_shape), N, eps), dim=0)
        if normalize_mi:
            H_MARG = torch.sub(H_MARG, E_M)
        else:
            MI = torch.sub(MI, E_M)
        del E_M

        for a in range(Q_A):
            A_fft_cuda = A_ffts[a]

            if bext == 0:
                E_M = compute_entropy(corr_apply(A_fft_cuda, M_B_FFT, ext_valid_shape), N, eps)
                if normalize_mi:
                    H_MARG = torch.sub(H_MARG, E_M)
                else:
                    MI = torch.sub(MI, E_M)
                del E_M
            E_J = torch.sum(compute_entropy(corr_apply(A_fft_cuda, b_fft[0], batched_valid_shape), N, eps), dim=0)
            if normalize_mi:
                H_AB = torch.sub(H_AB, E_J)
            else:
                MI = torch.add(MI, E_J)
            del E_J
            del A_fft_cuda
        del b_fft
        if bext == 0:
            del M_B_FFT

    del B_tensor_padded

    if normalize_mi:
        MI = torch.clamp((H_MARG / (H_AB + eps) - 1), 0.0, 1.0)

    if save_maps:
        maps.append(MI.cpu().numpy())

    (max_n, _) = torch.max(torch.reshape(N, (-1,)), 0)
    N_filt = torch.lt(N, overlap * max_n)
    MI[N_filt] = 0.0
    del N_filt, N

    MI_vec = torch.reshape(MI, (-1,))
    (val, ind) = torch.max(MI_vec, -1)

    sz_x = int(ext_valid_shape[3].cpu().numpy())
    y = ind // sz_x
    x = ind % sz_x

    # Adjust translations to account for padding
    translation_y = -(y - partial_overlap_pad_sz[1])
    translation_x = -(x - partial_overlap_pad_sz[0])

    # Convert to scalar values using .item()
    val = val.item()
    translation_y = translation_y.item()
    translation_x = translation_x.item()

    result = (val, translation_y, translation_x)

    if save_maps:
        return result, maps
    else:
        return result, None

from sklearn.cluster import MiniBatchKMeans


def image2cat_kmeans(
    I: np.ndarray,
    k: int,
    batch_size: int = 100,
    max_iter: int = 1000,
    random_seed: int = 1000,
) -> np.ndarray:
    """Quantize a grayscale image into integer MiniBatchKMeans labels."""
    total_shape = I.shape
    spatial_shape = total_shape
    channels = 1
    if k == 1:
        return np.zeros(spatial_shape, dtype='int')
    I_lin = I.reshape(-1, channels)
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter=max_iter, batch_size=batch_size, random_state=random_seed).fit(I_lin)
    I_res = kmeans.labels_
    return I_res.reshape(spatial_shape)
