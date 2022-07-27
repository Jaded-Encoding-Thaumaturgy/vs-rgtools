from __future__ import annotations

from functools import partial
from math import ceil, exp, log, pi, sqrt

import vapoursynth as vs
from vsexprtools import expr_func
from vsexprtools.util import PlanesT, aka_expr_available, norm_expr_planes, normalise_planes
from vsutil import (
    depth, disallow_variable_format, disallow_variable_resolution, get_depth, get_neutral_value, join, split
)

from .enum import ConvMode, LimitFilterMode
from .limit import limit_filter
from .util import mean_matrix, wmean_matrix

__all__ = [
    'blur', 'min_blur', 'box_blur', 'gauss_blur', 'gauss_fmtc_blur', 'sbr'
]

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def blur(
    clip: vs.VideoNode, radius: int = 1, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    if mode == ConvMode.SQUARE:
        matrix2 = [1, 3, 4, 3, 1]
        matrix3 = [1, 4, 8, 10, 8, 4, 1]
    elif mode in {ConvMode.HORIZONTAL, ConvMode.VERTICAL}:
        matrix2 = [1, 6, 15, 20, 15, 6, 1]
        matrix3 = [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    else:
        raise ValueError('blur: invalid mode specified!')

    if radius == 1:
        matrix = [1, 2, 1]
    elif radius == 2:
        matrix = matrix2
    elif radius == 3:
        matrix = matrix3
    else:
        raise ValueError('blur: invalid radius')

    return clip.std.Convolution(matrix, planes=planes, mode=mode)


@disallow_variable_format
@disallow_variable_resolution
def box_blur(clip: vs.VideoNode, radius: int = 1, passes: int = 1, planes: PlanesT = None) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    if radius > 12:
        blurred = clip.std.BoxBlur(planes, radius, passes, radius, passes)
    else:
        matrix_size = radius * 2 | 1
        blurred = clip
        for _ in range(passes):
            blurred = blurred.std.Convolution([1] * matrix_size, mode=ConvMode.SQUARE)

    return blurred


def _norm_gauss_sigma(clip: vs.VideoNode, sigma: float | None, sharp: float | None, func_name: str) -> float:
    if sigma is None:
        if sharp is None:
            raise ValueError(f"{func_name}: sigma and sharp can't be both None!")
        sigma = sqrt(1.0 / (2.0 * (sharp / 10.0) * log(2)))
    elif sharp is not None:
        raise ValueError(f"{func_name}: sigma and sharp can't both be float!")

    if sigma >= min(clip.width, clip.height):
        raise ValueError(f"{func_name}: sigma can't be bigger or equal than the smaller size of the clip!")

    return sigma


@disallow_variable_format
@disallow_variable_resolution
def gauss_blur(
    clip: vs.VideoNode,
    sigma: float | None = 0.5, sharp: float | None = None,
    mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    sigma = _norm_gauss_sigma(clip, sigma, sharp, 'gauss_blur')

    if sigma <= 0.333:
        return clip

    if sigma <= 4.333:
        taps = ceil(sigma * 6 + 1)

        if not taps % 2:
            taps += 1

        half_pisqrt = 1.0 / sqrt(2.0 * pi) * sigma
        doub_qsigma = 2 * sigma * sigma

        high, *kernel = [
            half_pisqrt * exp(-(x * x) / doub_qsigma)
            for x in range(taps // 2)
        ]

        kernel = [x * 1023 / high for x in kernel]
        kernel = [*kernel[::-1], 1023, *kernel]

        return clip.std.Convolution(kernel, mode=mode, planes=planes)

    return gauss_fmtc_blur(clip, sigma, sharp, True, mode, planes)


@disallow_variable_format
@disallow_variable_resolution
def gauss_fmtc_blur(
    clip: vs.VideoNode,
    sigma: float | None = 0.5, sharp: float | None = None,
    strict: bool = True, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    if strict:
        sigma = _norm_gauss_sigma(clip, sigma, sharp, 'gauss_fmtc_blur')
        wdown, hdown = [
            round((max(round(size / sigma), 2) if char in mode else size) / 2) * 2
            for size, char in [(clip.width, 'h'), (clip.height, 'v')]
        ]

        def _fmtc_blur(clip: vs.VideoNode) -> vs.VideoNode:
            down = clip.resize.Bilinear(wdown, hdown)
            return down.fmtc.resample(clip.width, clip.height, kernel='gauss', a1=9)
    else:
        if sigma is None or sigma < 1.0 or sigma > 100.0:
            raise ValueError('gauss_fmtc_blur: With strict=True sigma has to be > 1 and < 100!')
        elif sharp is None:
            sharp = 100
        elif sharp < 1.0 or sharp > 100.0:
            raise ValueError('gauss_fmtc_blur: With strict=True sharp has to be > 1 and < 100!')

        def _fmtc_blur(clip: vs.VideoNode) -> vs.VideoNode:
            down = clip.fmtc.resample(
                clip.width * 2, clip.height * 2, kernel='gauss', a1=sharp
            )
            return down.fmtc.resample(clip.width, clip.height, kernel='gauss', a1=sigma)

    if not {*range(clip.format.num_planes)} - {*planes}:
        blurred = _fmtc_blur(clip)
    else:
        blurred = join([
            _fmtc_blur(p) if i in planes else p for i, p in enumerate(split(clip))
        ])

    if (bits := get_depth(clip)) != get_depth(blurred):
        return depth(blurred, bits)

    return blurred


@disallow_variable_format
@disallow_variable_resolution
def min_blur(clip: vs.VideoNode, radius: int = 1, planes: PlanesT = None) -> vs.VideoNode:
    """
    MinBlur by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert clip.format

    planes = normalise_planes(clip, planes)
    pconv = partial(core.std.Convolution, planes=planes)

    median = clip.std.Median(planes) if radius in {0, 1} else clip.ctmf.CTMF(radius, None, planes)

    if radius == 0:
        weighted = sbr(clip, planes=planes)
    elif radius == 1:
        weighted = pconv(clip, wmean_matrix)
    elif radius == 2:
        weighted = pconv(pconv(clip, wmean_matrix), mean_matrix)
    else:
        weighted = pconv(pconv(pconv(clip, wmean_matrix), mean_matrix), mean_matrix)

    return limit_filter(weighted, clip, median, LimitFilterMode.DIFF_MIN, planes)


@disallow_variable_format
@disallow_variable_resolution
def sbr(
    clip: vs.VideoNode,
    radius: int = 1, mode: ConvMode = ConvMode.SQUARE,
    planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

    blur_func = partial(blur, radius=radius, mode=mode, planes=planes)

    weighted = blur_func(clip)

    diff = clip.std.MakeDiff(weighted, planes)

    diff_weighted = blur_func(diff)

    clips = [diff, diff_weighted]

    if aka_expr_available:
        clips.append(clip)
        expr = 'x y - A! x {mid} - XD! z A@ XD@ * 0 < {mid} A@ abs XD@ abs < A@ {mid} + x ? ? {mid} - -'
    else:
        expr = 'x y - x {mid} - * 0 < {mid} x y - abs x {mid} - abs < x y - {mid} + x ? ?'

    normalized_expr = norm_expr_planes(clip, expr, planes, mid=neutral)

    diff = expr_func(clips, normalized_expr)

    if aka_expr_available:
        return diff

    return clip.std.MakeDiff(diff, planes)
