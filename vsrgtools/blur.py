from __future__ import annotations

__all__ = [
    'blur', 'min_blur', 'box_blur', 'gauss_blur', 'sbr'
]


from functools import partial
from math import ceil, exp, log, pi, sqrt
from typing import Sequence

import vapoursynth as vs
from vsutil import disallow_variable_format, disallow_variable_resolution, get_neutral_value, join, split

from .enum import ConvMode
from .util import PlanesT, mean_matrix, norm_expr_planes, normalise_planes, normalise_seq, wmean_matrix

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
def box_blur(
    clip: vs.VideoNode, weights: Sequence[float], planes: PlanesT = None, expr: bool | None = False
) -> vs.VideoNode:
    assert clip.format

    if len(weights) != 9:
        raise ValueError('box_blur: weights has to be an array of length 9!')

    try:
        aka_expr = core.akarin.Expr
        expr = True if expr is None else expr
    except AttributeError:
        expr = False

    if expr:
        weights_string = ' '.join([
            x.format(w=w) for x, w in zip([
                'x[-1,-1] {w} *', 'x[0,-1] {w} *', 'x[1,-1] {w} *',
                'x[-1,0] {w} *', 'x {w} *', 'x[1,0] {w} *',
                'x[-1,1] {w} *', 'x[0,1] {w} *', 'x[1,1] {w} *'
            ], weights)
        ])

        add_string = '+ ' * 8

        expr_string = f'{weights_string} {add_string} {sum(map(float, weights))} /'

        return aka_expr(clip, [
            expr_string if i in normalise_planes(clip, planes) else ''
            for i in range(clip.format.num_planes)
        ])

    return clip.std.Convolution(weights, planes=planes)


@disallow_variable_format
@disallow_variable_resolution
def gauss_blur(
    clip: vs.VideoNode,
    sigma: float | None = 0.5, sharp: float | None = None,
    mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    if sigma is None:
        if sharp is None:
            raise ValueError("gauss_blur: sigma and sharp can't be both None!")
        sigma = sqrt(1.0 / (2.0 * (sharp / 10.0) * log(2)))
    elif sharp is not None:
        raise ValueError("gauss_blur: sigma and sharp can't both be float!")

    if sigma >= min(clip.width, clip.height):
        raise ValueError("gauss_blur: sigma can't be bigger or equal than the smaller size of the clip!")

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

    wdown, hdown = [
        round((max(round(size / sigma), 2) if char in mode else size) / 2) * 2
        for size, char in [(clip.width, 'h'), (clip.height, 'v')]
    ]

    def _fmtc_blur(clip: vs.VideoNode) -> vs.VideoNode:
        down = clip.resize.Bilinear(wdown, hdown)
        return down.fmtc.resample(clip.width, clip.height, kernel='gauss', a1=9)

    different_planes = len(set(range(clip.format.num_planes)) - set(planes))

    if not different_planes:
        return _fmtc_blur(clip)

    return join([_fmtc_blur(p) if i in planes else p for i, p in enumerate(split(clip))])


@disallow_variable_format
@disallow_variable_resolution
def min_blur(clip: vs.VideoNode, radius: int = 1, planes: PlanesT = None) -> vs.VideoNode:
    """
    MinBlur by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert clip.format

    planes = normalise_planes(clip, planes)
    pbox_blur = partial(box_blur, planes=planes)

    median = clip.std.Median(planes) if radius in {0, 1} else clip.ctmf.CTMF(radius, None, planes)

    if radius == 0:
        weighted = sbr(clip, planes=planes)
    elif radius == 1:
        weighted = pbox_blur(clip, wmean_matrix)
    elif radius == 2:
        weighted = pbox_blur(pbox_blur(clip, wmean_matrix), mean_matrix)
    else:
        weighted = pbox_blur(pbox_blur(pbox_blur(clip, wmean_matrix), mean_matrix), mean_matrix)

    return core.std.Expr(
        [clip, weighted, median],
        norm_expr_planes(clip, 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?', planes)
    )


@disallow_variable_format
@disallow_variable_resolution
def sbr(
    clip: vs.VideoNode,
    radius: int = 1, mode: ConvMode = ConvMode.SQUARE,
    planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    neutral = normalise_seq(
        [get_neutral_value(clip), get_neutral_value(clip, True)], clip.format.num_planes
    )

    blur_func = partial(blur, radius=radius, mode=mode, planes=planes)

    weighted = blur_func(clip)

    diff = clip.std.MakeDiff(weighted, planes)

    diff_weighted = blur_func(diff)

    diff = core.std.Expr(
        [diff, diff_weighted], norm_expr_planes(
            clip, 'x y - x {mid} - * 0 < {mid} x y - abs x {mid} - abs < x y - {mid} + x ? ?', planes, mid=neutral
        )
    )

    return clip.std.MakeDiff(diff, planes)
