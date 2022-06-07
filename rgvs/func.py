from __future__ import annotations

__all__ = ['sbr', 'minblur', 'boxblur', 'median', 'blur', 'min_filter', 'median_clip', 'median_diff']


from typing import Sequence
from enum import Enum
from functools import partial
from vsutil import disallow_variable_format, disallow_variable_resolution, Range as CRange

import vapoursynth as vs

from .util import normalise_planes, get_neutral_value, normalise_seq

core = vs.core

wmean_matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]
mean_matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1]


class MedianMode(str, Enum):
    SQUARE = 'hv'
    VERTICAL = 'v'
    HORIZONTAL = 'h'


def boxblur(
    clip: vs.VideoNode, weights: Sequence[float], planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    if len(weights) != 9:
        raise ValueError('boxblur: weights has to be an array of length 9!')

    try:
        aka_expr = core.akarin.Expr
    except AttributeError:
        pass
    else:
        weights_string = [
            x.format(w=w) for x, w in zip([
                'x[-1,-1] {w} *', 'x[0,-1] {w} *', 'x[1,-1] {w} *',
                'x[-1,0] {w} *', 'x {w} *', 'x[1,0] {w} *',
                'x[-1,1] {w} *', 'x[0,1] {w} *', 'x[1,1] {w} *'
            ], weights)
        ].join(' ')

        add_string = '+ ' * 8

        expr_string = f'{weights_string} {add_string} {sum(map(float, weights))} /'

        return aka_expr(clip, [
            expr_string if i in normalise_planes(planes) else ''
            for i in range(clip.format.num_planes)
        ])
    return clip.std.Convolution(weights, planes=planes)


@disallow_variable_format
@disallow_variable_resolution
def minblur(clip: vs.VideoNode, radius: int = 1, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    """
    MinBlur   by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert clip.format

    planes = normalise_planes(clip, planes)
    pboxblur = partial(boxblur, planes=planes)

    median = clip.std.Median(planes) if radius in {0, 1} else clip.ctmf.CTMF(radius, planes=planes)

    if radius == 0:
        weighted = sbr(clip, planes=planes)
    elif radius == 1:
        weighted = pboxblur(clip, wmean_matrix)
    elif radius == 2:
        weighted = pboxblur(pboxblur(clip, wmean_matrix), mean_matrix)
    else:
        weighted = pboxblur(pboxblur(pboxblur(clip, wmean_matrix), mean_matrix), mean_matrix)

    return core.std.Expr([clip, weighted, median], [
        'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
        if i in planes else '' for i in range(clip.format.num_planes)
    ])


@disallow_variable_format
@disallow_variable_resolution
def sbr(
    clip: vs.VideoNode,
    radius: int = 1, mode: MedianMode = MedianMode.SQUARE,
    planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)
    neutral = normalise_seq(
        [get_neutral_value(clip), get_neutral_value(clip, True)], clip.format.num_planes
    )

    if mode == MedianMode.SQUARE:
        matrix2 = [1, 3, 4, 3, 1]
        matrix3 = [1, 4, 8, 10, 8, 4, 1]
    elif mode in {MedianMode.HORIZONTAL, MedianMode.VERTICAL}:
        matrix2 = [1, 6, 15, 20, 15, 6, 1]
        matrix3 = [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    else:
        raise ValueError('sbr: invalid mode specified!')

    if radius == 1:
        matrix = [1, 2, 1]
    elif radius == 2:
        matrix = matrix2
    elif radius == 3:
        matrix = matrix3
    else:
        raise ValueError('sbr: invalid radius')

    pboxblur = partial(core.std.Convolution, matrix=matrix, planes=planes, mode=mode)

    weighted = pboxblur(clip)

    diff = clip.std.MakeDiff(weighted, planes)

    diff_weighted = pboxblur(diff)

    diff = core.std.Expr([diff, diff_weighted], [
        f'x y - x {mid} - * 0 < {mid} x y - abs x {mid} - abs < x y - {mid} + x ? ?'
        if i in planes else '' for i, mid in enumerate(neutral, 0)
    ])

    return clip.std.MakeDiff(diff, planes)


def median_diff(
    clip: vs.VideoNode,
    diffa: vs.VideoNode, diffb: vs.VideoNode,
    planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)
    neutral = normalise_seq(
        [get_neutral_value(clip), get_neutral_value(clip, True)], clip.format.num_planes
    )

    return core.std.Expr([clip, diffa, diffb], [
        f'x 0 y {mid} y z - min max y {mid} y z - max min -'
        if i in planes else '' for i, mid in enumerate(neutral, 0)
    ])
