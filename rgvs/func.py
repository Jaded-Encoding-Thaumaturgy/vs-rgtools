from __future__ import annotations

__all__ = ['sbr', 'minblur']


from typing import Sequence

import vapoursynth as vs

core = vs.core


def minblur(clip: vs.VideoNode, radius: int = 1, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    """
    MinBlur   by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    if not clip.format:
        raise ValueError('minblur: Variable format isn\'t supported')
    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    wmean_mat = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mean_mat = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if radius == 0:
        weighted = sbr(clip, planes=planes)
        median = core.std.Median(clip, planes)
    elif radius == 1:
        weighted = core.std.Convolution(clip, wmean_mat, planes=planes)
        median = core.std.Median(clip, planes)
    elif radius == 2:
        weighted = core.std.Convolution(clip, wmean_mat, planes=planes).std.Convolution(mean_mat, planes=planes)
        median = core.ctmf.CTMF(clip, radius, planes=planes)
    else:
        weighted = core.std.Convolution(clip, wmean_mat, planes=planes).std.Convolution(mean_mat, planes=planes)
        weighted = weighted.std.Convolution(mean_mat, planes=planes)
        median = core.ctmf.CTMF(clip, radius, planes=planes)

    return core.std.Expr(
        [clip, weighted, median],
        ['x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
         if i in planes else '' for i in range(clip.format.num_planes)]
    )


def sbr(clip: vs.VideoNode, radius: int = 1, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    """make a highpass on a blur's difference (well, kind of that)"""
    if not clip.format:
        raise ValueError('minblur: Variable format isn\'t supported')

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    neutral = [
        1 << (clip.format.bits_per_sample - 1)
        if clip.format.sample_type != vs.FLOAT else 0.
    ] * clip.format.num_planes

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if radius == 1:
        weighted = core.std.Convolution(clip, matrix1, planes=planes)
    elif radius == 2:
        weighted = core.std.Convolution(clip, matrix1, planes=planes).std.Convolution(matrix2, planes=planes)
    else:
        weighted = core.std.Convolution(clip, matrix1, planes=planes).std.Convolution(matrix2, planes=planes)
        weighted = weighted.std.Convolution(matrix2, planes=planes)

    diff = core.std.MakeDiff(clip, weighted, planes)

    if radius == 1:
        diff_weighted = core.std.Convolution(diff, matrix1, planes=planes)
    elif radius == 2:
        diff_weighted = core.std.Convolution(diff, matrix1, planes=planes).std.Convolution(matrix2, planes=planes)
    else:
        diff_weighted = core.std.Convolution(diff, matrix1, planes=planes).std.Convolution(matrix2, planes=planes)
        diff_weighted = diff_weighted.std.Convolution(matrix2, planes=planes)

    diff = core.std.Expr(
        [diff, diff_weighted],
        [f'x y - x {neutral[i]} - * 0 < {neutral[i]} x y - abs x {neutral[i]} - abs < x y - {neutral[i]} + x ? ?'
         if i in planes else '' for i in range(clip.format.num_planes)]
    )
    return core.std.MakeDiff(clip, diff, planes)
