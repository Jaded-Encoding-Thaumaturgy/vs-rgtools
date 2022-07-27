from __future__ import annotations

from typing import Sequence

import vapoursynth as vs
from vsexprtools import expr_func
from vsexprtools.util import aka_expr_available, PlanesT, VSFunction, norm_expr_planes, normalise_planes
from vsutil import disallow_variable_format, disallow_variable_resolution, fallback, get_neutral_value

from vsrgtools.enum import LimitFilterMode

__all__ = [
    'limit_filter', 'minimum_diff', 'median_diff', 'median_clips', 'flux_smooth'
]

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def limit_filter(
    src: vs.VideoNode, flt1: vs.VideoNode, flt2: vs.VideoNode,
    mode: LimitFilterMode = LimitFilterMode.SIMPLE_MIN,
    planes: PlanesT = None
) -> vs.VideoNode:
    if mode in {LimitFilterMode.SIMPLE_MIN, LimitFilterMode.SIMPLE_MAX}:
        expr = 'x z - abs x y - abs {op} z y ?'
    else:
        if aka_expr_available:
            expr = 'x y - A! x z - B! A@ B@ xor x A@ abs B@ abs {op} y z ? ?'
        else:
            expr = 'x y - x z - xor x x y - abs x z - abs {op} y z ? ?'

    return expr_func(
        [src, flt1, flt2], norm_expr_planes(src, expr, planes, op=mode.op)
    )


@disallow_variable_format
@disallow_variable_resolution
def minimum_diff(
    clip: vs.VideoNode,
    clip_func: VSFunction, diff_func: VSFunction | None = None,
    diff: bool | None = None, planes: PlanesT = None
) -> vs.VideoNode:
    planes = normalise_planes(clip, planes)

    diff = fallback(diff, diff_func is None)

    diff_func = fallback(diff_func, clip_func)

    if not diff:
        return median_clips([clip, clip_func(clip), diff_func(clip)], planes)

    filtered = clip_func(clip)

    diffa = core.std.MakeDiff(clip, filtered, planes)

    diffb = diff_func(diffa)

    return median_diff(clip, diffa, diffb, planes)


@disallow_variable_format
@disallow_variable_resolution
def median_diff(clip: vs.VideoNode, diffa: vs.VideoNode, diffb: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)
    neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

    return core.std.Expr(
        [clip, diffa, diffb],
        norm_expr_planes(clip, 'x 0 y {mid} y z - min max y {mid} y z - max min -', planes, mid=neutral)
    )


@disallow_variable_format
@disallow_variable_resolution
def median_clips(clips: Sequence[vs.VideoNode], planes: PlanesT = None) -> vs.VideoNode:
    return core.std.Expr(
        list(clips), norm_expr_planes(clips[0], 'x y z min max y z max min', planes)
    )


@disallow_variable_format
@disallow_variable_resolution
def flux_smooth(
    clip: vs.VideoNode, radius: int = 2, threshold: int = 7, scenechange: int = 24, planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    if radius < 1 or radius > 7:
        raise ValueError('flux_smooth: radius must be between 1 and 7 (inclusive)!')

    planes = normalise_planes(clip, planes, False)

    threshold = threshold << clip.format.bits_per_sample - 8

    cthreshold = threshold if (1 in planes or 2 in planes) else 0

    median = clip.tmedian.TemporalMedian(radius, planes)
    average = clip.focus2.TemporalSoften2(
        radius, threshold, cthreshold, scenechange
    )

    return limit_filter(clip, average, median, LimitFilterMode.DIFF_MIN, planes)
