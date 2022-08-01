from __future__ import annotations

import vapoursynth as vs
from vsexprtools import (
    EXPR_VARS, ExprOp, PlanesT, VSFunction, aka_expr_available, expr_func, norm_expr_planes, normalise_planes
)
from vsutil import disallow_variable_format, disallow_variable_resolution, fallback, get_neutral_value

from .enum import LimitFilterMode
from .limit import limit_filter

__all__ = [
    'minimum_diff', 'median_diff',
    'median_clips',
    'flux_smooth'
]

core = vs.core


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
        return median_clips(clip, clip_func(clip), diff_func(clip), planes=planes)

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

    if aka_expr_available:
        expr = 'y z - D! x D@ y {mid} clamp D@ {mid} y clamp - -'
    else:
        expr = 'x y z - y min {mid} max y z - {mid} min y max - -'

    return expr_func([clip, diffa, diffb], norm_expr_planes(clip, expr, planes, mid=neutral))


@disallow_variable_format
@disallow_variable_resolution
def median_clips(*clips: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    n_clips = len(clips)

    if n_clips > 26:
        raise ValueError('median_clip: You can pass only up to 26 clips!')

    if n_clips < 3:
        raise ValueError('median_clip: You must pass at least 3 clips!')

    if n_clips == 3:
        return expr_func(clips, 'x y z min max y z max min')

    all_clips = ' '.join(EXPR_VARS[i] for i in range(1, n_clips))

    n_ops = n_clips - 2

    all_clips_min, all_clips_max = [
        all_clips + f' {op}' * n_ops for op in (ExprOp.MIN, ExprOp.MAX)
    ]

    header = ''
    if aka_expr_available:
        header = f'{all_clips_min} YZMIN! {all_clips_max} YZMAX! '
        yzmin, yzmax = 'YZMIN@', 'YZMAX@'

    expr = f'{header} x {yzmin} min x = {yzmin} x {yzmax} max x = {yzmax} x ? ?'

    return expr_func(clips, norm_expr_planes(clips[0], expr, planes))


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

    return limit_filter(average, clip, median, LimitFilterMode.DIFF_MIN, planes)
