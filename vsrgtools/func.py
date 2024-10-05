from __future__ import annotations

import warnings
from typing import Iterable

from vsexprtools import complexpr_available, norm_expr
from vstools import (
    CustomIndexError, PlanesT, VSFunction, check_variable, core, fallback, get_neutral_values, normalize_planes, vs
)

from .enum import LimitFilterMode
from .freqs import MeanMode
from .limit import limit_filter

__all__ = [
    'minimum_diff', 'median_diff',
    'median_clips',
    'flux_smooth'
]


def minimum_diff(
    clip: vs.VideoNode,
    clip_func: VSFunction, diff_func: VSFunction | None = None,
    diff: bool | None = None, planes: PlanesT = None
) -> vs.VideoNode:
    planes = normalize_planes(clip, planes)

    diff = fallback(diff, diff_func is None)

    diff_func = fallback(diff_func, clip_func)

    if not diff:
        return MeanMode.MEDIAN(clip, clip_func(clip), diff_func(clip), planes=planes)

    filtered = clip_func(clip)

    diffa = core.std.MakeDiff(clip, filtered, planes)

    diffb = diff_func(diffa)

    return median_diff(clip, diffa, diffb, planes)


def median_diff(clip: vs.VideoNode, diffa: vs.VideoNode, diffb: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    assert check_variable(clip, median_diff)

    planes = normalize_planes(clip, planes)
    neutral = get_neutral_values(clip)

    if complexpr_available:
        expr = 'y z - D! x D@ y {mid} clamp D@ {mid} y clamp - -'
    else:
        expr = 'x y z - y min {mid} max y z - {mid} min y max - -'

    return norm_expr([clip, diffa, diffb], expr, planes, mid=neutral)


def median_clips(*_clips: vs.VideoNode | Iterable[vs.VideoNode], planes: PlanesT = None) -> vs.VideoNode:
    warnings.warn('median_clips: Function is deprecated and will be removed in a later version! Use MeanMode.MEDIAN')
    return MeanMode.MEDIAN(*_clips, planes=planes, func=median_clips)


def flux_smooth(
    clip: vs.VideoNode, radius: int = 2, threshold: int = 7, scenechange: int = 24, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, flux_smooth)

    if radius < 1 or radius > 7:
        raise CustomIndexError('Radius must be between 1 and 7 (inclusive)!', flux_smooth, reason=radius)

    planes = normalize_planes(clip, planes)

    threshold = threshold << clip.format.bits_per_sample - 8

    cthreshold = threshold if (1 in planes or 2 in planes) else 0

    median = clip.tmedian.TemporalMedian(radius, planes)  # type: ignore
    average = clip.focus2.TemporalSoften2(  # type: ignore
        radius, threshold, cthreshold, scenechange
    )

    return limit_filter(average, clip, median, LimitFilterMode.DIFF_MIN, planes)
