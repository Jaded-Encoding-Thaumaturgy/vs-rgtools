from __future__ import annotations

__all__ = [
    'min_filter', 'median_clips', 'median_diff'
]

from typing import Callable, Sequence

import vapoursynth as vs
from vsutil import disallow_variable_format, disallow_variable_resolution, fallback, get_neutral_value

from .enum import MinFilterMode
from .util import norm_expr_planes, normalise_planes, normalise_seq

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def min_filter(
    clip: vs.VideoNode,
    clip_func: Callable[[vs.VideoNode], vs.VideoNode],
    diff_func: Callable[[vs.VideoNode], vs.VideoNode] | None = None,
    mode: MinFilterMode | None = None,
    planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    assert clip.format

    planes = normalise_planes(clip, planes)

    mode = fallback(
        mode, MinFilterMode.DIFF if diff_func is None else MinFilterMode.CLIP
    )

    diff_func = fallback(diff_func, clip_func)

    if mode == MinFilterMode.CLIP:
        return median_clips([clip, clip_func(clip), diff_func(clip)], planes)

    filtered = clip_func(clip)

    diffa = core.std.MakeDiff(clip, filtered, planes)

    diffb = diff_func(diffa)

    return median_diff(clip, diffa, diffb, planes)


@disallow_variable_format
@disallow_variable_resolution
def median_clips(clips: Sequence[vs.VideoNode], planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    return core.std.Expr(
        list(clips), norm_expr_planes(clips[0], 'x y z min max y z max min', planes)
    )


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