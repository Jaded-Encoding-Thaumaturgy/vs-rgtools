from __future__ import annotations

from functools import partial
from itertools import count
from math import e, log, pi, sin, sqrt
from typing import Any, Literal

import vapoursynth as vs
from vsexprtools.util import PlanesT, VSFunction, aka_expr_available, norm_expr_planes, normalise_planes
from vsutil import EXPR_VARS
from vsutil import Range as CRange
from vsutil import disallow_variable_format, disallow_variable_resolution, get_peak_value, get_y, scale_value, split

from .blur import box_blur, gauss_blur
from .enum import ConvMode

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def replace_low_frequencies(
    flt: vs.VideoNode, ref: vs.VideoNode, LFR: float, DCTFlicker: bool = False,
    planes: PlanesT = None, mode: ConvMode = ConvMode.SQUARE
) -> vs.VideoNode:
    assert flt.format

    planes = normalise_planes(flt, planes)
    work_clip, *chroma = split(flt) if planes == [0] else (flt, )
    assert work_clip.format

    ref_work_clip = get_y(ref) if work_clip.format.num_planes == 1 else ref

    LFR = max(LFR or (300 * work_clip.width / 1920), 50)

    freq_sample = max(work_clip.width, work_clip.height) * 2    # Frequency sample rate is resolution * 2 (for Nyquist)
    k = sqrt(log(2) / 2) * LFR                                  # Constant for -3dB
    LFR = freq_sample / (k * 2 * pi)                            # Frequency Cutoff for Gaussian Sigma
    sec0 = sin(e) + 0.1

    sec = scale_value(sec0, 8, work_clip.format.bits_per_sample, range=CRange.FULL)

    expr = 'x y - z + '

    if DCTFlicker:
        expr += f'y z - D! y z = swap dup D@ 0 = 0 D@ 0 < -1 1 ? ? {sec} * + ?'

    flt_blur = gauss_blur(work_clip, LFR, None, mode)
    ref_blur = gauss_blur(ref_work_clip, LFR, None, mode)

    final = core.akarin.Expr(
        [work_clip, flt_blur, ref_blur], norm_expr_planes(work_clip, expr, planes)
    )

    return final if chroma else core.std.ShufflePlanes([final, flt], [0, 1, 2], vs.YUV)


def diff_merge(
    *clips: vs.VideoNode, filter: VSFunction = partial(box_blur, radius=1, passes=3),
    operator: Literal['<', '>'] = '>', abs_diff: bool = True, **kwargs: Any
) -> vs.VideoNode:
    if len(clips) < 2:
        raise ValueError('diff_merge: you must pass at least two clips!')

    blurs = [filter(clip, **kwargs) for clip in clips]

    expr_string = ''
    n_clips = len(clips)

    for src, flt in zip(EXPR_VARS[:n_clips], EXPR_VARS[n_clips:n_clips * 2]):
        expr_string += f'{src} {flt} - {abs_diff and "abs" or ""} {src.upper()}D! '

    for i, src, srcn in zip(count(), EXPR_VARS[:n_clips], EXPR_VARS[1:n_clips]):
        expr_string += f'{src.upper()}D@ {srcn.upper()}D@ {operator} {src} '

        if i == n_clips - 2:
            expr_string += f'{srcn} '

    expr_string += '? ' * (n_clips - 1)

    return core.akarin.Expr([*clips, *blurs], expr_string)


def lehmer_diff_merge(
    src: vs.VideoNode, flt: vs.VideoNode,
    filter: VSFunction = partial(box_blur, radius=3, passes=2),
    high_filter: VSFunction | None = None,
    planes: PlanesT = None
) -> vs.VideoNode:
    if high_filter is None:
        high_filter = filter

    src_high = filter(src)
    flt_high = high_filter(flt)

    if aka_expr_available:
        return core.akarin.Expr(
            [src, flt, src_high, flt_high],
            norm_expr_planes(
                src, 'x z - SD! y a - FD! SD@ 3 pow FD@ 3 pow + SD@ 2 pow FD@ 2 pow 0.0001 + + / x SD@ - +', planes
            )
        )

    peak = get_peak_value(src)

    src_high_diff = src.std.MakeDiff(src_high, planes)
    flt_high_diff = flt.std.MakeDiff(flt_high, planes)

    flt_high_diff = core.std.Expr(
        [src_high_diff, flt_high_diff],
        norm_expr_planes(
            src, f'x {peak} - 3 pow y {peak} - 3 pow + x {peak} - 2 pow y {peak} - 2 pow 0.0001 + + / {peak} +', planes
        )
    )

    return src.std.MakeDiff(src_high_diff, planes).std.MergeDiff(flt_high_diff, planes)
