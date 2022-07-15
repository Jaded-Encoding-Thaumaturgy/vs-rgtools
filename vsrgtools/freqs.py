from __future__ import annotations

from functools import partial
from itertools import count
from math import e, log, pi, sin, sqrt
from typing import Any, Literal

import vapoursynth as vs
from vsutil import EXPR_VARS
from vsutil import Range as CRange
from vsutil import disallow_variable_format, disallow_variable_resolution, get_y, scale_value, split

from .blur import box_blur, gauss_blur
from .enum import ConvMode
from .util import PlanesT, VSFunc, aka_expr_available, norm_expr_planes, normalise_planes

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
    *clips: vs.VideoNode, filter: VSFunc = partial(box_blur, radius=1, passes=3),
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
    low_filter: VSFunc = partial(box_blur, radius=3, passes=2),
    high_filter: VSFunc | None = None,
    planes: PlanesT = None
) -> vs.VideoNode:
    if high_filter is None:
        high_filter = low_filter

    low_filtered = low_filter(src)
    high_filtered = high_filter(flt)

    if aka_expr_available:
        return core.akarin.Expr(
            [src, flt, low_filtered, high_filtered],
            norm_expr_planes(
                src, 'x z - SD! y a - FD! SD@ 3 pow FD@ 3 pow + SD@ 2 pow FD@ 2 pow 0.0001 + + / x SD@ - +', planes
            )
        )

    src_high, flt_high = src.std.MakeDiff(low_filtered), flt.std.MakeDiff(high_filtered)

    flt_high = core.std.Expr(
        [src_high, flt_high],
        'x 32768 - 3 pow y 32768 - 3 pow + x 32768 - 2 pow y 32768 - 2 pow 0.0001 + + / 32768 +'
    )

    return src.std.MakeDiff(src_high).std.MergeDiff(flt_high)
