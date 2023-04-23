from __future__ import annotations

from functools import partial
from itertools import count
from math import e, log, pi, sin, sqrt
from typing import Any, Iterable, Literal

from vsexprtools import ExprOp, ExprVars, complexpr_available, norm_expr
from vstools import (
    ColorRange, ConvMode, CustomIndexError, CustomValueError, PlanesT, StrList, VSFunction, check_ref_clip,
    check_variable, disallow_variable_format, disallow_variable_resolution, flatten, get_video_format, get_y, join,
    normalize_planes, scale_value, split, vs
)

from .blur import box_blur, gauss_blur

__all__ = [
    'replace_low_frequencies',
    'diff_merge', 'lehmer_diff_merge'
]


@disallow_variable_format
@disallow_variable_resolution
def replace_low_frequencies(
    flt: vs.VideoNode, ref: vs.VideoNode, LFR: float, DCTFlicker: bool = False,
    planes: PlanesT = None, mode: ConvMode = ConvMode.SQUARE
) -> vs.VideoNode:
    assert check_variable(flt, replace_low_frequencies)
    check_ref_clip(flt, ref, replace_low_frequencies)

    planes = normalize_planes(flt, planes)
    work_clip, *chroma = split(flt) if planes == [0] else (flt, )
    assert check_variable(work_clip, replace_low_frequencies)

    ref_work_clip = get_y(ref) if work_clip.format.num_planes == 1 else ref

    LFR = max(LFR or (300 * work_clip.width / 1920), 50)

    freq_sample = max(work_clip.width, work_clip.height) * 2    # Frequency sample rate is resolution * 2 (for Nyquist)
    k = sqrt(log(2) / 2) * LFR                                  # Constant for -3dB
    LFR = freq_sample / (k * 2 * pi)                            # Frequency Cutoff for Gaussian Sigma
    sec0 = sin(e) + 0.1

    sec = scale_value(sec0, 8, work_clip.format.bits_per_sample, range_out=ColorRange.FULL)

    expr = 'x y - z + '

    if DCTFlicker:
        expr += f'y z - D! y z = swap dup D@ 0 = 0 D@ 0 < -1 1 ? ? {sec} * + ?'

    flt_blur = gauss_blur(work_clip, LFR, None, mode)
    ref_blur = gauss_blur(ref_work_clip, LFR, None, mode)

    final = norm_expr(
        [work_clip, flt_blur, ref_blur], expr, planes, force_akarin='vsrgtools.replace_low_frequencies'
    )

    if not chroma:
        return final

    return join([final, *chroma], flt.format.color_family)


def diff_merge(
    *_clips: vs.VideoNode | Iterable[vs.VideoNode],
    filter: VSFunction | list[VSFunction] = partial(box_blur, radius=1, passes=3),
    operator: Literal['<', '>'] = '>', abs_diff: bool = True, **kwargs: Any
) -> vs.VideoNode:

    clips = list[vs.VideoNode](flatten(_clips))  # type: ignore
    n_clips = len(clips)

    if not complexpr_available and n_clips > 13:
        raise CustomIndexError(f'Too many clips passed! ({n_clips})', diff_merge)
    elif n_clips < 2:
        raise CustomIndexError(f'You must pass at least two clips! ({n_clips})', diff_merge)

    if not filter:
        raise CustomValueError('You must pass at least one filter!', diff_merge)

    if not isinstance(filter, list):
        filter = [filter]

    if (lf := len(filter)) < n_clips:
        filter.extend(filter[-1:] * (n_clips - lf))
    elif lf > n_clips:
        filter = filter[:n_clips]

    blurs = [filt(clip, **kwargs) for filt, clip in zip(filter, clips)]

    expr_string = ''
    n_clips = len(clips)

    for src, flt in zip(ExprVars(n_clips), ExprVars(n_clips, n_clips * 2)):
        expr_string += f'{src} {flt} - {abs_diff and "abs" or ""} {src.upper()}D! '

    for i, src, srcn in zip(count(), ExprVars(n_clips), ExprVars(1, n_clips)):
        expr_string += f'{src.upper()}D@ {srcn.upper()}D@ {operator} {src} '

        if i == n_clips - 2:
            expr_string += f'{srcn} '

    expr_string += '? ' * (n_clips - 1)

    return norm_expr([*clips, *blurs], expr_string, force_akarin='vsrgtools.diff_merge')


def lehmer_diff_merge(
    *_clips: vs.VideoNode | Iterable[vs.VideoNode],
    filter: VSFunction | list[VSFunction] = partial(box_blur, radius=3, passes=2),
    planes: PlanesT = None
) -> vs.VideoNode:
    """
    Merge multiple sources.

    :param clips:           Clips to be merged together.
    :param filter:          Filter to be applied to clips to isolate high frequencies.
                            If a list, each filter will be mapped to corresponding clip at same index.
                            If not enough filters are passed, the last one will be reused.
    :param planes:          Planes to be processed.

    :return:                Merged clips.
    """

    clips = list[vs.VideoNode](flatten(_clips))  # type: ignore
    n_clips = len(clips)

    if n_clips < 2:
        raise CustomIndexError('You must pass at least two clips!', lehmer_diff_merge, n_clips)

    if not filter:
        raise CustomValueError('You must pass at least one filter!', lehmer_diff_merge)

    formats = {get_video_format(clip).id for clip in clips}

    if len(formats) > 1:
        raise CustomValueError('All clips must have the same format!', lehmer_diff_merge)

    if not isinstance(filter, list):
        filter = [filter]

    if (lf := len(filter)) < n_clips:
        filter.extend(filter[-1:] * (n_clips - lf))
    elif lf > n_clips:
        filter = filter[:n_clips]

    blurred_clips = [filt(clip) for filt, clip in zip(filter, clips)]

    counts = range(n_clips)

    clip_vars, blur_vars = ExprVars(n_clips), ExprVars(n_clips, n_clips * 2)

    n_op = n_clips - 1

    expr = StrList([
        [f'{clip} {blur} - D{i}!' for i, clip, blur in zip(counts, clip_vars, blur_vars)],
    ])

    for y in range(2):
        expr.extend([
            [f'D{i}@ {3 - y} pow' for i in counts],
            ExprOp.ADD * n_op, f'P{y + 1}!'
        ])

    expr.extend([
        'P2@ 0 = 0 P1@ P2@ / ?',
        blur_vars, ExprOp.ADD * n_op,
        n_clips, ExprOp.DIV, ExprOp.ADD
    ])

    return norm_expr([*clips, *blurred_clips], expr, planes)
