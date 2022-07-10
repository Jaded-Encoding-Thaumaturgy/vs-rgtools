from __future__ import annotations

from math import e, log, pi, sin, sqrt

import vapoursynth as vs
from vsutil import Range as CRange
from vsutil import disallow_variable_format, disallow_variable_resolution, get_y, scale_value, split

from .blur import gauss_blur
from .enum import ConvMode
from .util import PlanesT, norm_expr_planes, normalise_planes

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
    sec0 = sin(e) + .1

    sec = scale_value(sec0, 8, work_clip.format.bits_per_sample, range=CRange.FULL)

    expr = "x y - z + "

    if DCTFlicker:
        expr += f"y z - d! y z = swap dup d@ 0 = 0 d@ 0 < -1 1 ? ? {sec} * + ?"

    flt_blur = gauss_blur(work_clip, LFR, None, mode)
    ref_blur = gauss_blur(ref_work_clip, LFR, None, mode)

    final = core.akarin.Expr([work_clip, flt_blur, ref_blur], norm_expr_planes(work_clip, expr, planes))

    return final if chroma else core.std.ShufflePlanes([final, flt], [0, 1, 2], vs.YUV)
