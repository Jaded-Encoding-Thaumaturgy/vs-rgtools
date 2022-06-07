from typing import Sequence

import vapoursynth as vs
from vsutil import disallow_variable_format, disallow_variable_resolution, get_y, iterate, join, split

from .func import minblur
from .rgtools import repair
from .util import get_neutral_value, normalise_planes, normalise_seq

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def contrasharpening(
    flt: vs.VideoNode, src: vs.VideoNode, radius: int | None = None,
    rep: int = 13, planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    """
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was previously removed.
    Script by Didée, at the VERY GRAINY thread (http://forum.doom9.org/showthread.php?p=1076491#post1076491)
    :param flt:         Filtered clip
    :param src:         Source clip
    :param radius:      Spatial radius for contra-sharpening (1-3). Default is 2 for HD / 1 for SD.
    :param rep:         Mode of rgvs.Repair to limit the difference
    :param planes:      Planes to process, defaults to None
    :return:            Contrasharpened clip
    """

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening: Clips must be the same format')

    neutral = normalise_seq([get_neutral_value(flt), get_neutral_value(flt, True)], flt.format.num_planes)

    planes = normalise_planes(src, planes)

    if radius is None:
        radius = 2 if flt.width > 1024 or flt.height > 576 else 1

    # Damp down remaining spots of the denoised clip
    mblur = minblur(flt, radius, planes)

    wmean_mat = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mean_mat = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    rg11 = mblur.std.Convolution(wmean_mat, planes=planes)
    if radius >= 2:
        rg11 = rg11.std.Convolution(mean_mat, planes=planes)
    if radius >= 3:
        rg11 = rg11.std.Convolution(mean_mat, planes=planes)

    # Difference of a simple kernel blur
    diff_blur = mblur.std.MakeDiff(rg11, planes)

    # Difference achieved by the filtering
    diff_flt = src.std.MakeDiff(flt, planes)

    # Limit the difference to the max of what the filtering removed locally
    limit = repair(
        diff_blur, diff_flt, [rep if i in planes else 0 for i in range(flt.format.num_planes)]
    )

    # abs(diff) after limiting may not be bigger than before
    limit = core.std.Expr(
        [limit, diff_blur], [
            f'x {mid} - abs y {mid} - abs < x y ?' if i in planes else ''
            for i, mid in enumerate(neutral)
        ]
    )

    # Apply the limited difference (sharpening is just inverse blurring)
    return flt.std.MergeDiff(limit, planes)


@disallow_variable_format
def contrasharpening_dehalo(dehaloed: vs.VideoNode, src: vs.VideoNode, level: float) -> vs.VideoNode:
    """
    :param dehaloed:    Dehaloed clip
    :param src:         Source clip
    :param level:       Strengh level
    :return:            Contrasharpened clip
    """
    if dehaloed.format.id != src.format.id:
        raise ValueError('contrasharpening: Clips must be the same format')

    dehaloed_y, *chroma = split(dehaloed)
    neutral = get_neutral_value(dehaloed)

    weighted = dehaloed_y.std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1])
    weighted2 = weighted.ctmf.CTMF(radius=2)
    weighted2 = iterate(weighted2, lambda c: repair(c, weighted, 1), 2)

    diff = weighted.std.MakeDiff(weighted2).std.Expr(f'x {neutral} - 2.49 * {level} * {neutral} +')
    diff2 = core.std.Expr(
        [diff, get_y(src).std.MakeDiff(dehaloed_y)],
        f'x {neutral} - y {neutral} - * 0 < {neutral} x {neutral} - abs y {neutral} - abs < x y ? ?'
    )

    return join([dehaloed_y.std.MergeDiff(diff2)] + chroma, dehaloed.format.color_family)
