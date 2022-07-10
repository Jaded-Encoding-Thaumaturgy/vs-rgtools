from __future__ import annotations

from functools import partial

__all__ = ['contrasharpening', 'contrasharpening_dehalo']

import vapoursynth as vs
from vsutil import disallow_variable_format, disallow_variable_resolution, get_neutral_value, iterate

from .blur import blur, box_blur, min_blur
from .enum import RepairMode
from .rgtools import repair
from .util import PlanesT, aka_expr_available, norm_expr_planes, normalise_planes, wmean_matrix

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def contrasharpening(
    flt: vs.VideoNode, src: vs.VideoNode, radius: int | None = None,
    rep: int | RepairMode = RepairMode.MINMAX_SQUARE_REF3, planes: PlanesT = 0
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
    assert flt.format and src.format

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening: Clips must be the same format')

    neutral = [get_neutral_value(flt), get_neutral_value(flt, True)]

    planes = normalise_planes(flt, planes)

    if radius is None:
        radius = 2 if flt.width > 1024 or flt.height > 576 else 1

    # Damp down remaining spots of the denoised clip
    mblur = min_blur(flt, radius, planes)

    rg11 = blur(mblur, radius, planes=planes)

    # Difference of a simple kernel blur
    diff_blur = mblur.std.MakeDiff(rg11, planes)

    # Difference achieved by the filtering
    diff_flt = src.std.MakeDiff(flt, planes)

    # Limit the difference to the max of what the filtering removed locally
    limit = repair(diff_blur, diff_flt, [
        rep if i in planes else 0 for i in range(flt.format.num_planes)
    ])

    # abs(diff) after limiting may not be bigger than before
    # Apply the limited difference (sharpening is just inverse blurring)
    return core.std.Expr(
        [limit, diff_blur, flt],
        norm_expr_planes(flt, 'x {mid} - abs y {mid} - abs < x y ? {mid} - z +', planes, mid=neutral)
    )


@disallow_variable_format
@disallow_variable_resolution
def contrasharpening_dehalo(
        flt: vs.VideoNode, src: vs.VideoNode, level: float = 1.4, planes: PlanesT = 0) -> vs.VideoNode:
    """
    :param dehaloed:    Dehaloed clip
    :param src:         Source clip
    :param level:       Strengh level
    :return:            Contrasharpened clip
    """
    assert flt.format and src.format

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening_dehalo: Clips must be the same format')

    planes = normalise_planes(flt, planes)

    rep_modes = [
        i in planes and RepairMode.MINMAX_SQUARE1 or RepairMode.NONE for i in range(flt.format.num_planes)
    ]

    weighted = box_blur(flt, wmean_matrix, planes)
    weighted2 = weighted.ctmf.CTMF(radius=2, planes=planes)
    weighted2 = iterate(weighted2, partial(repair, repairclip=weighted, mode=rep_modes), 2)

    if aka_expr_available:
        return core.akarin.Expr(
            [weighted, weighted2, src, flt], norm_expr_planes(
                flt, f'x y - 2.49 * {level} * D! z a - DY! D@ DY@ * 0 < 0 D@ abs DY@ abs < D@ DY@ ? ? a +', planes
            )
        )

    neutral = [get_neutral_value(flt), get_neutral_value(flt, True)]

    diff = weighted.std.MakeDiff(weighted2, planes).std.Expr(
        norm_expr_planes(flt, f'x {neutral} - 2.49 * {level} * {neutral} +', planes)
    )

    diff = core.std.Expr(
        [diff, src.std.MakeDiff(flt, planes)], norm_expr_planes(
            flt, 'x {mid} - y {mid} - * 0 < {mid} x {mid} - abs y {mid} - abs < x y ? ?', planes, mid=neutral
        )
    )

    return flt.std.MergeDiff(diff, planes)
