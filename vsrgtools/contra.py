from __future__ import annotations

from functools import partial
from typing import Callable, List

__all__ = ['contrasharpening', 'contrasharpening_dehalo', 'contrasharpening_median', 'contrasharpening_repair']

import vapoursynth as vs
from vsexprtools.util import PlanesT, aka_expr_available, norm_expr_planes, normalise_planes
from vsutil import disallow_variable_format, disallow_variable_resolution, get_neutral_value, iterate

from .blur import blur, box_blur, min_blur
from .enum import RemoveGrainMode, RepairMode
from .rgtools import removegrain, repair
from .util import wmean_matrix

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

    weighted = flt.std.Convolution(wmean_matrix, planes=planes)
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


@disallow_variable_format
@disallow_variable_resolution
def contrasharpening_median(
    flt: vs.VideoNode, src: vs.VideoNode, blur_func: Callable[..., vs.VideoNode] = box_blur, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param flt:         Filtered clip
    :param src:         Source clip
    :param blur_func:   Function used to blur the filtered clip
    :param planes:      Planes to process, defaults to None
    :return:            Contrasharpened clip
    """
    assert flt.format and src.format

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening: Clips must be the same format')

    planes = normalise_planes(flt, planes)

    blur = blur_func(flt, planes=planes)

    return core.std.Expr(
        [flt, src, blur],
        norm_expr_planes(flt, 'x y < x x + z - x max y min x x + z - x min y max ?', planes)
    )


@disallow_variable_format
@disallow_variable_resolution
def contrasharpening_repair(
    flt: vs.VideoNode, src: vs.VideoNode,
    rep: int | RemoveGrainMode | List[int | RemoveGrainMode] = [
        RemoveGrainMode.BOX_BLUR, RemoveGrainMode.SQUARE_BLUR
    ], planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param flt:         Filtered clip
    :param src:         Source clip
    :param rep:         Mode of rgvs.Repair to limit the difference
    :param planes:      Planes to process, defaults to None
    :return:            Contrasharpened clip
    """
    assert flt.format and src.format

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening: Clips must be the same format')

    planes = normalise_planes(flt, planes)
    rmgrain = removegrain(flt, rep if isinstance(rep, list) else [
        rep if i in planes else 0 for i in range(flt.format.num_planes)
    ])

    return core.std.Expr(
        [flt, src, rmgrain], norm_expr_planes(flt, 'x dup + z - x y min max x y max min', planes)
    )
