from __future__ import annotations

__all__ = [
    'repair', 'removegrain',
    'clense', 'backward_clense', 'forward_clense',
    'vertical_cleaner', 'horizontal_cleaner'
]

from typing import Sequence

import vapoursynth as vs
from vsutil import disallow_variable_format, disallow_variable_resolution

from .aka_expr import (
    aka_removegrain_expr_11_12, aka_removegrain_expr_19, aka_removegrain_expr_20, aka_removegrain_expr_23,
    aka_removegrain_expr_24, removegrain_aka_exprs, repair_aka_exprs
)
from .blur import box_blur
from .util import PlanesT, normalise_seq, pick_func_stype, wmean_matrix

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: int | Sequence[int]) -> vs.VideoNode:
    assert clip.format

    mode = normalise_seq(mode, clip.format.num_planes)

    try:
        aka_expr = core.akarin.Expr
    except AttributeError:
        pass
    else:
        expr = list[str]()
        for m in mode:
            if m == 0:
                expr.append('')
            else:
                try:
                    function = repair_aka_exprs[m - 1]
                    if function is None:
                        raise ValueError
                    expr.append(function())
                except BaseException:
                    raise ValueError('repair: invalid mode specified')

        return aka_expr([clip, repairclip], expr)

    if (20 in mode or 23 in mode) and clip.format.sample_type == vs.FLOAT:
        raise ValueError('repair: rgsf mode is wrong')

    return pick_func_stype(clip, core.rgvs.Repair, core.rgsf.Repair)(clip, repairclip, mode)


@disallow_variable_format
@disallow_variable_resolution
def removegrain(clip: vs.VideoNode, mode: int | Sequence[int]) -> vs.VideoNode:
    assert clip.format

    mode = normalise_seq(mode, clip.format.num_planes)

    # rgvs is faster for integer clips
    if clip.format.sample_type == vs.INTEGER and all(m in range(24 + 1) for m in mode):
        return clip.rgvs.RemoveGrain(mode)

    try:
        aka_expr = core.akarin.Expr
    except AttributeError:
        pass
    else:
        expr = list[str]()
        for idx, m in enumerate(mode):
            if m == 0:
                expr.append('')
            elif 11 <= m <= 12:
                if all(mm == m for mm in mode):
                    return box_blur(clip, wmean_matrix)
                expr.append(aka_removegrain_expr_11_12())
            elif 13 <= m <= 16:
                return pick_func_stype(clip, clip.rgvs.RemoveGrain, clip.rgsf.RemoveGrain)(mode)
            elif m == 19:
                if all(mm == 19 for mm in mode):
                    return clip.std.Convolution([1, 1, 1, 1, 0, 1, 1, 1, 1])
                expr.append(aka_removegrain_expr_19())
            elif m == 20:
                if all(mm == 20 for mm in mode):
                    return clip.std.Convolution([1] * 9)
                expr.append(aka_removegrain_expr_20())
            elif m == 23:
                expr.append(aka_removegrain_expr_23(0 if idx == 0 else -0.5))
            elif m == 24:
                expr.append(aka_removegrain_expr_24(0 if idx == 0 else -0.5))
            else:
                try:
                    function = removegrain_aka_exprs[m - 1]
                    if function is None:
                        raise ValueError
                    expr.append(function())
                except BaseException:
                    raise ValueError('removegrain: invalid mode specified')

        return aka_expr(clip, expr).std.SetFrameProps(AkaExpr=mode)
    return clip.rgsf.RemoveGrain(mode)


@disallow_variable_format
@disallow_variable_resolution
def clense(
    clip: vs.VideoNode,
    previous_clip: vs.VideoNode | None = None, next_clip: vs.VideoNode | None = None,
    planes: PlanesT = None
) -> vs.VideoNode:
    return pick_func_stype(clip, clip.rgvs.Clense, clip.rgsf.Clense)(previous_clip, next_clip, planes)


@disallow_variable_format
@disallow_variable_resolution
def forward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return pick_func_stype(clip, clip.rgvs.ForwardClense, clip.rgsf.ForwardClense)(planes)


@disallow_variable_format
@disallow_variable_resolution
def backward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return pick_func_stype(clip, clip.rgvs.BackwardClense, clip.rgsf.BackwardClense)(planes)


@disallow_variable_format
@disallow_variable_resolution
def vertical_cleaner(clip: vs.VideoNode, mode: int | Sequence[int]) -> vs.VideoNode:
    return pick_func_stype(clip, clip.rgvs.VerticalCleaner, clip.rgsf.VerticalCleaner)(mode)


@disallow_variable_format
@disallow_variable_resolution
def horizontal_cleaner(clip: vs.VideoNode, mode: int | Sequence[int]) -> vs.VideoNode:
    return vertical_cleaner(clip.std.Transpose(), mode).std.Transpose()
