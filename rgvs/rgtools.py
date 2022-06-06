from __future__ import annotations

__all__ = ['repair', 'removegrain', 'vertical_cleaner', 'clense', 'backward_clense', 'forward_clense']

import warnings
from typing import List, Sequence

import vapoursynth as vs
from vsutil import disallow_variable_format

from ._expr_rg import (
    aka_removegrain_expr_1, aka_removegrain_expr_2_4, aka_removegrain_expr_5, aka_removegrain_expr_6,
    aka_removegrain_expr_7, aka_removegrain_expr_8, aka_removegrain_expr_9, aka_removegrain_expr_10,
    aka_removegrain_expr_11_12, aka_removegrain_expr_17, aka_removegrain_expr_18, aka_removegrain_expr_19,
    aka_removegrain_expr_20, aka_removegrain_expr_21_22, aka_removegrain_expr_23, aka_removegrain_expr_24,
    aka_removegrain_expr_26, aka_removegrain_expr_27, aka_removegrain_expr_28
)
from ._expr_rp import (
    aka_repair_expr_1_4, aka_repair_expr_5, aka_repair_expr_6, aka_repair_expr_7, aka_repair_expr_8,
    aka_repair_expr_9, aka_repair_expr_10, aka_repair_expr_11_14, aka_repair_expr_15, aka_repair_expr_16,
    aka_repair_expr_17, aka_repair_expr_18, aka_repair_expr_19, aka_repair_expr_20, aka_repair_expr_21,
    aka_repair_expr_22, aka_repair_expr_23, aka_repair_expr_24, aka_repair_expr_26, aka_repair_expr_27,
    aka_repair_expr_28
)
from .util import normalise_seq, pick_rg

core = vs.core


@disallow_variable_format
def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: int | Sequence[int]) -> vs.VideoNode:
    assert clip.format

    mode = normalise_seq(mode, clip.format.num_planes)

    try:
        aka_expr = core.akarin.Expr
    except AttributeError:
        pass
    else:
        expr: List[str] = []
        for m in mode:
            if m == 0:
                expr.append('')
            elif 1 <= m <= 4:
                expr.append(aka_repair_expr_1_4(m))
            elif m == 5:
                expr.append(aka_repair_expr_5())
            elif m == 6:
                expr.append(aka_repair_expr_6())
            elif m == 7:
                expr.append(aka_repair_expr_7())
            elif m == 8:
                expr.append(aka_repair_expr_8())
            elif m == 9:
                expr.append(aka_repair_expr_9())
            elif m == 10:
                expr.append(aka_repair_expr_10())
            elif 11 <= m <= 14:
                expr.append(aka_repair_expr_11_14(m - 10))
            elif m == 15:
                expr.append(aka_repair_expr_15())
            elif m == 16:
                expr.append(aka_repair_expr_16())
            elif m == 17:
                expr.append(aka_repair_expr_17())
            elif m == 18:
                expr.append(aka_repair_expr_18())
            elif m == 19:
                expr.append(aka_repair_expr_19())
            elif m == 20:
                expr.append(aka_repair_expr_20())
            elif m == 21:
                expr.append(aka_repair_expr_21())
            elif m == 22:
                expr.append(aka_repair_expr_22())
            elif m == 23:
                expr.append(aka_repair_expr_23())
            elif m == 24:
                expr.append(aka_repair_expr_24())
            elif m == 25:
                raise ValueError('repair: invalid mode specified')
            elif m == 26:
                expr.append(aka_repair_expr_26())
            elif m == 27:
                expr.append(aka_repair_expr_27())
            elif m == 28:
                expr.append(aka_repair_expr_28())
        return aka_expr([clip, repairclip], expr)
    if (20 in mode or 23 in mode) and clip.format.sample_type == vs.FLOAT:
        raise ValueError('rgsf is wrong')
    return pick_rg(clip, core.rgvs.Repair, core.rgsf.Repair)(clip, repairclip, mode)


@disallow_variable_format
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
        expr: List[str] = []
        for idx, m in enumerate(mode):
            if m == 0:
                expr.append('')
            if m == 1:
                expr.append(aka_removegrain_expr_1())
            elif 2 <= m <= 4:
                expr.append(aka_removegrain_expr_2_4(m))
            elif m == 5:
                expr.append(aka_removegrain_expr_5())
            elif m == 6:
                expr.append(aka_removegrain_expr_6())
            elif m == 7:
                expr.append(aka_removegrain_expr_7())
            elif m == 8:
                expr.append(aka_removegrain_expr_8())
            elif m == 9:
                expr.append(aka_removegrain_expr_9())
            elif m == 10:
                expr.append(aka_removegrain_expr_10())
            elif 11 <= m <= 12:
                if all(mm == m for mm in mode):
                    return clip.std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1])
                expr.append(aka_removegrain_expr_11_12())
            elif 13 <= m <= 16:
                return pick_rg(clip, clip.rgvs.RemoveGrain, clip.rgsf.RemoveGrain)(mode)
            elif m == 17:
                expr.append(aka_removegrain_expr_17())
            elif m == 18:
                expr.append(aka_removegrain_expr_18())
            elif m == 19:
                if all(mm == 19 for mm in mode):
                    return clip.std.Convolution([1, 1, 1, 1, 0, 1, 1, 1, 1])
                expr.append(aka_removegrain_expr_19())
            elif m == 20:
                if all(mm == 20 for mm in mode):
                    return clip.std.Convolution([1] * 9)
                expr.append(aka_removegrain_expr_20())
            elif m == 21:
                expr.append(aka_removegrain_expr_21_22())
            elif m == 22:
                expr.append(aka_removegrain_expr_21_22())
            elif m == 23:
                expr.append(aka_removegrain_expr_23(0 if idx == 0 else -0.5))
            elif m == 24:
                expr.append(aka_removegrain_expr_24(0 if idx == 0 else -0.5))
            elif m == 25:
                expr.append('')
                warnings.warn('mode 25 isn\'t implemented yet')
            elif m == 26:
                expr.append(aka_removegrain_expr_26())
            elif m == 27:
                expr.append(aka_removegrain_expr_27())
            elif m == 28:
                expr.append(aka_removegrain_expr_28())
        return aka_expr(clip, expr).std.SetFrameProps(AkaExpr=mode)
    return clip.rgsf.RemoveGrain(mode)


@disallow_variable_format
def clense(
    clip: vs.VideoNode,
    previous_clip: vs.VideoNode | None = None, next_clip: vs.VideoNode | None = None,
    planes: int | Sequence[int] | None = None, grey: bool = False
) -> vs.VideoNode:
    return pick_rg(clip, clip.rgvs.Clense, clip.rgsf.Clense)(previous_clip, next_clip, planes)


@disallow_variable_format
def forward_clense(clip: vs.VideoNode, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    return pick_rg(clip, clip.rgvs.ForwardClense, clip.rgsf.ForwardClense)(planes)


@disallow_variable_format
def backward_clense(clip: vs.VideoNode, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    return pick_rg(clip, clip.rgvs.BackwardClense, clip.rgsf.BackwardClense)(planes)


@disallow_variable_format
def vertical_cleaner(clip: vs.VideoNode, mode: int | Sequence[int]) -> vs.VideoNode:
    return pick_rg(clip, clip.rgvs.VerticalCleaner, clip.rgsf.VerticalCleaner)(mode)
