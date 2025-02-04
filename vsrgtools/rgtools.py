from __future__ import annotations

from vsexprtools import complexpr_available, expr_func
from vstools import NotFoundEnumValue, PlanesT, check_variable, core, normalize_seq, pick_func_stype, vs

from .aka_expr import (
    aka_removegrain_expr_11_12, aka_removegrain_expr_19, aka_removegrain_expr_20, aka_removegrain_expr_23,
    aka_removegrain_expr_24, removegrain_aka_exprs, repair_aka_exprs
)
from .enum import (
    BlurMatrix, RemoveGrainMode, RemoveGrainModeT, RepairMode, RepairModeT, VerticalCleanerMode, VerticalCleanerModeT
)

__all__ = [
    'repair', 'removegrain',
    'clense', 'backward_clense', 'forward_clense',
    'vertical_cleaner'
]


def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: RepairModeT) -> vs.VideoNode:
    assert check_variable(clip, repair)
    assert check_variable(repairclip, repair)

    is_float = clip.format.sample_type == vs.FLOAT
    mode = normalize_seq(mode, clip.format.num_planes)

    if not sum(mode):
        return clip

    if not complexpr_available:
        if (RepairMode.CLIP_REF_RG20 in mode or RepairMode.CLIP_REF_RG23 in mode) and is_float:
            raise NotFoundEnumValue(
                'Specified RepairMode for rgsf is not implemented!', repair, reason=iter(mode)
            )

        return pick_func_stype(clip, core.rgvs.Repair, core.rgsf.Repair)(clip, repairclip, mode)

    return core.akarin.Expr(
        [clip, repairclip], [repair_aka_exprs[m]() for m in mode], clip.format.id, True
    )


def removegrain(clip: vs.VideoNode, mode: RemoveGrainModeT) -> vs.VideoNode:
    assert check_variable(clip, removegrain)

    mode = normalize_seq(mode, clip.format.num_planes)
    mode = list(map(RemoveGrainMode, mode))

    if not sum(mode):
        return clip

    if clip.format.sample_type == vs.INTEGER and all(m in range(24 + 1) for m in mode):
        if hasattr(core, "zsmooth"):
            return clip.zsmooth.RemoveGrain(mode)

        if hasattr(core, 'rgvs'):
            return clip.rgvs.RemoveGrain(mode)

    if not complexpr_available:
        return clip.zsmooth.RemoveGrain(mode)

    expr = list[str]()

    for idx, m in enumerate(mode):
        if m == RemoveGrainMode.BINOMIAL_BLUR:
            if all(mm == m for mm in mode):
                return BlurMatrix.BINOMIAL()(clip)
            expr.append(aka_removegrain_expr_11_12())

        elif RemoveGrainMode.BOB_TOP_CLOSE <= m <= RemoveGrainMode.BOB_BOTTOM_INTER:
            return pick_func_stype(clip, core.lazy.rgvs.RemoveGrain, core.lazy.zsmooth.RemoveGrain)(clip, mode)

        elif m == RemoveGrainMode.BOX_BLUR_NO_CENTER:
            if set(mode) == {RemoveGrainMode.BOX_BLUR_NO_CENTER}:
                return BlurMatrix.CIRCLE()(clip)
            expr.append(aka_removegrain_expr_19())

        elif m == RemoveGrainMode.BOX_BLUR:
            if set(mode) == {RemoveGrainMode.BOX_BLUR}:
                return BlurMatrix.MEAN()(clip)
            expr.append(aka_removegrain_expr_20())

        elif m == RemoveGrainMode.EDGE_DEHALO:
            expr.append(aka_removegrain_expr_23(0 if idx == 0 else -0.5))

        elif m == RemoveGrainMode.EDGE_DEHALO2:
            expr.append(aka_removegrain_expr_24(0 if idx == 0 else -0.5))

        else:
            expr.append(removegrain_aka_exprs[m]())

    return expr_func(clip, expr, opt=True)


def clense(
    clip: vs.VideoNode,
    previous_clip: vs.VideoNode | None = None, next_clip: vs.VideoNode | None = None,
    planes: PlanesT = None
) -> vs.VideoNode:
    return pick_func_stype(clip, core.lazy.rgvs.Clense, core.lazy.rgsf.Clense)(clip, previous_clip, next_clip, planes)


def forward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return pick_func_stype(clip, core.lazy.rgvs.ForwardClense, core.lazy.rgsf.ForwardClense)(clip, planes)


def backward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return pick_func_stype(clip, core.lazy.rgvs.BackwardClense, core.lazy.rgsf.BackwardClense)(clip, planes)


def vertical_cleaner(clip: vs.VideoNode, mode: VerticalCleanerModeT = VerticalCleanerMode.MEDIAN) -> vs.VideoNode:
    return pick_func_stype(clip, core.lazy.rgvs.VerticalCleaner, core.lazy.rgsf.VerticalCleaner)(clip, mode)
