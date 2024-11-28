from __future__ import annotations

from vsexprtools import complexpr_available, expr_func
from vstools import (
    DependencyNotFoundError, FuncExceptT, NotFoundEnumValue, PlanesT, VSFunction, check_variable, core,
    normalize_seq, vs
)

from .aka_expr import (
    aka_removegrain_expr_11_12, aka_removegrain_expr_19, aka_removegrain_expr_20, aka_removegrain_expr_23,
    aka_removegrain_expr_24, removegrain_aka_exprs, repair_aka_exprs
)
from .enum import BlurMatrix, RemoveGrainMode, RemoveGrainModeT, RepairMode, RepairModeT, VerticalCleanerModeT

__all__ = [
    'repair', 'removegrain',

    'pick_rgplugin_stype',

    'clense', 'backward_clense', 'forward_clense',
    'vertical_cleaner', 'horizontal_cleaner'
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

        return pick_rgplugin_stype(clip, repair).Repair(clip, repairclip, mode)

    return core.akarin.Expr(
        [clip, repairclip], [repair_aka_exprs[m]() for m in mode], clip.format.id, True
    )


def removegrain(clip: vs.VideoNode, mode: RemoveGrainModeT) -> vs.VideoNode:
    assert check_variable(clip, removegrain)

    mode = normalize_seq(mode, clip.format.num_planes)
    mode = list(map(RemoveGrainMode, mode))

    if not sum(mode):
        return clip

    # rgvs is faster for integer clips
    if hasattr(core, 'rgvs') and clip.format.sample_type == vs.INTEGER and all(m in range(24 + 1) for m in mode):
        return clip.rgvs.RemoveGrain(mode)

    if not complexpr_available:
        return clip.rgsf.RemoveGrain(mode)

    expr = list[str]()
    for idx, m in enumerate(mode):
        if m == RemoveGrainMode.SQUARE_BLUR:
            if all(mm == m for mm in mode):
                return BlurMatrix.WMEAN(clip)
            expr.append(aka_removegrain_expr_11_12())
        elif RemoveGrainMode.BOB_TOP_CLOSE <= m <= RemoveGrainMode.BOB_BOTTOM_INTER:
            return pick_rgplugin_stype(clip, removegrain).RemoveGrain(mode)
        elif m == RemoveGrainMode.CIRCLE_BLUR:
            if set(mode) == {RemoveGrainMode.CIRCLE_BLUR}:
                return BlurMatrix.CIRCLE(clip)
            expr.append(aka_removegrain_expr_19())
        elif m == RemoveGrainMode.BOX_BLUR:
            if set(mode) == {RemoveGrainMode.BOX_BLUR}:
                return BlurMatrix.MEAN(clip)
            expr.append(aka_removegrain_expr_20())
        elif m == RemoveGrainMode.EDGE_DEHALO:
            expr.append(aka_removegrain_expr_23(0 if idx == 0 else -0.5))
        elif m == RemoveGrainMode.EDGE_DEHALO2:
            expr.append(aka_removegrain_expr_24(0 if idx == 0 else -0.5))
        else:
            expr.append(removegrain_aka_exprs[m]())

    return expr_func(clip, expr, opt=True)


def pick_rgplugin_stype(clip: vs.VideoNode, func_except: FuncExceptT | None = None) -> VSFunction:
    """Pick the RG plugin based on the sample type of the clip."""

    single_available = hasattr(core, 'rgvs')
    float_available = hasattr(core, 'rgsf')

    if not single_available and not float_available:
        raise DependencyNotFoundError(func_except or pick_rgplugin_stype, 'rgvs or rgsf')

    if clip.format.sample_type == vs.FLOAT:
        if not float_available:
            raise DependencyNotFoundError(func_except or pick_rgplugin_stype, 'rgsf')

        return clip.rgsf

    if not single_available:
        raise DependencyNotFoundError(func_except or pick_rgplugin_stype, 'rgvs')

    return clip.rgvs


def clense(
    clip: vs.VideoNode,
    previous_clip: vs.VideoNode | None = None, next_clip: vs.VideoNode | None = None,
    planes: PlanesT = None
) -> vs.VideoNode:
    return pick_rgplugin_stype(clip, clense).Clense(previous_clip, next_clip, planes)


def forward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return (pick_rgplugin_stype(clip, forward_clense)).ForwardClense(planes)


def backward_clense(clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
    return pick_rgplugin_stype(clip, backward_clense).BackwardClense(planes)


def vertical_cleaner(clip: vs.VideoNode, mode: VerticalCleanerModeT, func_except: FuncExceptT | None = None) -> vs.VideoNode:
    return pick_rgplugin_stype(clip, func_except or vertical_cleaner).VerticalCleaner(mode)


def horizontal_cleaner(clip: vs.VideoNode, mode: VerticalCleanerModeT) -> vs.VideoNode:
    return vertical_cleaner(clip.std.Transpose(), mode, horizontal_cleaner).std.Transpose()
