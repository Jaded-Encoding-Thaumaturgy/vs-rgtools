from __future__ import annotations

from typing import Any

from vsexprtools import norm_expr
from vstools import (
    CustomTypeError, PlanesT, VSFunction, check_ref_clip, check_variable, FunctionUtil, normalize_planes, vs, ConvMode
)

from .blur import gauss_blur, min_blur, box_blur, median_blur
from .enum import BlurMatrix
from .limit import limit_filter
from .rgtools import repair
from .util import normalize_radius

__all__ = [
    'unsharpen',
    'unsharp_masked',
    'limit_usm',
    'fine_sharp',
    'soothe'
]


def unsharpen(
    clip: vs.VideoNode, strength: float = 1.0, sigma: float | list[float] = 1.5,
    prefilter: vs.VideoNode | VSFunction | None = None, **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, unsharpen)

    ref = prefilter(clip) if callable(prefilter) else prefilter

    check_ref_clip(clip, ref)

    den = ref or clip
    blur = gauss_blur(den, sigma, **kwargs)

    unsharp = norm_expr([den, blur], f'x y - {strength} * x +', 0)

    if ref is not None:
        return unsharp

    return unsharp.std.MergeDiff(clip.std.MakeDiff(den))


def unsharp_masked(
    clip: vs.VideoNode, radius: int | list[int] = 1, strength: float = 100.0, planes: PlanesT = None
) -> vs.VideoNode:
    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, unsharp_masked, radius, planes, strength=strength)

    blurred = BlurMatrix.LOG(radius, strength=strength)(clip, planes)

    return norm_expr([clip, blurred], 'x dup y - +')


def limit_usm(
    clip: vs.VideoNode, blur: int | vs.VideoNode | VSFunction = 1, thr: int | tuple[int, int] = 3,
    elast: float = 4.0, bright_thr: int | None = None, planes: PlanesT = None
) -> vs.VideoNode:
    """Limited unsharp_masked."""

    if callable(blur):
        blurred = blur(clip)
    elif isinstance(blur, vs.VideoNode):
        blurred = blur
    elif blur <= 0:
        blurred = min_blur(clip, -blur, planes=planes)
    elif blur == 1:
        blurred = BlurMatrix.BINOMIAL()(clip, planes)
    elif blur == 2:
        blurred = BlurMatrix.MEAN()(clip, planes)
    else:
        raise CustomTypeError("'blur' must be an int, clip or a blurring function!", limit_usm, blur)

    sharp = norm_expr([clip, blurred], 'x dup y - +', planes)

    return limit_filter(sharp, clip, thr=thr, elast=elast, bright_thr=bright_thr)


def fine_sharp(
        clip: vs.VideoNode, mode: int = 1, sstr: float = 2.0, cstr: float | None = None, xstr: float = 0.19,
        lstr: float = 1.49, pstr: float = 1.272, ldmp: float | None = None, planes: PlanesT = 0
) -> vs.VideoNode:
    from scipy import interpolate

    func = FunctionUtil(clip, fine_sharp, planes)
    
    if cstr is None:
        cstr = interpolate.CubicSpline(
            x=(0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0, 255.0),
            y=(0, 0.1, 0.6, 0.9, 1.0, 1.09, 1.15, 1.19, 1.249, 1.5)
        )(sstr)

    if ldmp is None:
        ldmp = sstr + 0.1

    blur_kernel = BlurMatrix.BINOMIAL()
    blur_kernel2 = blur_kernel

    if mode < 0:
        cstr **= 0.8
        blur_kernel2 = box_blur

    mode = abs(mode)

    if mode == 1:
        blurred = median_blur(blur_kernel(func.work_clip))
    elif mode > 1:
        blurred = blur_kernel(median_blur(func.work_clip))
    if mode == 3:
        blurred = median_blur(blurred)

    diff = norm_expr(
        [func.work_clip, blurred],
        'range_size 256 / SCL! x y - SCL@ / D! D@ abs DA! DA@ {lstr} / 1 {pstr} / pow {sstr} * '
        'D@ DA@ 0.001 + / * D@ 2 pow D@ 2 pow {ldmp} + / * SCL@ * neutral +',
        lstr=lstr, pstr=pstr, sstr=sstr, ldmp=ldmp
    )

    sharp = func.work_clip

    if sstr:
        sharp = sharp.std.MergeDiff(diff)

    if cstr:
        diff = norm_expr(diff, 'x neutral - {cstr} * neutral +', cstr=cstr)
        diff = blur_kernel2(diff)
        sharp = sharp.std.MakeDiff(diff)

    if xstr:
        xysharp = norm_expr([sharp, box_blur(sharp)], 'x x y - 9.9 * +')
        rpsharp = repair(xysharp, sharp, 12)
        sharp = rpsharp.std.Merge(sharp, weight=[1 - xstr])

    return func.return_clip(sharp)


def soothe(
    flt: vs.VideoNode, src: vs.VideoNode, spatial_strength: int = 0, temporal_strength: int = 25,
    spatial_radius: int = 1, temporal_radius: int = 1, scenechange: bool = False, planes: PlanesT = 0
) -> vs.VideoNode:
    sharp_diff = src.std.MakeDiff(flt, planes)

    expr = (
        'x neutral - X! y neutral - Y! X@ 0 < Y@ 0 < xor X@ 100 / {strength} * '
        'X@ abs Y@ abs > X@ {strength} * Y@ 100 {strength} - * + 100 / X@ ? ? neutral +'
    )

    if spatial_strength:
        soothe = box_blur(sharp_diff, radius=spatial_radius, planes=planes)
        strength = 100 - abs(max(min(spatial_strength, 100), 0))
        sharp_diff = norm_expr([sharp_diff, soothe], expr, strength=strength, planes=planes)

    if temporal_strength:
        soothe = (
            BlurMatrix.MEAN(temporal_radius, mode=ConvMode.TEMPORAL)
            (sharp_diff, planes=planes, scenechange=scenechange)
        )
        strength = 100 - abs(max(min(temporal_strength, 100), -100))
        sharp_diff = norm_expr([sharp_diff, soothe], expr, strength=strength, planes=planes)

    return src.std.MakeDiff(sharp_diff, planes)
