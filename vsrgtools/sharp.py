from __future__ import annotations

from vsexprtools import norm_expr
from vstools import CustomTypeError, PlanesT, VSFunction, check_ref_clip, check_variable, normalize_planes, vs

from .blur import gauss_blur, min_blur
from .enum import BlurMatrix
from .limit import limit_filter
from .util import normalize_radius

__all__ = [
    'unsharpen',
    'unsharp_masked',
    'limit_usm',
    'shimmer_soothe'
]


def unsharpen(
    clip: vs.VideoNode, strength: float = 1.0, sigma: float | list[float] = 1.5,
    prefilter: vs.VideoNode | VSFunction | None = None
) -> vs.VideoNode:
    assert check_variable(clip, unsharpen)

    ref = prefilter(clip) if callable(prefilter) else prefilter

    check_ref_clip(clip, ref)

    den = ref or clip
    blur = gauss_blur(den, sigma)

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

    blurred = BlurMatrix.log(radius, strength)(clip, planes)

    return norm_expr([clip, blurred], 'x dup y - +')


def limit_usm(
    clip: vs.VideoNode, blur: int | vs.VideoNode | VSFunction = 1, thr: int | tuple[int, int] = 3,
    elast: float = 4.0, bright_thr: int | None = None, planes: PlanesT = None
) -> vs.VideoNode:
    """Limited unsharp_masked."""

    if callable(blur):
        blurred = blur(clip)
    elif isinstance(blur, vs.VideoNode):
        blurred = blur  # type: ignore
    elif blur <= 0:  # type: ignore
        blurred = min_blur(clip, -blur, planes)  # type: ignore
    elif blur == 1:
        blurred = BlurMatrix.WMEAN(clip, planes)
    elif blur == 2:
        blurred = BlurMatrix.MEAN(clip, planes)
    else:
        raise CustomTypeError("'blur' must be an int, clip or a blurring function!", limit_usm, blur)

    sharp = norm_expr([clip, blurred], 'x dup y - +', planes)

    return limit_filter(sharp, clip, thr=thr, elast=elast, bright_thr=bright_thr)


def shimmer_soothe(
    sharp: vs.VideoNode, orig: vs.VideoNode, strength: int = 75, tr: int = 1,
    scenechange: bool = False, planes: PlanesT = 0
) -> vs.VideoNode:
    diff = orig.std.MakeDiff(sharp, planes)

    strength = 100 - max(min(100, strength), 0)

    soft_diff = diff.std.AverageFrames([1] * (tr * 2 + 1), None, scenechange, planes)

    limit_diff = norm_expr([diff, soft_diff], [
        'x range_diff - y range_diff - xor x range_diff - 100 / {strength} '
        f'* range_diff + x range_diff - abs y range_diff - abs > x {strength} '
        f'* y 100 {strength} - * + 100 / x ? ?'
    ], planes, strength=strength)

    return norm_expr([orig, limit_diff], 'x y range_diff - -', planes)
