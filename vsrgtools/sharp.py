from __future__ import annotations

from math import log2
from vsexprtools import norm_expr
from vstools import vs, check_variable, get_sample_type, VSFunction, check_ref_clip, PlanesT, normalize_planes

from .blur import gauss_blur
from .util import normalize_radius

__all__ = [
    'unsharpen',
    'unsharp_masked'
]


def unsharpen(
    clip: vs.VideoNode, strength: float = 1.0, sigma: float | list[float] = 1.5,
    prefilter: vs.VideoNode | VSFunction | None = None
) -> vs.VideoNode:
    assert check_variable(clip, unsharpen)

    if callable(prefilter):
        ref = prefilter(clip)
    else:
        ref = prefilter

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

    if isinstance(radius, int):
        return normalize_radius(clip, unsharp_masked, radius, planes, strength=strength)

    strength = max(1e-6, min(log2(3) * strength / 100, log2(3)))

    weight = 0.5 ** strength / ((1 - 0.5 ** strength) * 0.5)

    if get_sample_type(clip) == 0:
        all_matrices = [[x * 1.0] for x in range(1, 1024)]

        for x in range((2 ** 10) - 1):
            while len(all_matrices[x]) < radius * 2 + 1:
                all_matrices[x].append(all_matrices[x][-1] / weight)

        error = list(map(sum, ([abs(x - round(x)) for x in matrix[1:]] for matrix in all_matrices)))
        matrix = list(map(float, map(round, all_matrices[error.index(min(error))])))
    else:
        matrix = [1.0]

        while len(matrix) < radius * 2 + 1:
            matrix.append(matrix[-1] / weight)

    if radius == 1:
        indices = (2, 1, 2, 1, 0, 1, 2, 1, 2)
    else:
        indices = (4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4)

    matrix = [matrix[x] for x in indices]

    blurred = clip.std.Convolution(matrix, planes=planes)

    return clip.std.MergeDiff(clip.std.MakeDiff(blurred, planes=planes), planes=planes)
