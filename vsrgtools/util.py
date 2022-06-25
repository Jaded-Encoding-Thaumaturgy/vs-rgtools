from __future__ import annotations

from typing import Callable, List, Sequence, TypeVar, Union

import vapoursynth as vs
from vsutil import disallow_variable_format, disallow_variable_resolution

core = vs.core

FINT = TypeVar('FINT', bound=Callable[..., vs.VideoNode])
FFLOAT = TypeVar('FFLOAT', bound=Callable[..., vs.VideoNode])
PlanesT = Union[int, Sequence[int], None]

wmean_matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]
mean_matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1]


@disallow_variable_format
@disallow_variable_resolution
def pick_rg(clip: vs.VideoNode, func_int: FINT, func_float: FFLOAT) -> FINT | FFLOAT:
    assert clip.format
    return func_float if clip.format.sample_type == vs.FLOAT else func_int


@disallow_variable_format
@disallow_variable_resolution
def minfilter(
    src: vs.VideoNode, flt1: vs.VideoNode, flt2: vs.VideoNode, planes: PlanesT = None
) -> vs.VideoNode:
    assert src.format

    return core.std.Expr(
        [src, flt1, flt2],
        norm_expr_planes(src, 'x z - abs x y - abs < z y ?', planes)
    )


@disallow_variable_format
@disallow_variable_resolution
def maxfilter(
    clip: vs.VideoNode, flt1: vs.VideoNode, flt2: vs.VideoNode, planes: PlanesT = None
) -> vs.VideoNode:
    assert clip.format

    return core.std.Expr([clip, flt1, flt2], [
        'x z - abs x y - abs > z y ?' if i in normalise_planes(clip, planes) else ''
        for i in range(clip.format.num_planes)
    ])


T = TypeVar('T', bound=Union[int, float, str])
Nb = TypeVar('Nb', float, int)


def normalise_seq(x: T | Sequence[T], length_max: int = 3) -> List[T]:
    if not isinstance(x, Sequence):
        return [x] * length_max

    x = list(x) + [x[-1]] * (length_max - len(x))
    return x[:length_max]


def clamp(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    return min_val if val < min_val else max_val if val > max_val else val


@disallow_variable_format
def normalise_planes(clip: vs.VideoNode, planes: PlanesT = None, pad: bool = True) -> List[int]:
    assert clip.format

    if planes is None:
        planes = list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]
    else:
        planes = list(planes)

    if pad:
        return normalise_seq(planes, clip.format.num_planes)

    return planes


@disallow_variable_format
def norm_expr_planes(clip: vs.VideoNode, expr: str | List[str], planes: PlanesT = None) -> List[str]:
    assert clip.format

    expr_array = normalise_seq(
        [expr] if isinstance(expr, str) else expr,
        clip.format.num_planes
    )

    planes = normalise_planes(clip, planes)

    return [exp if i in planes else '' for i, exp in enumerate(expr_array, 0)]
