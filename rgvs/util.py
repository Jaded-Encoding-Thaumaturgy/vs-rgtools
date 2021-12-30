from __future__ import annotations

__all__ = ['minfilter', 'maxfilter']

from typing import List, Sequence, TypeVar, Union, cast

import vapoursynth as vs

core = vs.core


def minfilter(src: vs.VideoNode, flt1: vs.VideoNode, flt2: vs.VideoNode, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    assert src.format
    if planes is not None:
        planes = normalise_seq(planes)
    else:
        planes = list(range(3))
    return core.std.Expr(
        [src, flt1, flt2],
        ['x z - abs x y - abs < z y ?' if i in planes else '' for i in range(src.format.num_planes)]
    )


def maxfilter(src: vs.VideoNode, flt1: vs.VideoNode, flt2: vs.VideoNode, planes: int | Sequence[int] | None = None) -> vs.VideoNode:
    assert src.format
    if planes is not None:
        planes = normalise_seq(planes)
    else:
        planes = list(range(3))
    return core.std.Expr(
        [src, flt1, flt2],
        ['x z - abs x y - abs > z y ?' if i in planes else '' for i in range(src.format.num_planes)]
    )


T = TypeVar('T')
Nb = TypeVar('Nb', bound=Union[float, int])


def normalise_seq(x: T | Sequence[T], length_max: int = 3) -> List[T]:
    if not isinstance(x, Sequence):
        return [x] * length_max
    x = cast(Sequence[T], x)
    return list(x) + [x[-1]] * (length_max - len(x))


def clamp(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    return min_val if val < min_val else max_val if val > max_val else val
