from __future__ import annotations

__all__ = ['minfilter', 'maxfilter']

from typing import List, Sequence, TypeVar, cast, Callable
from vsutil import disallow_variable_format, get_depth

import vapoursynth as vs

core = vs.core

FINT = TypeVar('FINT', bound=Callable[..., vs.VideoNode])
FFLOAT = TypeVar('FFLOAT', bound=Callable[..., vs.VideoNode])


def pick_rg(clip: vs.VideoNode, func_int: FINT, func_float: FFLOAT) -> FINT | FFLOAT:
    assert clip.format
    return func_float if clip.format.sample_type == vs.FLOAT else func_int


def minfilter(
    src: vs.VideoNode,
    flt1: vs.VideoNode, flt2: vs.VideoNode,
    planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    assert src.format

    return core.std.Expr(
        [src, flt1, flt2],
        ['x z - abs x y - abs < z y ?' if i in normalise_planes(planes) else '' for i in range(src.format.num_planes)]
    )


def maxfilter(
    src: vs.VideoNode,
    flt1: vs.VideoNode, flt2: vs.VideoNode,
    planes: int | Sequence[int] | None = None
) -> vs.VideoNode:
    assert src.format

    return core.std.Expr(
        [src, flt1, flt2],
        ['x z - abs x y - abs > z y ?' if i in normalise_planes(planes) else '' for i in range(src.format.num_planes)]
    )


T = TypeVar('T')
Nb = TypeVar('Nb', float, int)


def normalise_seq(x: T | Sequence[T], length_max: int = 3) -> List[T]:
    if not isinstance(x, Sequence):
        return [x] * length_max
    x = cast(Sequence[T], x)
    return list(x) + [x[-1]] * (length_max - len(x))


def clamp(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    return min_val if val < min_val else max_val if val > max_val else val


def normalise_planes(clip: vs.VideoNode, planes: int | Sequence[int] | None) -> List[int]:
    assert clip.format

    if planes is None:
        planes = list(range(3))
    else:
        planes = normalise_seq(planes, clip.format.num_planes)

    return planes


# Waiting for vsutil 0.6.1 zzzzz

@disallow_variable_format
def get_lowest_value(clip: vs.VideoNode, chroma: bool = False) -> float:
    is_float = clip.format.sample_type == vs.FLOAT

    return -0.5 if chroma and is_float else 0.


@disallow_variable_format
def get_neutral_value(clip: vs.VideoNode, chroma: bool = False) -> float:
    is_float = clip.format.sample_type == vs.FLOAT

    return (0. if chroma else 0.5) if is_float else float(1 << (get_depth(clip) - 1))


@disallow_variable_format
def get_peak_value(clip: vs.VideoNode, chroma: bool = False) -> float:
    is_float = clip.format.sample_type == vs.FLOAT

    return (0.5 if chroma else 1.) if is_float else (1 << get_depth(clip)) - 1.
