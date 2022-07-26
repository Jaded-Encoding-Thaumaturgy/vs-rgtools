from __future__ import annotations

from functools import partial
from typing import Any, Callable, TypeVar

import vapoursynth as vs
from vsutil import disallow_variable_format, disallow_variable_resolution

core = vs.core

FINT = TypeVar('FINT', bound=Callable[..., vs.VideoNode])
FFLOAT = TypeVar('FFLOAT', bound=Callable[..., vs.VideoNode])

wmean_matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]
mean_matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1]


@disallow_variable_format
@disallow_variable_resolution
def pick_func_stype(clip: vs.VideoNode, func_int: FINT, func_float: FFLOAT) -> FINT | FFLOAT:
    assert clip.format
    return func_float if clip.format.sample_type == vs.FLOAT else func_int


IT = TypeVar('IT')
IR = TypeVar('IR')


def iterate(base: IT, function: Callable[[IT | IR], IT | IR], count: int, *args: Any, **kwargs: Any) -> IT | IR:
    if count <= 0:
        return base

    func = partial(function, **kwargs)

    def _iterate(x: IT | IR, n: int) -> IT | IR:
        return n and _iterate(func(x, *args), n - 1) or x

    return _iterate(base, count)
