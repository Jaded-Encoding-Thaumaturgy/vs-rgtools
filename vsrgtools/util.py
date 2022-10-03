from __future__ import annotations

from typing import Sequence, TypeVar, cast

from vstools import PlanesT, normalize_planes, normalize_seq, to_arr, vs

from .enum import RemoveGrainMode, RepairMode

__all__ = [
    'wmean_matrix', 'mean_matrix',
    'norm_rmode_planes'
]

wmean_matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]
mean_matrix = [1, 1, 1, 1, 1, 1, 1, 1, 1]

RModeT = TypeVar('RModeT', RemoveGrainMode, RepairMode)


def norm_rmode_planes(
    clip: vs.VideoNode, mode: int | RModeT | Sequence[int | RModeT], planes: PlanesT = None
) -> list[int]:
    assert clip.format

    modes_array = normalize_seq(to_arr(mode), clip.format.num_planes)  # type: ignore[var-annotated,arg-type]

    planes = normalize_planes(clip, planes)

    return [
        cast(RModeT, rep if i in planes else 0) for i, rep in enumerate(modes_array, 0)
    ]
