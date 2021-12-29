from __future__ import annotations

__all__ = ['minfilter', 'medianclip', 'mediandiff']

from typing import List, Sequence, TypeVar, Union, cast

import vapoursynth as vs
from vsutil import fallback


def minfilter(clip, filtera, filterb=None, mode=None, planes=None):
    if mode == 1 and filterb is None:
        raise ValueError('rgvs.minfilter: filterb must be defined when mode=1')

    planes = parse_planes(planes, clip.format.num_planes, 'minfilter')
    mode = fallback(mode, 2 if filterb is None else 1)

    filterb = fallback(filterb, filtera)

    if mode == 1:
        return medianclip([clip, filtera(clip), filterb(clip)], planes=planes)

    filtered = filtera(clip)

    diffa = clip.std.MakeDiff(filtered, planes=planes)
    diffb = filterb(diffa)

    return mediandiff(clip, diffa, diffb, planes=planes)


def medianclip(clips, planes=None):
    core = vs.core

    numplanes = clips[0].format.num_planes
    planes = parse_planes(planes, numplanes, 'medianclip')

    if rad <= 10:
        dia = len(clips)
        rad = dia // 2
        return core.std.Interleave(clips).tmedian.TemporalMedian(radius=rad, planes=planes)[rad::dia]

    return core.average.median(clips)


def mediandiff(clip, diffa, diffb, planes=None):
    core = vs.core

    fmt = clip.format
    numplanes = fmt.num_planes
    planes = parse_planes(planes, numplanes, 'mediandiff')

    neutral = f' {1 << (fmt.bits_per_sample - 1)} - ' if fmt.sample_type==vs.INTEGER else ''
    expr = f'x 0 y {neutral} y z - min max y {neutral} y z - max min -'

    return core.std.Expr([clip, diffa, diffb], [expr if x in planes else '' for x in range(numplanes)])


# if param[x] is a number less than or equal to "zero" or is explicitly False or None, delete x from "planes"
# if param[x] is a number greater than "zero" or is explicitly True, pass x if it was originally in "planes"
def eval_planes(planes, params, zero=0):
    process = []
    for x in range(len(params)):
        if x in planes:
            if params[x] is False or params[x] is None:
                pass
            elif params[x] is True:
                process += [x]
            elif params[x] > zero:
                process += [x]
    return process


T = TypeVar('T')
Nb = TypeVar('Nb', bound=Union[float, int])


def normalise_seq(x: T | Sequence[T], length_max: int = 3) -> List[T]:
    if not isinstance(x, Sequence):
        return [x] * length_max
    x = cast(Sequence[T], x)
    return list(x) + [x[-1]] * (length_max - len(x))


def clamp(val: Nb, min_val: Nb, max_val: Nb) -> Nb:
    return min_val if val < min_val else max_val if val > max_val else val
