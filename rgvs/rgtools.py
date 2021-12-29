from __future__ import annotations

__all__ = ['repair', 'removegrainm', 'removegrain', 'verticalcleaner', 'clense', 'backwardclense', 'forwardclense']

from typing import List, Sequence

import vapoursynth as vs
from vsutil import disallow_variable_format

from .util import eval_planes, fallback, medianclip, normalise_seq

core = vs.core


@disallow_variable_format
def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: int | Sequence[int]) -> vs.VideoNode:
    assert clip.format

    mode = normalise_seq(mode, clip.format.num_planes)

    try:
        aka_expr = core.akarin.Expr
    except AttributeError:
        pass
    else:
        expr: List[str] = []
        for m in mode:
            if m == 0:
                expr.append('')
            elif 1 <= m <= 4:
                expr.append(_aka_repair_expr_1_4(m))
            elif m == 5:
                expr.append(_aka_repair_expr_5())
            elif m == 6:
                expr.append(_aka_repair_expr_6())
            elif m == 7:
                expr.append(_aka_repair_expr_7())
            elif m == 8:
                expr.append(_aka_repair_expr_8())
            elif m == 9:
                expr.append(_aka_repair_expr_9())
            elif m == 10:
                expr.append(_aka_repair_expr_10())
            elif 11 <= m <= 14:
                expr.append(_aka_repair_expr_11_14(m - 10))
            elif m == 15:
                expr.append(_aka_repair_expr_15())
            elif m == 16:
                expr.append(_aka_repair_expr_16())
            elif m == 17:
                expr.append(_aka_repair_expr_17())
            elif m == 18:
                expr.append(_aka_repair_expr_18())
            elif m == 19:
                expr.append(_aka_repair_expr_19())
            elif m == 20:
                expr.append(_aka_repair_expr_20())
            elif m == 21:
                expr.append(_aka_repair_expr_21())
            elif m == 22:
                expr.append(_aka_repair_expr_22())
            elif m == 23:
                expr.append(_aka_repair_expr_23())
            elif m == 24:
                expr.append(_aka_repair_expr_24())
            elif m == 25:
                raise ValueError('repair: invalid mode specified')
            elif m == 26:
                expr.append(_aka_repair_expr_26())
            elif m == 27:
                expr.append(_aka_repair_expr_27())
            elif m == 28:
                expr.append(_aka_repair_expr_28())
        return aka_expr([clip, repairclip], expr).std.SetFrameProps(AkaExpr=mode)
    if (20 in mode or 23 in mode) and clip.format.sample_type == vs.FLOAT:
        raise ValueError('rgsf is wrong')
    return (core.rgsf.Repair if clip.format.sample_type == vs.FLOAT else core.rgvs.Repair)(clip, repairclip, mode)


A1 = 'y[-1,-1]'
A2 = 'y[0,-1]'
A3 = 'y[1,-1]'
A4 = 'y[-1,0]'
A5 = 'y[1,0]'
A6 = 'y[-1,1]'
A7 = 'y[0,1]'
A8 = 'y[1,1]'
c = 'y'
val = 'x'
PIXELS = ' '.join([A1, A2, A3, A4, A5, A6, A7, A8])


def _aka_repair_expr_1_4(m: int) -> str:
    return f'{PIXELS} y sort9 dup{9 - m} max_val! dup{m - 1} min_val! drop9 x min_val@ max_val@ clamp'


def _aka_repair_expr_5() -> str:
    return (
        f'x y {A1} {A8} min min y {A1} {A8} max max clamp clamp1! '
        f'x y {A2} {A7} min min y {A2} {A7} max max clamp clamp2! '
        f'x y {A3} {A6} min min y {A3} {A6} max max clamp clamp3! '
        f'x y {A4} {A5} min min y {A4} {A5} max max clamp clamp4! '
        'x clamp1@ - abs c1! '
        'x clamp2@ - abs c2! '
        'x clamp3@ - abs c3! '
        'x clamp4@ - abs c4! '
        'c1@ c2@ c3@ c4@ min min min mindiff! '
        'mindiff@ c4@ = clamp4@ mindiff@ c2@ = clamp2@ mindiff@ c3@ = clamp3@ clamp1@ ? ? ?'
    )


def _aka_repair_expr_6() -> str:
    return (
        f'y {A1} {A8} min min mil1! '
        f'y {A1} {A8} max max mal1! '
        f'y {A2} {A7} min min mil2! '
        f'y {A2} {A7} max max mal2! '
        f'y {A3} {A6} min min mil3! '
        f'y {A3} {A6} max max mal3! '
        f'y {A4} {A5} min min mil4! '
        f'y {A4} {A5} max max mal4! '
        'mal1@ mil1@ - d1! '
        'mal2@ mil2@ - d2! '
        'mal3@ mil3@ - d3! '
        'mal4@ mil4@ - d4! '
        'x mil1@ mal1@ clamp clamp1! '
        'x mil2@ mal2@ clamp clamp2! '
        'x mil3@ mal3@ clamp clamp3! '
        'x mil4@ mal4@ clamp clamp4! '
        'x clamp1@ - abs 2 * d1@ + c1! '
        'x clamp2@ - abs 2 * d2@ + c2! '
        'x clamp3@ - abs 2 * d3@ + c3! '
        'x clamp4@ - abs 2 * d4@ + c4! '
        'c1@ c2@ c3@ c4@ min min min mindiff! '
        'mindiff@ c4@ = clamp4@ mindiff@ c2@ = clamp2@ mindiff@ c3@ = clamp3@ clamp1@ ? ? ?'
    )


def _aka_repair_expr_7() -> str:
    return (
        f'y {A1} {A8} min min mil1! '
        f'y {A1} {A8} max max mal1! '
        f'y {A2} {A7} min min mil2! '
        f'y {A2} {A7} max max mal2! '
        f'y {A3} {A6} min min mil3! '
        f'y {A3} {A6} max max mal3! '
        f'y {A4} {A5} min min mil4! '
        f'y {A4} {A5} max max mal4! '
        'mal1@ mil1@ - d1! '
        'mal2@ mil2@ - d2! '
        'mal3@ mil3@ - d3! '
        'mal4@ mil4@ - d4! '
        'x mil1@ mal1@ clamp clamp1! '
        'x mil2@ mal2@ clamp clamp2! '
        'x mil3@ mal3@ clamp clamp3! '
        'x mil4@ mal4@ clamp clamp4! '
        # Only change is removing the "* 2"
        'x clamp1@ - abs d1@ + c1! '
        'x clamp2@ - abs d2@ + c2! '
        'x clamp3@ - abs d3@ + c3! '
        'x clamp4@ - abs d4@ + c4! '
        'c1@ c2@ c3@ c4@ min min min mindiff! '
        'mindiff@ c4@ = clamp4@ mindiff@ c2@ = clamp2@ mindiff@ c3@ = clamp3@ clamp1@ ? ? ?'
    )


def _aka_repair_expr_8() -> str:
    return (
        f'y {A1} {A8} min min mil1! '
        f'y {A1} {A8} max max mal1! '
        f'y {A2} {A7} min min mil2! '
        f'y {A2} {A7} max max mal2! '
        f'y {A3} {A6} min min mil3! '
        f'y {A3} {A6} max max mal3! '
        f'y {A4} {A5} min min mil4! '
        f'y {A4} {A5} max max mal4! '
        'mal1@ mil1@ - d1! '
        'mal2@ mil2@ - d2! '
        'mal3@ mil3@ - d3! '
        'mal4@ mil4@ - d4! '
        'x mil1@ mal1@ clamp clamp1! '
        'x mil2@ mal2@ clamp clamp2! '
        'x mil3@ mal3@ clamp clamp3! '
        'x mil4@ mal4@ clamp clamp4! '
        'x clamp1@ - abs d1@ 2 * + c1! '
        'x clamp2@ - abs d2@ 2 * + c2! '
        'x clamp3@ - abs d3@ 2 * + c3! '
        'x clamp4@ - abs d4@ 2 * + c4! '
        'c1@ c2@ c3@ c4@ min min min mindiff! '
        'mindiff@ c4@ = clamp4@ mindiff@ c2@ = clamp2@ mindiff@ c3@ = clamp3@ clamp1@ ? ? ?'
    )


def _aka_repair_expr_9() -> str:
    return (
        f'y {A1} {A8} min min mil1! '
        f'y {A1} {A8} max max mal1! '
        f'y {A2} {A7} min min mil2! '
        f'y {A2} {A7} max max mal2! '
        f'y {A3} {A6} min min mil3! '
        f'y {A3} {A6} max max mal3! '
        f'y {A4} {A5} min min mil4! '
        f'y {A4} {A5} max max mal4! '
        'mal1@ mil1@ - d1! '
        'mal2@ mil2@ - d2! '
        'mal3@ mil3@ - d3! '
        'mal4@ mil4@ - d4! '
        'd1@ d2@ d3@ d4@ min min min mindiff! '
        'mindiff@ d4@ = x mil4@ mal4@ clamp mindiff@ d2@ = x mil2@ mal2@ clamp mindiff@ d3@ = x mil3@ mal3@ clamp x mil1@ mal1@ clamp ? ? ?'
    )


def _aka_repair_expr_10() -> str:
    return (
        f'x {A1} - abs d1! '
        f'x {A2} - abs d2! '
        f'x {A3} - abs d3! '
        f'x {A4} - abs d4! '
        f'x {A5} - abs d5! '
        f'x {A6} - abs d6! '
        f'x {A7} - abs d7! '
        f'x {A8} - abs d8! '
        f'x y - abs dc! '
        'd1@ d2@ d3@ d4@ d5@ d6@ d7@ d8@ dc@ min min min min min min min min mindiff! '
        f'mindiff@ d7@ = {A7} mindiff@ d8@ = {A8} mindiff@ d6@ = {A6} mindiff@ d2@ = {A2} mindiff@ d3@ = {A3} mindiff@ d1@ = {A1} '
        f'mindiff@ d5@ = {A5} mindiff@ dc@ = y {A4} ? ? ? ? ? ? ? ?'
    )


def _aka_repair_expr_11_14(m: int) -> str:
    return f'{PIXELS} sort8 dup{8 - m} max_val! dup{m - 1} min_val! drop8 x y min_val@ min y max_val@ max clamp'


def _aka_repair_expr_15() -> str:
    return (
        f'{A1} {A8} min mil1! '
        f'{A1} {A8} max mal1! '
        f'{A2} {A7} min mil2! '
        f'{A2} {A7} max mal2! '
        f'{A3} {A6} min mil3! '
        f'{A3} {A6} max mal3! '
        f'{A4} {A5} min mil4! '
        f'{A4} {A5} max mal4! '
        'y mil1@ mal1@ clamp clamp1! '
        'y mil2@ mal2@ clamp clamp2! '
        'y mil3@ mal3@ clamp clamp3! '
        'y mil4@ mal4@ clamp clamp4! '
        'y clamp1@ - abs c1! '
        'y clamp2@ - abs c2! '
        'y clamp3@ - abs c3! '
        'y clamp4@ - abs c4! '
        'c1@ c2@ c3@ c4@ min min min mindiff! '
        'mindiff@ c4@ = x y mil4@ min y mal4@ max clamp '
        'mindiff@ c2@ = x y mil2@ min y mal2@ max clamp '
        'mindiff@ c3@ = x y mil3@ min y mal3@ max clamp '
        'x y mil1@ min y mal1@ max clamp ? ? ?'
    )


def _aka_repair_expr_16() -> str:
    return (
        f'{A1} {A8} min mil1! '
        f'{A1} {A8} max mal1! '
        f'{A2} {A7} min mil2! '
        f'{A2} {A7} max mal2! '
        f'{A3} {A6} min mil3! '
        f'{A3} {A6} max mal3! '
        f'{A4} {A5} min mil4! '
        f'{A4} {A5} max mal4! '
        'mal1@ mil1@ - d1! '
        'mal2@ mil2@ - d2! '
        'mal3@ mil3@ - d3! '
        'mal4@ mil4@ - d4! '
        'y y mil1@ mal1@ clamp - abs 2 * d1@ + c1! '
        'y y mil2@ mal2@ clamp - abs 2 * d2@ + c2! '
        'y y mil3@ mal3@ clamp - abs 2 * d3@ + c3! '
        'y y mil4@ mal4@ clamp - abs 2 * d4@ + c4! '
        'c1@ c2@ c3@ c4@ min min min mindiff! '
        'mindiff@ c4@ = x y mil4@ min y mal4@ max clamp '
        'mindiff@ c2@ = x y mil2@ min y mal2@ max clamp '
        'mindiff@ c3@ = x y mil3@ min y mal3@ max clamp '
        'x y mil1@ min y mal1@ max clamp ? ? ?'
    )


def _aka_repair_expr_17() -> str:
    return (
        f'{A1} {A8} min mil1! '
        f'{A1} {A8} max mal1! '
        f'{A2} {A7} min mil2! '
        f'{A2} {A7} max mal2! '
        f'{A3} {A6} min mil3! '
        f'{A3} {A6} max mal3! '
        f'{A4} {A5} min mil4! '
        f'{A4} {A5} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ min min min minmal! '
        'x y maxmil@ minmal@ min min y maxmil@ minmal@ max max clamp'
    )


def _aka_repair_expr_18() -> str:
    return (
        f'y {A1} - abs y {A8} - abs max d1! '
        f'y {A2} - abs y {A7} - abs max d2! '
        f'y {A3} - abs y {A6} - abs max d3! '
        f'y {A4} - abs y {A5} - abs max d4! '
        'd1@ d2@ d3@ d4@ min min min mindiff! '
        f'mindiff@ d4@ = x {A4} {A5} min y min {A4} {A5} max y max clamp '
        f'mindiff@ d2@ = x {A2} {A7} min y min {A2} {A7} max y max clamp '
        f'mindiff@ d3@ = x {A3} {A6} min y min {A3} {A6} max y max clamp '
        f'x {A1} {A8} min y min {A1} {A8} max y max clamp ? ? ?'
    )


def _aka_repair_expr_19() -> str:
    return (
        f'y {A1} - abs y {A2} - abs y {A3} - abs y {A4} - abs y {A5} - abs y {A6} - abs y {A7} - abs y {A8} - abs '
        'min min min min min min min mindiff! '
        'x y mindiff@ - y mindiff@ + clamp'
    )


def _aka_repair_expr_20() -> str:
    return (
        f'y {A1} - abs d1! '
        f'y {A2} - abs d2! '
        f'y {A3} - abs d3! '
        f'y {A4} - abs d4! '
        f'y {A5} - abs d5! '
        f'y {A6} - abs d6! '
        f'y {A7} - abs d7! '
        f'y {A8} - abs d8! '
        'd1@ d2@ min mindiff! '
        'd1@ d2@ max maxdiff! '
        'maxdiff@ mindiff@ d3@ clamp maxdiff! '
        'mindiff@ d3@ min mindiff! '
        'maxdiff@ mindiff@ d4@ clamp maxdiff! '
        'mindiff@ d4@ min mindiff! '
        'maxdiff@ mindiff@ d5@ clamp maxdiff! '
        'mindiff@ d5@ min mindiff! '
        'maxdiff@ mindiff@ d6@ clamp maxdiff! '
        'mindiff@ d6@ min mindiff! '
        'maxdiff@ mindiff@ d7@ clamp maxdiff! '
        'mindiff@ d7@ min mindiff! '
        'maxdiff@ mindiff@ d8@ clamp maxdiff! '
        'x y maxdiff@ - y maxdiff@ + clamp'
    )


def _aka_repair_expr_21() -> str:
    return (
        f'{A1} {A8} max y - y {A1} {A8} min - max {A2} {A7} max y - y {A2} {A7} min - max '
        f'{A3} {A6} max y - y {A3} {A6} min - max {A4} {A5} max y - y {A4} {A5} min - max min min min u! '
        'x y u@ - y u@ + clamp'
    )


def _aka_repair_expr_22() -> str:
    return (
        f'x {A1} - abs x {A2} - abs x {A3} - abs x {A4} - abs x {A5} - abs x {A6} - abs x {A7} - abs x {A8} - abs '
        'min min min min min min min mindiff! '
        'y x mindiff@ - x mindiff@ + clamp'
    )


def _aka_repair_expr_23() -> str:
    return (
        f'x {A1} - abs d1! '
        f'x {A2} - abs d2! '
        f'x {A3} - abs d3! '
        f'x {A4} - abs d4! '
        f'x {A5} - abs d5! '
        f'x {A6} - abs d6! '
        f'x {A7} - abs d7! '
        f'x {A8} - abs d8! '
        'd1@ d2@ min mindiff! '
        'd1@ d2@ max maxdiff! '
        'maxdiff@ mindiff@ d3@ clamp maxdiff! '
        'mindiff@ d3@ min mindiff! '
        'maxdiff@ mindiff@ d4@ clamp maxdiff! '
        'mindiff@ d4@ min mindiff! '
        'maxdiff@ mindiff@ d5@ clamp maxdiff! '
        'mindiff@ d5@ min mindiff! '
        'maxdiff@ mindiff@ d6@ clamp maxdiff! '
        'mindiff@ d6@ min mindiff! '
        'maxdiff@ mindiff@ d7@ clamp maxdiff! '
        'mindiff@ d7@ min mindiff! '
        'maxdiff@ mindiff@ d8@ clamp maxdiff! '
        'y x maxdiff@ - x maxdiff@ + clamp'
    )


def _aka_repair_expr_24() -> str:
    return (
        f'{A1} {A8} max x - x {A1} {A8} min - max {A2} {A7} max x - x {A2} {A7} min - max '
        f'{A3} {A6} max x - x {A3} {A6} min - max {A4} {A5} max x - x {A4} {A5} min - max min min min u! '
        'y x u@ - x u@ + clamp'
    )


def _aka_repair_expr_26() -> str:
    return (
        f'{A1} {A2} min mil1! '
        f'{A1} {A2} max mal1! '
        f'{A2} {A3} min mil2! '
        f'{A2} {A3} max mal2! '
        f'{A3} {A5} min mil3! '
        f'{A3} {A5} max mal3! '
        f'{A5} {A8} min mil4! '
        f'{A5} {A8} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ min min min minmal! '
        f'{A7} {A8} min mil1! '
        f'{A7} {A8} max mal1! '
        f'{A6} {A7} min mil2! '
        f'{A6} {A7} max mal2! '
        f'{A4} {A6} min mil3! '
        f'{A4} {A6} max mal3! '
        f'{A1} {A4} min mil4! '
        f'{A1} {A4} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ maxmil@ max max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ minmal@ min min min min minmal! '
        'x y maxmil@ minmal@ min min y maxmil@ minmal@ max max clamp'
    )


def _aka_repair_expr_27() -> str:
    return (
        f'{A1} {A8} min mil1! '
        f'{A1} {A8} max mal1! '
        f'{A1} {A2} min mil2! '
        f'{A1} {A2} max mal2! '
        f'{A7} {A8} min mil3! '
        f'{A7} {A8} max mal3! '
        f'{A2} {A7} min mil4! '
        f'{A2} {A7} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ min min min minmal! '
        f'{A2} {A3} min mil1! '
        f'{A2} {A3} max mal1! '
        f'{A6} {A7} min mil2! '
        f'{A6} {A7} max mal2! '
        f'{A3} {A6} min mil3! '
        f'{A3} {A6} max mal3! '
        f'{A3} {A5} min mil4! '
        f'{A3} {A5} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ maxmil@ max max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ minmal@ min min min min minmal! '
        f'{A4} {A6} min mil1! '
        f'{A4} {A6} max mal1! '
        f'{A4} {A5} min mil2! '
        f'{A4} {A5} max mal2! '
        f'{A5} {A8} min mil3! '
        f'{A5} {A8} max mal3! '
        f'{A1} {A4} min mil4! '
        f'{A1} {A4} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ maxmil@ max max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ minmal@ min min min min minmal! '
        'x y maxmil@ minmal@ min min y maxmil@ minmal@ max max clamp'
    )


def _aka_repair_expr_28() -> str:
    return (
        f'{A1} {A2} min mil1! '
        f'{A1} {A2} max mal1! '
        f'{A2} {A3} min mil2! '
        f'{A2} {A3} max mal2! '
        f'{A3} {A5} min mil3! '
        f'{A3} {A5} max mal3! '
        f'{A5} {A8} min mil4! '
        f'{A5} {A8} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ min min min minmal! '
        f'{A7} {A8} min mil1! '
        f'{A7} {A8} max mal1! '
        f'{A6} {A7} min mil2! '
        f'{A6} {A7} max mal2! '
        f'{A4} {A6} min mil3! '
        f'{A4} {A6} max mal3! '
        f'{A1} {A5} min mil4! '
        f'{A1} {A5} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ maxmil@ max max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ minmal@ min min min min minmal! '
        f'{A1} {A8} min mil1! '
        f'{A1} {A8} max mal1! '
        f'{A3} {A6} min mil2! '
        f'{A3} {A6} max mal2! '
        f'{A2} {A7} min mil3! '
        f'{A2} {A7} max mal3! '
        f'{A4} {A5} min mil4! '
        f'{A4} {A5} max mal4! '
        'mil1@ mil2@ mil3@ mil4@ maxmil@ max max max max maxmil! '
        'mal1@ mal2@ mal3@ mal4@ minmal@ min min min min minmal! '
        'x y maxmil@ minmal@ min min y maxmil@ minmal@ max max clamp'
    )


def removegrain(clip, mode=2, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('removegrain: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    mode = append_params(mode, numplanes)

    planes = parse_planes(planes, numplanes, 'removegrain')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]

########################################################################
########################################################################
### Mode 4 - Use std.median()
    def _rg4(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.median(planes=planes)
    planes_rg4  = []
    for i in range(numplanes):
        if mode[i] == 4:
            planes_rg4 += [i]
            mode[i] = 0
    clip = _rg4(clip, planes_rg4)
########################################################################
########################################################################


########################################################################
########################################################################
### Mode 11/12 - Use std.Convolution([1,2,1,2,4,2,1,2,1])
    def _rg11(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.Convolution([1,2,1,2,4,2,1,2,1], planes=planes)
    planes_rg11 = []
    for i in range(numplanes):
        if mode[i] in (11, 12):
            planes_rg11 += [i]
            mode[i] = 0
    clip = _rg11(clip, planes_rg11)
########################################################################
########################################################################


########################################################################
########################################################################
### Mode 19 - Use std.Convolution([1,1,1,1,0,1,1,1,1])
    def _rg19(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.Convolution([1,1,1,1,0,1,1,1,1], planes=planes)
    planes_rg19 = []
    for i in range(numplanes):
        if mode[i] == 19:
            planes_rg19 += [i]
            mode[i] = 0
    clip = _rg19(clip, planes_rg19)
########################################################################
########################################################################


########################################################################
########################################################################
### Mode 20 - Use std.Convolution([1,1,1,1,1,1,1,1,1])
    def _rg20(clip, planes):
        if len(planes)==0:
            return clip
        return clip.std.Convolution([1]*9, planes=planes)
    planes_rg20 = []
    for i in range(numplanes):
        if mode[i] == 20:
            planes_rg20 += [i]
            mode[i] = 0
    clip = _rg20(clip, planes_rg20)
########################################################################
########################################################################

    if max(mode) == 0:
        return clip
    if isflt:
        return clip.rgsf.removegrain(mode)
    return clip.rgvs.removegrain(mode)


def removegrainm(clip, mode=2, modeu=None, modev=None, iter=None, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('removegrainm: This is not a clip')

    numplanes = clip.format.num_planes
    planes = parse_planes(planes, numplanes, 'removegrainm')

    mode = [mode] if isinstance(mode, int) else mode
    modeu = [0] if numplanes < 2 else fallback([modeu] if isinstance(modeu, int) else modeu, mode )
    modev = [0] if numplanes < 3 else fallback([modev] if isinstance(modev, int) else modev, modeu)

    iter = fallback(iter, [len(x) for x in (mode,modeu,modev)])
    iter = append_params(iter, 3)
    iter = [iter[x] if x in planes else 0 for x in range(3)]

    mode  = [0] if iter[0] is 0 else append_params(mode , iter[0])
    modeu = [0] if iter[1] is 0 else append_params(modeu, iter[1])
    modev = [0] if iter[2] is 0 else append_params(modev, iter[2])

    iterall = max(iter[:numplanes])

    while len(mode ) < iterall:
        mode .append(0)
    while len(modeu) < iterall:
        modeu.append(0)
    while len(modev) < iterall:
        modev.append(0)

    for x in range(iterall):
        clip = removegrain(clip, [mode[x], modeu[x], modev[x]])

    return clip


def clense(clip, previous=None, next=None, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('clense: This is not a clip')
    if next is not None:
        if not isinstance(next, vs.VideoNode):
            raise TypeError('clense: This is not a clip')
    if previous is not None:
        if not isinstance(previous, vs.VideoNode):
            raise TypeError('clense: This is not a clip')

    planes = parse_planes(planes, clip.format.num_planes, 'clense')
    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)
    if len(planes) == 0:
        return clip

    if previous == next == None:
        return clip.tmedian.TemporalMedian(1, planes)

    previous = fallback(previous, clip)
    previous = previous[0] + previous[:-1]

    next = fallback(next, clip)
    next = next[1:] + next[-1]

    return medianclip([previous, clip, next], planes=planes)


def forwardclense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('forwardclense: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    planes = parse_planes(planes, clip.format.num_planes, 'forwardclense')

    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)

    if len(planes) == 0:
        return clip
    if isflt:
        return clip.rgsf.forwardclense(planes)
    return clip.rgvs.forwardclense(planes)


def backwardclense(clip, planes=None, grey=False):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('backwardclense: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    planes = parse_planes(planes, clip.format.num_planes, 'backwardclense')

    if grey:
        planes = eval_planes(planes, [1] + [0] * (numplanes - 1), 0)

    if len(planes) == 0:
        return clip
    if isflt:
        return clip.rgsf.backwardclense(planes)
    return clip.rgvs.backwardclense(planes)


def verticalcleaner(clip, mode=1, planes=None):
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('verticalcleaner: This is not a clip')

    fmt = clip.format
    isflt = fmt.sample_type
    numplanes = fmt.num_planes

    mode = append_params(mode, numplanes)

    planes = parse_planes(planes, numplanes, 'verticalcleaner')
    mode = [max(mode[x], 0) if x in planes else 0 for x in range(numplanes)]

    if max(mode) == 0:
        return clip
    if isflt:
        return clip.rgsf.verticalcleaner(mode)
    return clip.rgvs.verticalcleaner(mode)
