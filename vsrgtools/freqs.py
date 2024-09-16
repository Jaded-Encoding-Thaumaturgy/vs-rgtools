from __future__ import annotations

from itertools import count
from math import e, log, pi, sin, sqrt
from typing import Iterable

from vsexprtools import ExprOp, ExprVars, combine, norm_expr
from vstools import (
    ColorRange, ConvMode, CustomIntEnum, CustomNotImplementedError, FuncExceptT, FunctionUtil, KwargsT, PlanesT,
    StrList, check_ref_clip, flatten_vnodes, get_y, scale_value, vs
)

from .blur import gauss_blur

__all__ = [
    'replace_low_frequencies',

    'MeanMode'
]


def replace_low_frequencies(
    flt: vs.VideoNode, ref: vs.VideoNode, LFR: float, DCTFlicker: bool = False,
    planes: PlanesT = None, mode: ConvMode = ConvMode.HV
) -> vs.VideoNode:
    func = FunctionUtil(flt, replace_low_frequencies, planes, (vs.YUV, vs.GRAY))

    check_ref_clip(flt, ref, func.func)

    ref_work_clip = get_y(ref) if func.luma_only else ref

    LFR = max(LFR or (300 * func.work_clip.width / 1920), 50.0)

    # Frequency sample rate is resolution * 2 (for Nyquist)
    freq_sample = max(func.work_clip.width, func.work_clip.height) * 2

    k = sqrt(log(2) / 2) * LFR          # Constant for -3dB
    f_cut = freq_sample / (k * 2 * pi)  # Frequency Cutoff for Gaussian Sigma

    expr = 'x y - z + '

    if DCTFlicker:
        sec = scale_value(sin(e) + 0.1, 8, func.work_clip, range_out=ColorRange.FULL)
        expr += f'y z - D! y z = swap dup D@ 0 = 0 D@ 0 < -1 1 ? ? {sec} * + ?'

    flt_blur = gauss_blur(func.work_clip, f_cut, None, mode)
    ref_blur = gauss_blur(ref_work_clip, f_cut, None, mode)

    final = norm_expr([func.work_clip, flt_blur, ref_blur], expr, planes, force_akarin=func.func)

    return func.return_clip(final)


class MeanMode(CustomIntEnum):
    MINIMUM = -2
    HARMONIC = -1
    GEOMETRIC = 0

    ARITHMETIC = 1

    RMS = 2
    CUBIC = 3
    MAXIMUM = 4

    LEHMER = 10

    MINIMUM_ABS = 20
    MAXIMUM_ABS = 21

    MEDIAN = 30

    def __call__(
        self, *_clips: vs.VideoNode | Iterable[vs.VideoNode], planes: PlanesT = None, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        n_clips = len(clips)
        n_op = n_clips - 1

        if n_clips < 2:
            return next(iter(clips))

        kwargs = KwargsT(planes=planes, func=func)

        if self == MeanMode.MINIMUM:
            return ExprOp.MIN(clips, **kwargs)

        if self == MeanMode.MAXIMUM:
            return ExprOp.MAX(clips, **kwargs)

        if self == MeanMode.GEOMETRIC:
            return combine(clips, ExprOp.MUL, None, None, [1 / n_clips, ExprOp.POW], **kwargs)

        if self == MeanMode.LEHMER:
            counts = range(n_clips)
            clip_vars = ExprVars(n_clips)

            expr = StrList([[f'{clip} range_diff - D{i}!' for i, clip in zip(counts, clip_vars)]])

            for y in range(2):
                expr.extend([
                    [f'D{i}@ {3 - y} pow' for i in counts],
                    ExprOp.ADD * n_op, f'P{y + 1}!'
                ])

            expr.append('P2@ 0 = 0 P1@ P2@ / ? range_diff +')

            return norm_expr(clips, expr, **kwargs)

        if self in {MeanMode.RMS, MeanMode.ARITHMETIC, MeanMode.CUBIC, MeanMode.HARMONIC}:
            return combine(
                clips, ExprOp.ADD, f'{self.value} {ExprOp.POW}', None, [
                    n_clips, ExprOp.DIV, 1 / self, ExprOp.POW
                ], **kwargs
            )

        if self in {MeanMode.MINIMUM_ABS, MeanMode.MAXIMUM_ABS}:
            operator = ExprOp.MIN if self is MeanMode.MINIMUM_ABS else ExprOp.MAX

            expr_string = ''
            for src in ExprVars(n_clips):
                expr_string += f'{src} range_diff - abs {src.upper()}D! '

            for i, src, srcn in zip(count(), ExprVars(n_clips), ExprVars(1, n_clips)):
                expr_string += f'{src.upper()}D@ {srcn.upper()}D@ {operator} {src} '

                if i == n_clips - 2:
                    expr_string += f'{srcn} '

            expr_string += '? ' * n_op

            return norm_expr(clips, expr_string, **kwargs)

        if self == MeanMode.MEDIAN:
            all_clips = str(ExprVars(1, n_clips))

            n_ops = n_clips - 2

            yzmin, yzmax = [
                all_clips + f' {op}' * n_ops for op in (ExprOp.MIN, ExprOp.MAX)
            ]

            expr = f'{yzmin} YZMIN! {yzmax} YZMAX! x YZMIN@ min x = YZMIN@ x YZMAX@ max x = YZMAX@ x ? ?'

            return norm_expr(clips, expr, planes)

        raise CustomNotImplementedError
