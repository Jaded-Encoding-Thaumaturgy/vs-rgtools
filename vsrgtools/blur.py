from __future__ import annotations

from functools import partial
from itertools import count
from math import ceil, exp, log, pi, sqrt

from vsexprtools import EXPR_VARS, aka_expr_available, norm_expr
from vstools import (
    ConvMode, CustomNotImplementedError, CustomOverflowError, CustomValueError, FuncExceptT, NotFoundEnumValue, PlanesT,
    core, depth, disallow_variable_format, disallow_variable_resolution, get_depth, get_neutral_value, join,
    normalize_planes, split, vs, check_variable
)

from .enum import LimitFilterMode
from .limit import limit_filter
from .util import normalize_radius

__all__ = [
    'blur', 'box_blur', 'side_box_blur',
    'gauss_blur', 'gauss_fmtc_blur',
    'min_blur', 'sbr'
]


@disallow_variable_format
@disallow_variable_resolution
def blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, blur, radius, planes, mode=mode)

    if mode == ConvMode.SQUARE:
        matrix2 = [1, 3, 4, 3, 1]
        matrix3 = [1, 4, 8, 10, 8, 4, 1]
    elif mode in {ConvMode.HORIZONTAL, ConvMode.VERTICAL}:
        matrix2 = [1, 6, 15, 20, 15, 6, 1]
        matrix3 = [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
    else:
        raise NotFoundEnumValue('Invalid mode specified!', blur)

    if radius == 1:
        matrix = [1, 2, 1]
    elif radius == 2:
        matrix = matrix2
    elif radius == 3:
        matrix = matrix3
    else:
        raise CustomNotImplementedError('This radius isn\'t supported!', blur)

    return clip.std.Convolution(matrix, planes=planes, mode=mode)


@disallow_variable_format
@disallow_variable_resolution
def box_blur(clip: vs.VideoNode, radius: int | list[int] = 1, passes: int = 1, planes: PlanesT = None) -> vs.VideoNode:
    assert check_variable(clip, box_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, box_blur, radius, planes, passes=passes)

    if radius > 12:
        blurred = clip.std.BoxBlur(planes, radius, passes, radius, passes)
    else:
        matrix_size = radius * 2 | 1
        blurred = clip
        for _ in range(passes):
            blurred = blurred.std.Convolution(
                [1] * matrix_size, planes=planes, mode=ConvMode.SQUARE
            )

    return blurred


@disallow_variable_format
@disallow_variable_resolution
def side_box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None,
    inverse: bool = False, expr: bool | None = None
) -> vs.VideoNode:
    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, side_box_blur, radius, planes, inverse=inverse, expr=expr)

    half_kernel = [(1 if i <= 0 else 0) for i in range(-radius, radius + 1)]

    conv_m1 = partial(core.std.Convolution, matrix=half_kernel, planes=planes)
    conv_m2 = partial(core.std.Convolution, matrix=half_kernel[::-1], planes=planes)
    blur_pt = partial(core.std.BoxBlur, planes=planes)

    vrt_filters, hrz_filters = [
        [
            partial(conv_m1, mode=mode), partial(conv_m2, mode=mode),
            partial(blur_pt, hradius=hr, vradius=vr, hpasses=h, vpasses=v)
        ] for h, hr, v, vr, mode in [
            (0, None, 1, radius, ConvMode.VERTICAL), (1, radius, 0, None, ConvMode.HORIZONTAL)
        ]
    ]

    vrt_intermediates = (vrt_flt(clip) for vrt_flt in vrt_filters)
    intermediates = list(
        hrz_flt(vrt_intermediate)
        for i, vrt_intermediate in enumerate(vrt_intermediates)
        for j, hrz_flt in enumerate(hrz_filters) if not i == j == 2
    )

    comp_blur = None if inverse else box_blur(clip, radius, 1, planes)

    if aka_expr_available if expr is None else expr:
        template = '{cum} x - abs {new} x - abs < {cum} {new} ?'

        cum_expr, cumc = '', 'y'
        n_inter = len(intermediates)

        for i, newc, var in zip(count(), EXPR_VARS[2:], EXPR_VARS[4:]):
            if i == n_inter - 1:
                break

            cum_expr += template.format(cum=cumc, new=newc)

            if i != n_inter - 2:
                cumc = var.upper()
                cum_expr += f' {cumc}! '
                cumc = f'{cumc}@'

        if comp_blur:
            clips = [clip, *intermediates, comp_blur]
            cum_expr = f'x {cum_expr} - {EXPR_VARS[n_inter + 1]} +'
        else:
            clips = [clip, *intermediates]

        cum = norm_expr(clips, cum_expr, planes, force_akarin='vsrgtools.side_box_blur')
    else:
        cum = intermediates[0]
        for new in intermediates[1:]:
            cum = limit_filter(clip, cum, new, LimitFilterMode.SIMPLE2_MIN, planes)

        if comp_blur:
            cum = clip.std.MakeDiff(cum).std.MergeDiff(comp_blur)

    if comp_blur:
        return box_blur(cum, 1, min(radius // 2, 1))

    return cum


def _norm_gauss_sigma(clip: vs.VideoNode, sigma: float | None, sharp: float | None, func: FuncExceptT) -> float:
    if sigma is None:
        if sharp is None:
            raise CustomValueError("Sigma and sharp can't be both None!", func)
        sigma = sqrt(1.0 / (2.0 * (sharp / 10.0) * log(2)))
    elif sharp is not None:
        raise CustomValueError("Sigma and sharp can't both be float!", func)

    if sigma >= min(clip.width, clip.height):
        raise CustomOverflowError("Sigma can't be bigger or equal than the smaller size of the clip!", func)

    return sigma


@disallow_variable_format
@disallow_variable_resolution
def gauss_blur(
    clip: vs.VideoNode,
    sigma: float | list[float] | None = 0.5, sharp: float | list[float] | None = None,
    mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, gauss_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(sigma, list):
        return normalize_radius(clip, gauss_blur, ('sigma', sigma), planes, mode=mode)
    elif isinstance(sharp, list):
        return normalize_radius(clip, gauss_blur, ('sharp', sharp), planes, mode=mode)

    sigma = _norm_gauss_sigma(clip, sigma, sharp, gauss_blur)

    if sigma <= 0.333:
        return clip

    if sigma <= 4.333:
        taps = ceil(sigma * 6 + 1)

        if not taps % 2:
            taps += 1

        half_pisqrt = 1.0 / sqrt(2.0 * pi) * sigma
        doub_qsigma = 2 * sigma * sigma

        high, *kernel = [
            half_pisqrt * exp(-(x * x) / doub_qsigma)
            for x in range(taps // 2)
        ]

        kernel = [x * 1023 / high for x in kernel]
        kernel = [*kernel[::-1], 1023, *kernel]

        return clip.std.Convolution(kernel, mode=mode, planes=planes)

    return gauss_fmtc_blur(clip, sigma, sharp, True, mode, planes)


@disallow_variable_format
@disallow_variable_resolution
def gauss_fmtc_blur(
    clip: vs.VideoNode,
    sigma: float | list[float] | None = 0.5, sharp: float | list[float] | None = None,
    strict: bool = True, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, gauss_fmtc_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(sigma, list):
        return normalize_radius(clip, gauss_blur, ('sigma', sigma), planes, strict=strict, mode=mode)
    elif isinstance(sharp, list):
        return normalize_radius(clip, gauss_blur, ('sharp', sharp), planes, strict=strict, mode=mode)

    if strict:
        sigma = _norm_gauss_sigma(clip, sigma, sharp, gauss_fmtc_blur)
        wdown, hdown = [
            round((max(round(size / sigma), 2) if char in mode else size) / 2) * 2
            for size, char in [(clip.width, 'h'), (clip.height, 'v')]
        ]

        def _fmtc_blur(clip: vs.VideoNode) -> vs.VideoNode:
            down = clip.resize.Bilinear(wdown, hdown)
            return down.fmtc.resample(clip.width, clip.height, kernel='gauss', a1=9)
    else:
        if sigma is None or sigma < 1.0 or sigma > 100.0:
            raise CustomValueError('Sigma has to be > 1 and < 100!', gauss_fmtc_blur, reason='strict=True')
        elif sharp is None:
            sharp = 100
        elif sharp < 1.0 or sharp > 100.0:
            raise CustomValueError('Sharp has to be > 1 and < 100!', gauss_fmtc_blur, reason='strict=True')

        def _fmtc_blur(clip: vs.VideoNode) -> vs.VideoNode:
            down = clip.fmtc.resample(
                clip.width * 2, clip.height * 2, kernel='gauss', a1=sharp
            )
            return down.fmtc.resample(clip.width, clip.height, kernel='gauss', a1=sigma)

    if not {*range(clip.format.num_planes)} - {*planes}:
        blurred = _fmtc_blur(clip)
    else:
        blurred = join([
            _fmtc_blur(p) if i in planes else p for i, p in enumerate(split(clip))
        ])

    if (bits := get_depth(clip)) != get_depth(blurred):
        return depth(blurred, bits)

    return blurred


@disallow_variable_format
@disallow_variable_resolution
def min_blur(clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None) -> vs.VideoNode:
    """
    MinBlur by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert check_variable(clip, min_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, min_blur, radius, planes)

    median = clip.std.Median(planes) if radius in {0, 1} else clip.ctmf.CTMF(radius, None, planes)

    if radius:
        weighted = blur(clip, radius)
    else:
        weighted = sbr(clip, planes=planes)

    return limit_filter(weighted, clip, median, LimitFilterMode.DIFF_MIN, planes)


@disallow_variable_format
@disallow_variable_resolution
def sbr(
    clip: vs.VideoNode,
    radius: int | list[int] = 1, mode: ConvMode = ConvMode.SQUARE,
    planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, sbr)

    planes = normalize_planes(clip, planes)

    neutral = [get_neutral_value(clip), get_neutral_value(clip, True)]

    blur_func = partial(blur, radius=radius, mode=mode, planes=planes)

    weighted = blur_func(clip)

    diff = clip.std.MakeDiff(weighted, planes)

    diff_weighted = blur_func(diff)

    clips = [diff, diff_weighted]

    if aka_expr_available:
        clips.append(clip)
        expr = 'x y - A! x {mid} - XD! z A@ XD@ * 0 < {mid} A@ abs XD@ abs < A@ {mid} + x ? ? {mid} - -'
    else:
        expr = 'x y - x {mid} - * 0 < {mid} x y - abs x {mid} - abs < x y - {mid} + x ? ?'

    diff = norm_expr(clips, expr, planes, mid=neutral)

    if aka_expr_available:
        return diff

    return clip.std.MakeDiff(diff, planes)
