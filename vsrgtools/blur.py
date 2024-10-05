from __future__ import annotations

from functools import lru_cache, partial
from itertools import count
from math import e, log2
from typing import Any

from vsexprtools import ExprOp, ExprVars, complexpr_available, norm_expr
from vskernels import Gaussian
from vspyplugin import FilterMode, ProcessMode, PyPluginCuda
from vstools import (
    ConvMode, CustomNotImplementedError, CustomRuntimeError, FunctionUtil, NotFoundEnumValue, PlanesT, StrList,
    check_variable, core, depth, fallback, get_depth, get_neutral_values, join, normalize_planes, normalize_seq, split,
    to_arr, vs
)

from .enum import BlurMatrix, LimitFilterMode
from .limit import limit_filter
from .util import normalize_radius

__all__ = [
    'blur', 'box_blur', 'side_box_blur',
    'gauss_blur',
    'min_blur', 'sbr', 'median_blur',
    'bilateral'
]


def blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, mode: ConvMode = ConvMode.HV, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, blur, radius, planes, mode=mode)

    if mode == ConvMode.HV:
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


def box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, passes: int = 1,
    mode: ConvMode = ConvMode.HV, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, box_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, box_blur, radius, planes, passes=passes)

    if not radius:
        return clip

    box_args = (
        planes,
        radius, 0 if mode == ConvMode.VERTICAL else passes,
        radius, 0 if mode == ConvMode.HORIZONTAL else passes
    )

    if hasattr(core, 'vszip'):
        blurred = clip.vszip.BoxBlur(*box_args)
    else:
        fp16 = clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample == 16

        if radius > 12 and not fp16:
            blurred = clip.std.BoxBlur(*box_args)
        else:
            matrix_size = radius * 2 | 1

            if fp16:
                matrix_size **= 2

            blurred = clip
            for _ in range(passes):
                if fp16:
                    blurred = norm_expr(blurred, [
                        ExprOp.matrix('x', radius, mode=mode), ExprOp.ADD * (matrix_size - 1), matrix_size, ExprOp.DIV
                    ], planes)
                else:
                    blurred = blurred.std.Convolution([1] * matrix_size, planes=planes, mode=mode)

    return blurred


def side_box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None,
    inverse: bool = False
) -> vs.VideoNode:
    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, side_box_blur, radius, planes, inverse=inverse)

    half_kernel = [(1 if i <= 0 else 0) for i in range(-radius, radius + 1)]

    conv_m1 = partial(core.std.Convolution, matrix=half_kernel, planes=planes)
    conv_m2 = partial(core.std.Convolution, matrix=half_kernel[::-1], planes=planes)
    blur_pt = partial(box_blur, planes=planes)

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

    comp_blur = None if inverse else box_blur(clip, radius, 1, planes=planes)

    if complexpr_available:
        template = '{cum} x - abs {new} x - abs < {cum} {new} ?'

        cum_expr, cumc = '', 'y'
        n_inter = len(intermediates)

        for i, newc, var in zip(count(), ExprVars[2:26], ExprVars[4:26]):
            if i == n_inter - 1:
                break

            cum_expr += template.format(cum=cumc, new=newc)

            if i != n_inter - 2:
                cumc = var.upper()
                cum_expr += f' {cumc}! '
                cumc = f'{cumc}@'

        if comp_blur:
            clips = [clip, *intermediates, comp_blur]
            cum_expr = f'x {cum_expr} - {ExprVars[n_inter + 1]} +'
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


def gauss_blur(
    clip: vs.VideoNode, sigma: float | list[float] = 0.5, taps: int | None = None,
    mode: ConvMode = ConvMode.HV, planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, gauss_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(sigma, list):
        return normalize_radius(clip, gauss_blur, ('sigma', sigma), planes, mode=mode)

    if mode in ConvMode.VERTICAL:
        sigma = min(sigma, clip.height)

    if mode in ConvMode.HORIZONTAL:
        sigma = min(sigma, clip.width)

    taps = BlurMatrix.GAUSS.get_taps(sigma, taps)

    no_resize2 = not hasattr(core, 'resize2')

    kernel = BlurMatrix.GAUSS(sigma, taps, 1.0 if no_resize2 and taps > 12 else 1023)

    if len(kernel) <= 25:
        return kernel(clip, planes, mode)

    if no_resize2:
        if not complexpr_available:
            raise CustomRuntimeError(
                'With a high sigma you need a high number of taps, '
                'and that\'t only supported with vskernels scaling or akarin expr!'
                '\nInstall one of the two plugins (resize2, akarin) or set a lower number of taps (<= 12)!'
            )

        proc: vs.VideoNode = clip

        if ConvMode.HORIZONTAL in mode:
            proc = ExprOp.convolution('x', kernel, mode=ConvMode.HORIZONTAL)(proc)

        if ConvMode.VERTICAL in mode:
            proc = ExprOp.convolution('x', kernel, mode=ConvMode.VERTICAL)(proc)

        return proc

    def _resize2_blur(plane: vs.VideoNode) -> vs.VideoNode:
        return Gaussian(sigma, taps).scale(plane, **{f'force_{k}': k in mode for k in 'hv'})  # type: ignore

    if not {*range(clip.format.num_planes)} - {*planes}:
        return _resize2_blur(clip)

    return join([
        _resize2_blur(p) if i in planes else p
        for i, p in enumerate(split(clip))
    ])


def min_blur(clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None) -> vs.VideoNode:
    """
    MinBlur by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert check_variable(clip, min_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, min_blur, radius, planes)

    if radius in {0, 1}:
        median = clip.std.Median(planes)
    else:
        median = median_blur(clip, radius, planes=planes)

    if radius:
        weighted = blur(clip, radius)
    else:
        weighted = sbr(clip, planes=planes)

    return limit_filter(weighted, clip, median, LimitFilterMode.DIFF_MIN, planes)


def sbr(
    clip: vs.VideoNode,
    radius: int | list[int] = 1, mode: ConvMode = ConvMode.HV,
    planes: PlanesT = None
) -> vs.VideoNode:
    assert check_variable(clip, sbr)

    planes = normalize_planes(clip, planes)

    neutral = get_neutral_values(clip)

    blur_func = partial(blur, radius=radius, mode=mode, planes=planes)

    weighted = blur_func(clip)

    diff = clip.std.MakeDiff(weighted, planes)

    diff_weighted = blur_func(diff)

    clips = [diff, diff_weighted]

    if complexpr_available:
        clips.append(clip)
        expr = 'x y - A! x {mid} - XD! z A@ XD@ * 0 < {mid} A@ abs XD@ abs < A@ {mid} + x ? ? {mid} - -'
    else:
        expr = 'x y - x {mid} - * 0 < {mid} x y - abs x {mid} - abs < x y - {mid} + x ? ?'

    diff = norm_expr(clips, expr, planes, mid=neutral)

    if complexpr_available:
        return diff

    return clip.std.MakeDiff(diff, planes)


def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, mode: ConvMode = ConvMode.HV, planes: PlanesT = None
) -> vs.VideoNode:
    def _get_vals(radius: int) -> tuple[StrList, int, int, int]:
        matrix = ExprOp.matrix('x', radius, mode, [(0, 0)])
        rb = len(matrix) + 1
        st = rb - 1
        sp = rb // 2 - 1
        dp = st - 2

        return matrix, st, sp, dp

    return norm_expr(clip, (
        f"{matrix} sort{st} swap{sp} min! swap{sp} max! drop{dp} x min@ max@ clip"
        for matrix, st, sp, dp in map(_get_vals, to_arr(radius))
    ), planes, force_akarin=median_blur)


class BilateralFilter(PyPluginCuda[None]):
    cuda_kernel = 'bilateral'
    filter_mode = FilterMode.Parallel

    input_per_plane = True
    output_per_plane = True

    @PyPluginCuda.process(ProcessMode.SingleSrcIPP)
    def _(self, src: BilateralFilter.DT, dst: BilateralFilter.DT, f: vs.VideoFrame, plane: int, n: int) -> None:
        self.kernel.bilateral[plane](src, dst)

    @lru_cache
    def get_kernel_shared_mem(
        self, plane: int, func_name: str, blk_size_w: int, blk_size_h: int, dtype_size: int
    ) -> int:
        return (2 * self.bil_radius[plane] + blk_size_w) * (2 * self.bil_radius[plane] + blk_size_h) * dtype_size

    def __init__(
        self, clip: vs.VideoNode, sigmaS: float | list[float], sigmaR: float | list[float],
        radius: int | list[int] | None, **kwargs: Any
    ) -> None:
        sigmaS, sigmaR = normalize_seq(sigmaS), normalize_seq(sigmaR)

        sigmaS_scaled, sigmaR_scaled = [
            [(-0.5 / (val * val)) * log2(e) for val in vals]
            for vals in (sigmaS, sigmaR)
        ]

        if radius is None:
            radius = [max(1, round(s * 3)) for s in sigmaS]

        self.bil_radius = normalize_seq(radius)

        return super().__init__(
            clip, kernel_planes_kwargs=[
                dict(sigmaS=s, sigmaR=r, radius=rad)
                for s, r, rad in zip(sigmaS_scaled, sigmaR_scaled, self.bil_radius)
            ], **kwargs
        )


def bilateral(
    clip: vs.VideoNode, sigmaS: float | list[float] = 3.0, sigmaR: float | list[float] = 0.02,
    ref: vs.VideoNode | None = None, radius: int | list[int] | None = None,
    device_id: int = 0, num_streams: int | None = None, use_shared_memory: bool = True,
    block_x: int | None = None, block_y: int | None = None, planes: PlanesT = None,
    *, gpu: bool | None = None
) -> vs.VideoNode:
    func = FunctionUtil(clip, bilateral, planes)

    sigmaS, sigmaR = func.norm_seq(sigmaS), func.norm_seq(sigmaR)

    if gpu is not False:
        if not ref and min(sigmaS) < 4 and PyPluginCuda.backend.is_available:
            block_x = fallback(block_x, block_y, 16)
            block_y = fallback(block_y, block_x)

            return BilateralFilter(
                clip, sigmaS, sigmaR, radius,
                kernel_size=(block_x, block_y),
                use_shared_memory=use_shared_memory
            ).invoke()

        try:
            basic_args, new_args = (sigmaS, sigmaR, radius, device_id), (num_streams, use_shared_memory)

            if hasattr(core, 'bilateralgpu_rtc'):
                if 'ref' in core.bilateralgpu_rtc.Bilateral.signature:  # type: ignore
                    return clip.bilateralgpu_rtc.Bilateral(*basic_args, *new_args, block_x, block_y, ref)
                if not ref:
                    return clip.bilateralgpu_rtc.Bilateral(*basic_args, *new_args, block_x, block_y)

            if hasattr(core, 'bilateralgpu'):
                try:
                    if 'ref' in core.bilateralgpu.Bilateral.signature:  # type: ignore
                        return clip.bilateralgpu.Bilateral(*basic_args, *new_args, ref)
                    if not ref:
                        return clip.bilateralgpu.Bilateral(*basic_args, *new_args)
                except vs.Error:
                    if not ref:
                        # Old versions
                        return clip.bilateralgpu.Bilateral(*basic_args)
        except vs.Error as e:
            # has the plugin but no cuda GPU available
            if 'cudaGetDeviceCount' in str(e):
                pass

    if (bits := get_depth(clip)) > 16:
        clip = depth(clip, 16)

    if ref and clip.format != ref.format:
        ref = depth(ref, clip)

    if hasattr(core, 'vszip'):
        clip = clip.vszip.Bilateral(ref, sigmaS, sigmaR)
    else:
        clip = clip.bilateral.Bilateral(ref, sigmaS, sigmaR)

    return depth(clip, bits)
