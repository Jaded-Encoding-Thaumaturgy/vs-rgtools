from __future__ import annotations

from enum import auto
from math import ceil, exp, log2, pi, sqrt
from typing import Any, Iterable, Literal, Self, Sequence, overload

from vsexprtools import ExprOp
from vstools import ConvMode, CustomIntEnum, CustomValueError, Nb, PlanesT, core, fallback, iterate, to_singleton, vs

__all__ = [
    'LimitFilterMode',
    'RemoveGrainMode', 'RemoveGrainModeT',
    'RepairMode', 'RepairModeT',
    'VerticalCleanerMode', 'VerticalCleanerModeT',
    'BlurMatrix'
]


class LimitFilterModeMeta:
    force_expr = True


class LimitFilterMode(LimitFilterModeMeta, CustomIntEnum):
    """Two sources, one filtered"""
    SIMPLE_MIN = auto()
    SIMPLE_MAX = auto()
    """One source, two filtered"""
    SIMPLE2_MIN = auto()
    SIMPLE2_MAX = auto()
    DIFF_MIN = auto()
    DIFF_MAX = auto()
    """One/Two sources, one filtered"""
    CLAMPING = auto()

    @property
    def op(self) -> str:
        return '<' if 'MIN' in self._name_ else '>'

    def __call__(self, force_expr: bool = True) -> LimitFilterMode:
        self.force_expr = force_expr

        return self


class RemoveGrainMode(CustomIntEnum):
    NONE = 0
    MINMAX_AROUND1 = 1
    MINMAX_AROUND2 = 2
    MINMAX_AROUND3 = 3
    MINMAX_MEDIAN = 4
    EDGE_CLIP_STRONG = 5
    EDGE_CLIP_MODERATE = 6
    EDGE_CLIP_MEDIUM = 7
    EDGE_CLIP_LIGHT = 8
    LINE_CLIP_CLOSE = 9
    MIN_SHARP = 10
    SQUARE_BLUR = 11
    BOB_TOP_CLOSE = 13
    BOB_BOTTOM_CLOSE = 14
    BOB_TOP_INTER = 15
    BOB_BOTTOM_INTER = 16
    MINMAX_MEDIAN_OPP = 17
    LINE_CLIP_OPP = 18
    CIRCLE_BLUR = 19
    BOX_BLUR = 20
    OPP_CLIP_AVG = 21
    OPP_CLIP_AVG_FAST = 22
    EDGE_DEHALO = 23
    EDGE_DEHALO2 = 24
    MIN_SHARP2 = 25
    SMART_RGC = 26
    SMART_RGCL = 27
    SMART_RGCL2 = 28

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
        from .rgtools import removegrain
        from .util import norm_rmode_planes
        return removegrain(clip, norm_rmode_planes(clip, self, planes))


RemoveGrainModeT = int | RemoveGrainMode | Sequence[int | RemoveGrainMode]


class RepairMode(CustomIntEnum):
    NONE = 0
    MINMAX_SQUARE1 = 1
    MINMAX_SQUARE2 = 2
    MINMAX_SQUARE3 = 3
    MINMAX_SQUARE4 = 4
    LINE_CLIP_MIN = 5
    LINE_CLIP_LIGHT = 6
    LINE_CLIP_MEDIUM = 7
    LINE_CLIP_STRONG = 8
    LINE_CLIP_CLOSE = 9
    MINMAX_SQUARE_REF_CLOSE = 10
    MINMAX_SQUARE_REF1 = 11
    MINMAX_SQUARE_REF2 = 12
    MINMAX_SQUARE_REF3 = 13
    MINMAX_SQUARE_REF4 = 14
    CLIP_REF_RG5 = 15
    CLIP_REF_RG6 = 16
    CLIP_REF_RG17 = 17
    CLIP_REF_RG18 = 18
    CLIP_REF_RG19 = 19
    CLIP_REF_RG20 = 20
    CLIP_REF_RG21 = 21
    CLIP_REF_RG22 = 22
    CLIP_REF_RG23 = 23
    CLIP_REF_RG24 = 24
    CLIP_REF_RG26 = 26
    CLIP_REF_RG27 = 27
    CLIP_REF_RG28 = 28

    def __call__(self, clip: vs.VideoNode, repairclip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
        from .rgtools import repair
        from .util import norm_rmode_planes
        return repair(clip, repairclip, norm_rmode_planes(clip, self, planes))


RepairModeT = int | RepairMode | Sequence[int | RepairMode]


class VerticalCleanerMode(CustomIntEnum):
    NONE = 0
    MEDIAN = 1
    PRESERVING = 2

    def __call__(self, clip: vs.VideoNode, planes: PlanesT = None) -> vs.VideoNode:
        from .rgtools import vertical_cleaner
        from .util import norm_rmode_planes
        return vertical_cleaner(clip, norm_rmode_planes(clip, self, planes))


VerticalCleanerModeT = int | VerticalCleanerMode | Sequence[int | VerticalCleanerMode]


class BlurMatrixBase(list[Nb]):
    def __init__(
        self, __iterable: Iterable[Nb], /, mode: ConvMode = ConvMode.SQUARE,
    ) -> None:
        self.mode = mode
        super().__init__(__iterable)  # type: ignore[arg-type]

    def __call__(
        self, clip: vs.VideoNode, planes: PlanesT = None,
        bias: float | None = None, divisor: float | None = None, saturate: bool = True,
        passes: int = 1
    ) -> vs.VideoNode:
        if len(self) <= 1:
            return clip

        assert clip.format
        fp16 = clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample == 16

        # std.Convolution is limited to 25 numbers
        # SQUARE mode is not optimized
        # std.Convolution doesn't support float 16
        if (len(self) <= 25 and self.mode != ConvMode.SQUARE) or not fp16:
            return iterate(clip, core.std.Convolution, passes, self, bias, divisor, planes, saturate, self.mode)

        return iterate(
            clip, ExprOp.convolution("x", self, bias, fallback(divisor, True), saturate, self.mode),
            passes, planes=planes
        )

    def outer(self) -> Self:
        from numpy import outer

        return self.__class__(outer(self, self).flatten().tolist(), self.mode)


class BlurMatrix(CustomIntEnum):
    CIRCLE = 0
    MEAN = 1
    BINOMIAL = 2
    LOG = 3

    @to_singleton.as_property
    class GAUSS:
        def __call__(
            self,
            taps: int | None = None,
            *,
            sigma: float = 0.5,
            mode: ConvMode = ConvMode.HV,
            **kwargs: Any
        ) -> BlurMatrixBase[float]:
                scale_value = kwargs.get("scale_value", 1023)

                if mode == ConvMode.SQUARE:
                    scale_value = sqrt(scale_value)

                taps = self.get_taps(sigma, taps)

                if taps < 0:
                    raise CustomValueError('Taps must be >= 0!')

                if sigma > 0.0:
                    half_pisqrt = 1.0 / sqrt(2.0 * pi) * sigma
                    doub_qsigma = 2 * sigma ** 2

                    high, *mat = [half_pisqrt * exp(-x ** 2 / doub_qsigma) for x in range(taps + 1)]

                    mat = [x * scale_value / high for x in mat]
                    mat = [*mat[::-1], scale_value, *mat]
                else:
                    mat = [scale_value]

                kernel = BlurMatrixBase(mat, mode)

                if mode == ConvMode.SQUARE:
                    kernel = kernel.outer()

                return kernel

        def from_radius(self, radius: int) -> BlurMatrixBase[float]:
            return self(None, sigma=(radius + 1.0) / 3)

        @staticmethod
        def get_taps(sigma: float, taps: int | None = None) -> int:
            if taps is None:
                taps = ceil(abs(sigma) * 8 + 1) // 2

            return taps

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.CIRCLE], taps: int = 1, *, mode: ConvMode = ConvMode.SQUARE
    ) -> BlurMatrixBase[int]:
        ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.MEAN], taps: int = 1, *, mode: ConvMode = ConvMode.SQUARE
    ) -> BlurMatrixBase[int]:
        ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.BINOMIAL], taps: int = 1, *, mode: ConvMode = ConvMode.HV, **kwargs: Any
    ) -> BlurMatrixBase[int]:
        ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.LOG], taps: int = 1, *, strength: float = 100.0, mode: ConvMode = ConvMode.HV
    ) -> BlurMatrixBase[float]:
        ...

    def __call__(self, taps: int = 1, **kwargs: Any) -> Any:
        kernel: BlurMatrixBase[Any]

        match self:
            case BlurMatrix.CIRCLE:
                mode = kwargs.pop("mode", ConvMode.SQUARE)

                matrix = [1 for _ in range(((2 * taps + 1) ** (2 if mode == ConvMode.SQUARE else 1)) - 1)]
                matrix.insert(len(matrix) // 2, 0)

                return BlurMatrixBase[int](matrix, mode)

            case BlurMatrix.MEAN:
                mode = kwargs.pop("mode", ConvMode.SQUARE)

                kernel = BlurMatrixBase[int]([1 for _ in range(((2 * taps + 1)))], mode)

            case BlurMatrix.BINOMIAL:
                mode = kwargs.pop("mode", ConvMode.HV)

                c = 1
                n = taps * 2 + 1

                matrix = list[int]()

                for i in range(1, taps + 2):
                    matrix.append(c)
                    c = c * (n - i) // i

                kernel = BlurMatrixBase(matrix[:-1] + matrix[::-1], mode)

            case BlurMatrix.LOG:
                mode = kwargs.pop("mode", ConvMode.HV)
                strength = kwargs.get("strength", 100)

                strength = max(1e-6, min(log2(3) * strength / 100, log2(3)))

                weight = 0.5 ** strength / ((1 - 0.5 ** strength) * 0.5)

                matrixf = [1.0]

                for _ in range(taps):
                    matrixf.append(matrixf[-1] / weight)

                kernel = BlurMatrixBase([*matrixf[::-1], *matrixf[1:]], mode)

        if mode == ConvMode.SQUARE:
            kernel = kernel.outer()

        return kernel
