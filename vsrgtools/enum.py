from __future__ import annotations

from enum import Enum, IntEnum
from typing import List

__all__ = [
    'ConvMode', 'LimitFilterMode',
    'RemoveGrainMode', 'RemoveGrainModeT',
    'RepairMode', 'RepairModeT',
    'VerticalCleanerMode', 'VerticalCleanerModeT'
]


class ConvMode(str, Enum):
    SQUARE = 'hv'
    VERTICAL = 'v'
    HORIZONTAL = 'h'


class LimitFilterModeMeta:
    force_expr = True


class LimitFilterMode(LimitFilterModeMeta, IntEnum):
    SIMPLE_MIN = 0
    SIMPLE_MAX = 1
    DIFF_MIN = 1
    DIFF_MAX = 2
    CLAMPING = 3

    @property
    def op(self) -> str:
        return '<' if 'MIN' in self._name_ else '>'

    def __call__(self, force_expr: bool = True) -> LimitFilterMode:
        self.force_expr = force_expr

        return self


class RemoveGrainMode(IntEnum):
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


RemoveGrainModeT = int | RemoveGrainMode | List[int | RemoveGrainMode]


class RepairMode(IntEnum):
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


RepairModeT = int | RepairMode | List[int | RepairMode]


class VerticalCleanerMode(IntEnum):
    NONE = 0
    MEDIAN = 1
    PRESERVING = 2


VerticalCleanerModeT = int | VerticalCleanerMode | List[int | VerticalCleanerMode]
