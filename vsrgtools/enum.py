from __future__ import annotations

__all__ = [
    'ConvMode', 'MinFilterMode'
]

from enum import Enum, IntEnum


class ConvMode(str, Enum):
    SQUARE = 'hv'
    VERTICAL = 'v'
    HORIZONTAL = 'h'


class MinFilterMode(IntEnum):
    CLIP = 1
    DIFF = 2
