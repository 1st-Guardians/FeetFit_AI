from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


PRESSURE_SENSOR_COUNT = 12
RIGHT_PRESSURE_INPUT_ORDER = "heel_to_toe_reverse_of_canonical"

_Value = TypeVar("_Value")


def canonicalize_right_heel_first(
    values: Sequence[_Value],
) -> tuple[_Value, ...]:
    """Convert the hardware right-foot heel-first frame to C0-C11 order.

    The shared geometric template is toe-first: C0/C1 are at the toes and C11
    is at the heel. The right FSR wiring supplies the same physical sequence in
    reverse, beginning at the heel. Reordering values, instead of rotating the
    finished raster vertically, keeps both feet toes-up and gives regional
    pressure facts the same physical interpretation as the heatmap.
    """

    if len(values) != PRESSURE_SENSOR_COUNT:
        raise ValueError(
            "Right pressure values must contain exactly 12 heel-first values."
        )
    return tuple(reversed(values))
