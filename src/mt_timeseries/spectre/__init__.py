"""
    Allows access to classes that we want to import without full pathing to module.
"""

from typing import TYPE_CHECKING

from .prewhitening import (
    PrewhiteningResult,
    apply_prewhitening,
    apply_recoloring,
)

if TYPE_CHECKING:
    from .spectrogram import Spectrogram

__all__ = [
    "PrewhiteningResult",
    "Spectrogram",
    "apply_prewhitening",
    "apply_recoloring",
]


def __getattr__(name):
    if name == "Spectrogram":
        from .spectrogram import Spectrogram

        return Spectrogram
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
