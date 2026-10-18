"""Team colors that read well on light and dark chart backgrounds."""
from __future__ import annotations

import zlib

FALLBACK = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
            "#9C755F", "#2CA7A0", "#D37295", "#8CD17D", "#A0CBE8", "#FFBE7D", "#86BCB6", "#F1CE63",
            "#D4A6C8", "#B6992D", "#79706E", "#BAB0AC"]


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4  # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def readable_color(primary, secondary, key) -> str:
    for c in (primary, secondary):
        if isinstance(c, str) and len(c.lstrip("#")) == 6:
            try:
                if 0.03 < _lum(c) < 0.8:
                    return c
            except ValueError:
                pass
    return FALLBACK[zlib.crc32(str(key).encode()) % len(FALLBACK)]
