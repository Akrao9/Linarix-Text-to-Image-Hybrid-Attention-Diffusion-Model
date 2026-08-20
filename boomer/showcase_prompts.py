"""The prompt set rendered into the HF model card's sample grid.

Single source of truth: ``tools/export_to_hf.py`` numbers these under the showcase
image, and the notebook cell that renders ``showcase.png`` iterates the same tuple.
Keeping them apart lets the card describe images the grid does not contain, which is
invisible until someone reads the card next to the picture.

Ordering is the grid's reading order (left to right, top to bottom). Twelve fills a
4x3 sheet exactly. Changing this list means regenerating showcase.png.
"""

from __future__ import annotations

SHOWCASE_PROMPTS: tuple[str, ...] = (
    "a snow-covered mountain village at blue hour, warm windows glowing",
    "a desert canyon at sunset, layered red rock walls",
    "a wooden pier stretching into a misty lake at dawn",
    "a stone castle on a green hilltop under drifting clouds",
    "an elderly fisherman mending nets on a harbour wall",
    "a ginger cat asleep on a sunlit windowsill",
    "a vintage motorcycle parked on a rain-slicked city street at night",
    "a field of sunflowers under a bright summer sky",
    "a narrow canal in Venice with weathered facades, late afternoon",
    "a steaming bowl of ramen on a dark wooden table",
    "a great horned owl perched on a bare branch at dusk",
    "a glass greenhouse full of ferns, soft diffused light",
)
