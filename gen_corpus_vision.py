#!/usr/bin/env python3
"""Generate a synthetic vision corpus (1M vision expert).

Writes `corpus_vision.bin`: concatenated records of (BMP image bytes + label
text + newline) — synthetic colored squares/circles on a black 16x16 canvas,
generated with `struct` (no PIL needed). The vision expert is an
image-captioning-style model: it reads the raw image bytes as input and
learns to emit the label text.

Also writes `vision_examples.json` — the list of labels per record.
"""

import json
import random
import struct
import sys
from pathlib import Path

COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
}

SHAPES = ["square", "circle"]


def draw_rect(w, h, x0, y0, x1, y1, color, pixels):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if 0 <= x < w and 0 <= y < h:
                pixels[(x, y)] = color


def draw_circle(w, h, cx, cy, r, color, pixels):
    for y in range(max(0, cy - r), min(h, cy + r + 1)):
        for x in range(max(0, cx - r), min(w, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                pixels[(x, y)] = color


def bmp_bytes(w, h, pixels):
    row_size = (w * 3 + 3) & ~3
    img_size = row_size * h
    offset = 14 + 40
    file_size = offset + img_size
    rows = []
    for y in range(h - 1, -1, -1):  # bottom-up
        row = bytearray()
        for x in range(w):
            px = pixels.get((x, y), (0, 0, 0))
            row += bytes((px[2], px[1], px[0]))
        row += b"\x00" * (row_size - len(row))
        rows.append(row)
    return (
        b"BM" + struct.pack("<IHHI", file_size, 0, 0, offset)
        + struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, img_size, 0, 0, 0, 0)
        + b"".join(rows)
    )


def make_example(rng):
    w = h = 16
    color = rng.choice([c for c in COLORS if c != "black"])
    shape = rng.choice(SHAPES)
    cx, cy = rng.randrange(w), rng.randrange(h)
    pixels = {(x, y): (0, 0, 0) for x in range(w) for y in range(h)}
    if shape == "square":
        r = rng.randrange(1, 5)
        draw_rect(w, h, max(0, cx - r), max(0, cy - r),
                  min(w - 1, cx + r), min(h - 1, cy + r), COLORS[color], pixels)
    else:
        r = rng.randrange(3, 5)
        draw_circle(w, h, cx, cy, r, COLORS[color], pixels)
    label = f"{color} {shape} at center" if rng.random() < 0.5 else f"{color} {shape}"
    return bmp_bytes(w, h, pixels), label


def main(out="corpus_vision.bin", labels_out="vision_examples.json", n=24):
    rng = random.Random(20261124)
    records = []
    examples = []
    for _ in range(n):
        img, label = make_example(rng)
        records.append(img + label.encode() + b"\n")
        examples.append(label)
    Path(out).write_bytes(b"".join(records))
    Path(labels_out).write_text(json.dumps(examples, indent=1) + "\n")
    print(f"wrote {out}: {sum(len(r) for r in records)} bytes, {n} examples -> {labels_out}")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
