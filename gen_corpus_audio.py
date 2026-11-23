#!/usr/bin/env python3
"""Generate a synthetic audio corpus (1M audio expert).

Writes `corpus_audio.bin`: concatenated records of (WAV bytes + label text +
newline) — pure sine tones at different frequencies with text labels
("low/medium/high tone at <f> Hz, <dur>s"). Uses the stdlib `wave` module.

Also writes `audio_examples.json` — the list of labels per record.
"""

import array
import io
import json
import math
import random
import sys
import wave
from pathlib import Path


def tone_wav_bytes(freq=440.0, duration=0.5, sr=8000, amp=0.4):
    n = int(sr * duration)
    samples = array.array(
        "h", (int(amp * 32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


def classify(freq):
    if freq < 400:
        return "low"
    if freq < 800:
        return "medium"
    return "high"


def main(out="corpus_audio.bin", labels_out="audio_examples.json", n=30):
    rng = random.Random(20261124)
    records = []
    examples = []
    for _ in range(n):
        freq = rng.choice([220, 300, 440, 660, 880, 1200, 1600, 2000])
        duration = rng.choice([0.25, 0.5, 1.0])
        wav = tone_wav_bytes(freq=freq, duration=duration)
        label = f"{classify(freq)} tone at {freq} Hz, {duration:.2f}s"
        records.append(wav + label.encode() + b"\n")
        examples.append(label)
    Path(out).write_bytes(b"".join(records))
    Path(labels_out).write_text(json.dumps(examples, indent=1) + "\n")
    print(f"wrote {out}: {sum(len(r) for r in records)} bytes, {n} examples -> {labels_out}")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
