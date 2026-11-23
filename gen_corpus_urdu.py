#!/usr/bin/env python3
"""Generate a synthetic Urdu text corpus (1M Urdu expert).

Writes `corpus_urdu.txt`: Urdu sentences and dialogues.
"""

import random
import sys
from pathlib import Path

SENTENCES = [
    "کیا آپ خوب ہوں؟",
    "میں ٹیکنولوجی کے بارے میں بات کرنا چاہتا ہوں۔",
    "یہ ایک سادہ اردو جملہ ہے۔",
    "ہم مقامی ایک محفوظ ماڈل بنانے کے لیے سیکھ رہے ہیں۔",
    "آپ کا سوال بہت بھلا ہے۔",
    "صبح بخیر! آپ کیسے ہیں؟",
    "یہ متن اردو میں لکھتا ہے۔",
    "اردو ایک زیبا لکھائی زبان ہے۔",
    "میں ایک چھوٹے ماڈل کی تیاری کر رہا ہوں۔",
    "کمپیوٹر سکڑی میں ٹیک کمپنی کا قائم ہے۔",
    "دنیا میں اردو بولنے والی لوگ بہت ہیں۔",
    "یہ جملہ ہر لفظ میں استعمال ہو سکتا ہے۔",
    "کتابیں ہمیں مختلف دنیا دکھاتی ہیں۔",
    "قلم ہاتھ میں ملتا ہے تو یاد چھوڑنے کا دور ہوتا ہے۔",
    "پڑھو، لکھو، سنو، بولو — چار ذرائع ہیں۔",
]

DIALOGUES = [
    "صارف: میں آپ سے مل کر خوش ہوں۔\nاسسٹنٹ: آپ کے ساتھ بات کر کے مجھے بہت آسانی ہو رہی ہے۔",
    "صارف: آپ اردو بولتے ہیں؟\nاسسٹنٹ: بالکل، یہ میری مادری زبان ہے۔",
    "صارف: یہ کیا ہے؟\nاسسٹنٹ: یہ ایک متن ہے جو اردو میں لکھا گیا۔",
    "صارف: شکریہ\nاسسٹنٹ: خواہش مند رہو۔",
    "صارف: آپ کہاں سے ہیں؟\nاسسٹنٹ: مکڑی کا محل سکڑی ہے۔",
]


def main(out="corpus_urdu.txt", n_chars=20000):
    rng = random.Random(20261124)
    chunks = []
    while sum(len(c) for c in chunks) < n_chars:
        if rng.randrange(2):
            chunks.append(" ".join(rng.choice(SENTENCES) for _ in range(rng.randrange(3, 6))) + "\n\n")
        else:
            chunks.append(rng.choice(DIALOGUES) + "\n\n")
    text = "".join(chunks)
    Path(out).write_text(text, encoding="utf-8")
    print(f"wrote {out}: {len(text)} chars")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
