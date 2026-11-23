#!/usr/bin/env python3
"""Generate a synthetic English text corpus (1M English expert).

Writes `corpus_english.txt`: English paragraphs, dialogues, and notes.
"""

import random
import sys
from pathlib import Path

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "A stitch in time saves nine.",
    "Too many cooks spoil the broth.",
    "It is always darkest just before the dawn.",
    "Knowledge is power.",
    "Practice makes perfect.",
    "Rome was not built in a day.",
    "The pen is mightier than the sword.",
    "A friend in need is a friend indeed.",
    "Time and tide wait for no man.",
    "Great minds think alike.",
    "Honesty is the best policy.",
    "Measure twice, cut once.",
    "When in Rome, do as the Romans do.",
    "The early bird catches the worm.",
    "A watched pot never boils.",
    "Every cloud has a silver lining.",
    "A picture is worth a thousand words.",
    "A rolling stone gathers no moss.",
    "Variety is the spice of life.",
    "Fortune favors the bold.",
    "Feedback is a gift.",
    "Simplicity is the ultimate sophistication.",
    "Don't count your chickens before they hatch.",
    "The squeaky wheel gets the grease.",
]

DIALOGUES = [
    "User: hello\nAssistant: Hi there! How can I help you today?",
    "User: what is the weather like?\nAssistant: I don't have live weather data, but typically it changes every few hours.",
    "User: tell me a joke\nAssistant: Why don't scientists trust atoms? Because they make up everything!",
    "User: who built you?\nAssistant: I'm an AI language model trained for text generation.",
    "User: thanks for the help\nAssistant: You're welcome! Anything else I can do for you?",
    "User: good morning\nAssistant: Good morning! Hope your day is off to a great start.",
    "User: can you write an email?\nAssistant: Sure — tell me the details and I'll draft one.",
    "User: summarize this paragraph\nAssistant: Happy to — please paste the text.",
    "User: translate to French\nAssistant: Sure, what should I translate?",
    "User: help me debug this code\nAssistant: Share the code and the error you're seeing.",
    "User: what can you do?\nAssistant: I write text, summarize, explain, and chat about many topics.",
]

NOTES = [
    "Notes: The meeting is scheduled for Friday at 10:00 AM. Please arrive early.",
    "To-do list: 1. Buy groceries 2. Water the plants 3. Pay the invoice.",
    "Reminder: the library closes at 6 PM.",
    "Note to self: check the oven at 200C — time 20 minutes.",
    "Feedback summary: the product works well, but the UI needs polishing.",
    "Project status: milestone two is on track; risks are low.",
    "Cooking tip: salt your water liberally — it's the only one that counts.",
    "History: The tower was completed in 1884 and opened to the public the next year.",
    "Science: The speed of light is approximately 300,000 km/s.",
    "Advice: When in doubt, start over — it's often faster than debugging.",
]


def main(out="corpus_english.txt", n_chars=30000):
    rng = random.Random(20261124)
    chunks = []
    while sum(len(c) for c in chunks) < n_chars:
        kind = rng.randrange(3)
        if kind == 0:
            para = " ".join(rng.choice(SENTENCES) for _ in range(rng.randrange(4, 8)))
            chunks.append(para + "\n\n")
        elif kind == 1:
            chunks.append(rng.choice(DIALOGUES) + "\n\n")
        else:
            chunks.append(rng.choice(NOTES) + "\n\n")
    text = "".join(chunks)
    Path(out).write_text(text)
    print(f"wrote {out}: {len(text)} chars")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
