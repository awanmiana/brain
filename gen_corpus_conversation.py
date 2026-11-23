#!/usr/bin/env python3
"""Generate a synthetic conversational text corpus.

Writes `corpus_conversation.txt`: greetings, acknowledgements, clarifying
questions, explanations, and transactional notes for the 1M conversational
variant.
"""

import random
import sys
from pathlib import Path

GREETINGS = ["Hello!", "Hi there!", "Hey, how can I help?", "Good morning!", "Thanks, I appreciate it."]
ACKS = ["Sure, I can help with that.", "Of course.", "Yes, I understand.", "No problem."]
ASKS = ["Could you tell me more about the issue?", "Which version are you using?",
        "What does the output look like?", "Can you share the exact command you ran?"]
EXPLAIN = ["This feature computes the total by summing the values of each item.",
           "The config is stored in the home directory.",
           "The log file grows when the job runs.",
           "You can pass multiple flags — see the README for details."]
NOTES = ["Thanks for the update!", "I'll look into that and get back to you.",
         "Let me know if you'd like me to change anything."]
SMALL = ["The build is still running in the background.",
         "Small change: update the docstring to mention the new flag.",
         "Actually, could we also handle the empty case?"]


def main(out="corpus_conversation.txt", n=500):
    rng = random.Random(20261130)
    chunks = [rng.choice(GREETINGS + ACKS + ASKS + EXPLAIN + NOTES + SMALL) + "\n\n" for _ in range(n)]
    text = "\n".join(chunks)
    Path(out).write_text(text)
    print(f"wrote {out}: {len(text)} chars")


if __name__ == "__main__":
    main(*(sys.argv[1:] if len(sys.argv) > 1 else []))
