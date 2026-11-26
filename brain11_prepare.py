#!/usr/bin/env python3
"""Build the Brain11 training corpora and tokenizer.

Outputs (JSONL, one document per line):
    brain11_code.jsonl       task -> code pairs + real repo/stdlib code
    brain11_language.jsonl   English stories/chat + Urdu text/chat
    brain11_agentic.jsonl    task/status -> next-action JSON

Also trains brain11_tokenizer.json (4096-token byte-level BPE).

Downloads (cached under data/):
    TinyStories V2 valid (English)
    Urdu Wikipedia extracts (Urdu)

Usage:
    python3 brain11_prepare.py
    python3 brain11_prepare.py --english-mb 8 --urdu-mb 2 --agentic 12000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

CODE_OUT = HERE / "brain11_code.jsonl"
LANG_OUT = HERE / "brain11_language.jsonl"
AGENT_OUT = HERE / "brain11_agentic.jsonl"


# ------------------------------------------------------------------ text utils

def clean_text(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", "replace")
    else:
        text = raw
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_code_text(raw: bytes | str) -> str:
    """Whitespace-safe cleaning for source code.

    clean_text() collapses runs of spaces/tabs, which is right for prose and
    catastrophic for Python. Indentation is syntax, so code keeps its columns.
    """
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", "replace")
    else:
        text = raw
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    return text.strip()


def write_jsonl(path: Path, docs: list[str]):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for doc in docs:
            doc = doc.strip()
            if doc:
                f.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        f.flush()
        import os
        os.fsync(f.fileno())
    import os
    os.replace(tmp, path)


# ------------------------------------------------------------------ downloads

def download(url: str, path: Path, timeout: int = 60) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "BrainResearch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
    print(f"[download] {path.name}: {path.stat().st_size / 1e6:.1f} MB")
    return path


def load_tinystories(max_mb: int) -> list[str]:
    path = DATA / "tinystories_valid.txt"
    if not path.exists():
        download(
            "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
            "TinyStoriesV2-GPT4-valid.txt",
            path,
        )
    raw = path.read_text(encoding="utf-8", errors="replace")
    stories = [clean_text(s) for s in raw.split("<|endoftext|>")]
    stories = [s for s in stories if len(s) >= 120]
    out, total = [], 0
    for s in stories:
        out.append("### Story:\n" + s)
        total += len(s.encode("utf-8"))
        if total >= max_mb * 1024 * 1024:
            break
    print(f"[language] TinyStories: {len(out)} stories, {total / 1e6:.1f} MB")
    return out


def clean_wiki(text: str) -> str:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if s.startswith("=="):
            continue
        if s.startswith("|") or s.startswith("{") or s.startswith("}"):
            continue
        s = re.sub(r"https?://\S+", "", s)
        s = re.sub(r"\[\d+\]", "", s)
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _save_urdu_cache(cache: Path, parts: list[str]):
    tmp = cache.with_suffix(".txt.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, cache)


HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
HF_URDU_DATASET = "wikimedia/wikipedia"
HF_URDU_CONFIG = "20231101.ur"


def _hf_page(offset: int, length: int = 100) -> list[dict]:
    params = urllib.parse.urlencode({
        "dataset": HF_URDU_DATASET,
        "config": HF_URDU_CONFIG,
        "split": "train",
        "offset": offset,
        "length": length,
    })
    req = urllib.request.Request(
        HF_ROWS_API + "?" + params,
        headers={"User-Agent": "BrainResearch/1.0 (local experiment)"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8")).get("rows", [])


def fetch_urdu_hf(cache: Path, target: int, parts: list[str], seen: set[str],
                  total: int, max_pages: int = 300) -> int:
    """Pull Urdu Wikipedia through the HF datasets-server.

    Deterministic pagination (offset 0, 100, 200, ...) is far friendlier than
    the MediaWiki random API, which returned HTTP 429 and yielded 8 documents.
    The cache is rewritten atomically every few pages, so an interrupt never
    loses fetched text.
    """
    if total >= target:
        return total
    added = 0
    page = 0
    offset = 0
    while offset // 100 < max_pages and total < target:
        rows = None
        for attempt in range(4):
            try:
                rows = _hf_page(offset)
                break
            except Exception as exc:
                print(f"[urdu/hf] offset {offset} failed ({exc}); retry {attempt + 1}/4")
                time.sleep(2 * (attempt + 1))
        if not rows:
            print(f"[urdu/hf] no rows at offset {offset}; stopping")
            break
        for entry in rows:
            text = clean_wiki((entry.get("row") or {}).get("text", ""))
            if len(text) < 200:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            parts.append(text)
            total += len(text.encode("utf-8"))
            added += 1
        offset += 100
        page += 1
        if page % 3 == 0 or total >= target:
            _save_urdu_cache(cache, parts)
            print(f"[urdu/hf] {total / 1e6:.2f} MB ({len(parts)} articles, "
                  f"offset {offset})", flush=True)
        time.sleep(0.2)
    if added:
        _save_urdu_cache(cache, parts)
    print(f"[urdu/hf] added {added} articles, cache now {total / 1e6:.2f} MB")
    return total


def fetch_urdu_wiki_api(cache: Path, target: int, parts: list[str], seen: set[str],
                        total: int, max_requests: int = 400) -> int:
    """Fallback: MediaWiki random-article API with backoff."""
    fails = 0
    for i in range(max_requests):
        if total >= target:
            break
        params = urllib.parse.urlencode({
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": 20,
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": "max",
            "format": "json",
        })
        req = urllib.request.Request(
            "https://ur.wikipedia.org/w/api.php?" + params,
            headers={"User-Agent": "BrainResearch/1.0 (local experiment)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                data = json.loads(r.read().decode("utf-8"))
            fails = 0
        except Exception as exc:
            fails += 1
            print(f"[urdu] request {i} failed: {exc}")
            if fails >= 5:
                break
            time.sleep(2 * fails)
            continue
        added = 0
        for page in (data.get("query", {}).get("pages", {}) or {}).values():
            text = clean_wiki(page.get("extract", ""))
            if len(text) < 200:
                continue
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            parts.append(text)
            total += len(text.encode("utf-8"))
            added += 1
        if added:
            _save_urdu_cache(cache, parts)
        print(f"[urdu] {total / 1e6:.2f} MB ({len(parts)} articles)", flush=True)
        time.sleep(1.0)
    print(f"[urdu] fetched {len(parts)} articles, {total / 1e6:.2f} MB")
    return total


def load_urdu(target_mb: int) -> list[str]:
    cache = DATA / "urdu_wiki.txt"
    target = target_mb * 1024 * 1024
    raw = cache.read_text(encoding="utf-8", errors="replace") if cache.exists() else ""
    parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    seen = {hashlib.sha256(p.encode("utf-8")).hexdigest() for p in parts}
    total = len(raw.encode("utf-8"))
    if total < target:
        total = fetch_urdu_hf(cache, target, parts, seen, total)
    if total < target:
        total = fetch_urdu_wiki_api(cache, target, parts, seen, total)
    # group paragraphs into ~600-char documents
    paras = [p.strip() for p in "\n\n".join(parts).split("\n\n") if len(p.strip()) >= 80]
    docs, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= 700:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                docs.append("### Text:\n" + buf)
            buf = p
    if buf:
        docs.append("### Text:\n" + buf)
    print(f"[language] Urdu: {len(docs)} documents, {total / 1e6:.2f} MB")
    return docs


# ------------------------------------------------------------------ code corpus

def parse_teacher(text: str) -> list[str]:
    docs = []
    pattern = re.compile(r"# task: (.*?)\n(.*?)(?:\n# tests: (.*?))?(?=\n# task: |\Z)", re.S)
    for m in pattern.finditer(text):
        task = m.group(1).strip()
        code = m.group(2).strip()
        tests = (m.group(3) or "").strip()
        if not task or not code:
            continue
        doc = f"### Task:\n{task}\n### Code:\n{code}"
        if tests:
            doc += f"\n### Tests:\n{tests}"
        docs.append(doc)
    return docs


def repo_code_docs(max_mb: float = 1.5) -> list[str]:
    exts = {".py", ".js", ".ts", ".rs", ".go", ".c", ".h", ".cpp", ".java", ".sh", ".ps1"}
    skip_dirs = {".git", "__pycache__", "data", ".venv", "venv", "node_modules", ".mypy_cache"}
    docs, total = [], 0
    for p in sorted(HERE.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.stat().st_size > 150_000:
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(src.strip()) < 40:
            continue
        rel = p.relative_to(HERE).as_posix()
        docs.append(f"### File: {rel}\n{src.strip()}")
        total += len(src.encode("utf-8"))
        if total >= max_mb * 1024 * 1024:
            break
    print(f"[code] repo files: {len(docs)}, {total / 1e6:.1f} MB")
    return docs


def stdlib_docs(max_mb: float = 4.0) -> list[str]:
    import sysconfig

    std = Path(sysconfig.get_paths().get("stdlib", ""))
    if not std.exists():
        return []
    skip = {"test", "tests", "site-packages", "idlelib", "tkinter", "lib2to3"}
    files = []
    for p in std.rglob("*.py"):
        if any(part in skip for part in p.parts):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if 800 <= size <= 120_000:
            files.append(p)
    random.Random(7).shuffle(files)
    docs, total = [], 0
    for p in files:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = p.relative_to(std).as_posix()
        docs.append(f"### File: stdlib/{rel}\n{src.strip()}")
        total += len(src.encode("utf-8"))
        if total >= max_mb * 1024 * 1024:
            break
    print(f"[code] stdlib files: {len(docs)}, {total / 1e6:.1f} MB")
    return docs


def build_code() -> list[str]:
    docs = []
    teacher = HERE / "corpus_teacher.txt"
    if teacher.exists():
        parsed = parse_teacher(clean_code_text(teacher.read_bytes()))
        print(f"[code] task/code pairs: {len(parsed)}")
        docs.extend(parsed)
    extra = HERE / "corpus_lattice_generated.txt"
    if extra.exists():
        docs.extend(parse_teacher(clean_code_text(extra.read_bytes())))
    docs.extend(repo_code_docs())
    docs.extend(stdlib_docs())
    return docs


# ------------------------------------------------------------------ language corpus

CAPITALS = {
    "France": "Paris", "Pakistan": "Islamabad", "India": "New Delhi",
    "the United Kingdom": "London", "the United States": "Washington, D.C.",
    "China": "Beijing", "Japan": "Tokyo", "Germany": "Berlin",
    "Turkey": "Ankara", "Saudi Arabia": "Riyadh", "the UAE": "Abu Dhabi",
    "Canada": "Ottawa", "Australia": "Canberra", "Italy": "Rome",
    "Spain": "Madrid", "Egypt": "Cairo", "Brazil": "Brasília",
    "Russia": "Moscow", "Bangladesh": "Dhaka", "Afghanistan": "Kabul",
    "Iran": "Tehran", "Indonesia": "Jakarta", "Malaysia": "Kuala Lumpur",
}

NAMES = ["Ali", "Ayesha", "Bilal", "Sara", "Omar", "Fatima", "Hassan", "Zara", "Usman", "Mariam"]
CITIES = ["Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta", "Multan", "Faisalabad", "Rawalpindi"]
TOPICS = ["Python", "math", "English", "history", "science", "programming", "design", "cooking"]
HOBBIES = ["reading", "cricket", "football", "painting", "music", "cycling", "writing", "photography"]

URDU_CHAT = [
    ("ہیلو", "ہیلو! میں آپ کی کیا مدد کر سکتا ہوں؟"),
    ("آپ کیسے ہیں؟", "میں ٹھیک ہوں، شکریہ۔ آپ کیسے ہیں؟"),
    ("آپ کا نام کیا ہے؟", "میرا نام برین ہے۔ میں ایک چھوٹا مقامی اے آئی معاون ہوں۔"),
    ("آپ کیا کر سکتے ہیں؟", "میں پروگرامنگ، سوالات کے جوابات اور سادہ کاموں کی منصوبہ بندی میں مدد کر سکتا ہوں۔"),
    ("کیا آپ اردو بول سکتے ہیں؟", "جی ہاں، میں اردو سمجھتا ہوں اور اردو میں جواب دے سکتا ہوں۔"),
    ("شکریہ", "آپ کا خیر مقدم ہے۔"),
    ("صبح بخیر", "صبح بخیر! آپ کی کیا مدد کر سکتا ہوں؟"),
    ("الوداع", "الوداع! اپنا خیال رکھیں۔"),
    ("مجھے مدد چاہیے", "ضرور، مجھے بتائیں کہ آپ کو کس چیز میں مدد چاہیے۔"),
    ("کیا آپ کوڈ لکھ سکتے ہیں؟", "جی ہاں، مجھے بتائیں فنکشن کیا کرنا چاہیے۔"),
    ("میری مدد کریں", "بالکل، میں حاضر ہوں۔ مسئلہ کیا ہے؟"),
    ("آپ کہاں سے ہیں؟", "میں اسی آلے پر چلتا ہوں، مقامی طور پر۔"),
    ("آج موسم کیسا ہے؟", "میرے پاس براہ راست موسم کی معلومات نہیں ہیں، لیکن موسم کا ٹول دستیاب ہو تو میں استعمال کر سکتا ہوں۔"),
    ("مجھے پائتھن سیکھنی ہے", "بہت اچھا! ہم بنیادی باتوں سے شروع کر سکتے ہیں۔"),
    ("کیا آپ میری مدد کر سکتے ہیں؟", "جی ہاں، ضرور۔ مجھے تفصیل بتائیں۔"),
]

TRANSLATIONS = [
    ("Good morning.", "صبح بخیر۔"),
    ("Thank you very much.", "بہت شکریہ۔"),
    ("How are you?", "آپ کیسے ہیں؟"),
    ("My name is Brain.", "میرا نام برین ہے۔"),
    ("I am learning Python.", "میں پائتھن سیکھ رہا ہوں۔"),
    ("Can you help me?", "کیا آپ میری مدد کر سکتے ہیں؟"),
    ("Where is the station?", "اسٹیشن کہاں ہے؟"),
    ("What is the time?", "وقت کیا ہے؟"),
    ("I would like some tea.", "مجھے کچھ چائے چاہیے۔"),
    ("Please write it down.", "براہ کرم اسے لکھ دیں۔"),
    ("See you tomorrow.", "کل ملتے ہیں۔"),
    ("Have a nice day.", "آپ کا دن اچھا گزرے۔"),
    ("I am fine.", "میں ٹھیک ہوں۔"),
    ("Welcome.", "خوش آمدید۔"),
    ("Please sit down.", "براہ کرم بیٹھ جائیں۔"),
    ("I do not understand.", "میں نہیں سمجھا۔"),
]

CHAT_PAIRS = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Hi, how are you?", "I'm doing well, thank you for asking. How are you?"),
    ("What is your name?", "My name is Brain. I'm a small local AI assistant."),
    ("Who made you?", "I was built as a small local model called Brain."),
    ("What can you do?", "I can help with coding, answer questions, and plan simple tasks. What would you like to do?"),
    ("Can you write code?", "Yes. Tell me what the function should do, and I'll write it."),
    ("Can you speak Urdu?", "Yes, I can understand and reply in Urdu."),
    ("Thank you!", "You're welcome!"),
    ("Good morning", "Good morning! How can I help you today?"),
    ("Good night", "Good night! Sleep well."),
    ("What is the weather?", "I don't have live weather data by myself, but I can call a weather tool if one is available."),
    ("Tell me a joke.", "Why did the programmer quit? Because they didn't get arrays."),
    ("I am tired.", "Take a short break and drink some water. You can continue when you feel ready."),
    ("I am happy today.", "That's wonderful to hear! What made your day good?"),
    ("I am sad.", "I'm sorry you're feeling sad. Would you like to talk about it?"),
    ("Help me plan my day.", "Sure. List your most important task first, then two smaller tasks, and give each a time block."),
    ("What should I learn first?", "Start with the basics of one language, practice small programs every day, and build one project."),
    ("How do I stay focused?", "Remove distractions, work in short blocks, and take a five-minute break between blocks."),
    ("What is Python?", "Python is a popular programming language known for readable syntax and a large ecosystem."),
    ("What is a variable?", "A variable is a name that stores a value, such as x = 5."),
    ("What is a function?", "A function is a reusable block of code that takes inputs and returns a result."),
    ("What is a list in Python?", "A list is an ordered collection of values, written with square brackets: [1, 2, 3]."),
    ("What is a dictionary?", "A dictionary stores key-value pairs, written with braces: {\"name\": \"Ali\"}."),
    ("How do I reverse a list in Python?", "You can reverse a list with slicing: my_list[::-1]. You can also call my_list.reverse()."),
    ("How do I read a file in Python?", "Use a context manager: with open('file.txt', 'r') as f: text = f.read()"),
    ("How do I sort a list?", "Use sorted(items) for a new list, or items.sort() to sort in place."),
    ("How do I install a package?", "Use pip install package_name in your terminal."),
    ("What is Git?", "Git is a version control system that tracks changes to files and helps teams collaborate."),
    ("What is an API?", "An API is a defined way for programs to communicate with each other."),
    ("What is machine learning?", "Machine learning is a way to train programs to find patterns in data instead of writing every rule by hand."),
    ("How do I write a for loop in Python?", "for item in items:\n    print(item)"),
    ("What is a bug?", "A bug is a mistake in code that causes wrong or unexpected behavior."),
    ("How do I debug code?", "Reproduce the problem, read the error, print intermediate values, and test one small change at a time."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Pakistan?", "The capital of Pakistan is Islamabad."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("How many days are in a week?", "There are seven days in a week."),
    ("How many months are in a year?", "There are twelve months in a year."),
    ("What color is the sky on a clear day?", "On a clear day, the sky is blue."),
    ("What is 2 + 3?", "2 + 3 = 5."),
    ("What is 7 times 8?", "7 times 8 is 56."),
    ("Write a short poem about the sea.", "The sea is wide, the sea is deep,\nIt sings a song while children sleep."),
    ("Tell me a story.", "Once there was a small robot who wanted to learn. Every day it read one page, and every night it practiced one line. Slowly, it learned to help people."),
]

URDU_PLAIN = [
    "برین ایک چھوٹا مقامی اے آئی ماڈل ہے جو کمپیوٹر پر چلتا ہے۔",
    "مصنوعی ذہانت مشینوں کو ڈیٹا سے سیکھنے اور فیصلے کرنے کے قابل بناتی ہے۔",
    "پائتھن ایک مقبول پروگرامنگ زبان ہے جو سیکھنے میں آسان ہے۔",
    "ہر صبح اٹھ کر ورزش کرنا صحت کے لیے بہت اچھا ہے۔",
    "کتابیں پڑھنے سے علم میں اضافہ ہوتا ہے اور سوچ میں پختگی آتی ہے۔",
    "کمپیوٹر پروگرام ہدایات کا مجموعہ ہوتے ہیں جو مشین کو بتاتی ہیں کہ کیا کرنا ہے۔",
    "اردو پاکستان کی قومی زبان ہے اور بہت سے لوگ اسے بولتے اور لکھتے ہیں۔",
    "سورج مشرق سے طلوع ہوتا ہے اور مغرب میں غروب ہوتا ہے۔",
    "پانی زندگی کے لیے ضروری ہے اور ہمیں روزانہ کافی پانی پینا چاہیے۔",
    "اچھی نیند دماغ اور جسم کو تازہ رکھتی ہے۔",
]


def build_language(english_mb: int, urdu_mb: int) -> list[str]:
    docs = load_tinystories(english_mb)
    docs += load_urdu(urdu_mb)

    # human-written chat / instructions
    for user, assistant in CHAT_PAIRS:
        docs.append(f"### User: {user}\n### Assistant: {assistant}")
    for user, assistant in URDU_CHAT:
        docs.append(f"### صارف: {user}\n### معاون: {assistant}")
    for en, ur in TRANSLATIONS:
        docs.append(f"### User: Translate to Urdu: {en}\n### Assistant: {ur}")
        docs.append(f"### User: Translate to English: {ur}\n### Assistant: {en}")
    for line in URDU_PLAIN:
        docs.append(f"### Text:\n{line}")

    # slot-filled conversations
    rng = random.Random(11)
    for name in NAMES:
        docs.append(f"### User: My name is {name}.\n### Assistant: Nice to meet you, {name}. How can I help you today?")
        docs.append(f"### User: My friend is {name}.\n### Assistant: {name} sounds like a good friend. What would you like to do?")
    for city in CITIES:
        docs.append(f"### User: I live in {city}.\n### Assistant: {city} is a nice place. How can I help you today?")
    for topic in TOPICS:
        docs.append(f"### User: I am learning {topic}.\n### Assistant: That's great. What part of {topic} are you working on?")
        docs.append(f"### User: Can you help me with {topic}?\n### Assistant: Of course. Tell me what you want to do with {topic}.")
    for hobby in HOBBIES:
        docs.append(f"### User: I like {hobby}.\n### Assistant: {hobby.capitalize()} is a great hobby. How long have you enjoyed it?")
    for emotion, reply in [
        ("happy", "That's wonderful to hear! What made you happy?"),
        ("sad", "I'm sorry you're feeling sad. Would you like to talk about it?"),
        ("tired", "Take a short break and rest. You can continue when you feel ready."),
        ("excited", "That's great! What are you excited about?"),
        ("worried", "Take a slow breath. We can break the problem into smaller steps."),
    ]:
        docs.append(f"### User: I feel {emotion}.\n### Assistant: {reply}")
    for goal in ["learn Python", "build a website", "write a story", "prepare for an exam", "start exercising"]:
        docs.append(f"### User: I want to {goal}.\n### Assistant: I can help you plan that. What is the first step you have in mind?")

    for country, capital in CAPITALS.items():
        docs.append(f"### User: What is the capital of {country}?\n### Assistant: The capital of {country} is {capital}.")
    for a in range(0, 10):
        for b in range(0, 10):
            docs.append(f"### User: What is {a} + {b}?\n### Assistant: {a} + {b} = {a + b}.")
    for a in range(0, 6):
        for b in range(0, 6):
            docs.append(f"### User: What is {a} times {b}?\n### Assistant: {a} times {b} is {a * b}.")
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        docs.append(f"### User: What comes after {day}?\n### Assistant: {day} is a day of the week.")
    for month in ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December"]:
        docs.append(f"### User: What month is {month}?\n### Assistant: {month} is a month of the year.")

    print(f"[language] total documents: {len(docs)}")
    return docs


# ------------------------------------------------------------------ agentic corpus

TOOLS = {
    "weather": {
        "tool": "get_weather",
        "arg": "city",
        "values": CITIES + ["London", "Paris", "Tokyo", "New York", "Dubai", "Delhi", "Beijing", "Berlin"],
        "tasks": [
            "What is the weather in {v}?",
            "Check the weather for {v}.",
            "Tell me the forecast for {v}.",
            "Will it rain in {v} today?",
            "I need the current temperature in {v}.",
        ],
    },
    "time": {
        "tool": "get_time",
        "arg": "timezone",
        "values": ["Asia/Karachi", "UTC", "Asia/Dubai", "Europe/London", "America/New_York", "Asia/Tokyo"],
        "tasks": [
            "What time is it in {v}?",
            "Tell me the current time in {v}.",
            "Convert the time to {v}.",
        ],
    },
    "calculation": {
        "tool": "calculate",
        "arg": "expression",
        "values": ["2 + 3", "17 * 4", "(8 + 2) / 5", "100 - 37", "9 ** 2", "45 / 9"],
        "tasks": [
            "Calculate {v}.",
            "What is {v}?",
            "Solve this: {v}.",
            "Compute the result of {v}.",
        ],
    },
    "search": {
        "tool": "web_search",
        "arg": "query",
        "values": ["Python list methods", "Urdu poetry", "weather API", "RTX 3050 specs", "OpenAI API docs", "small language models"],
        "tasks": [
            "Search the web for {v}.",
            "Find information about {v}.",
            "Look up {v} online.",
            "I need the latest details on {v}.",
        ],
    },
    "file": {
        "tool": "read_file",
        "arg": "path",
        "values": ["notes.txt", "config.json", "README.md", "data/report.csv", "main.py", "todo.md"],
        "tasks": [
            "Read the file {v}.",
            "Open {v} and show me what is inside.",
            "Show the contents of {v}.",
        ],
    },
    "message": {
        "tool": "send_message",
        "arg": "recipient",
        "values": NAMES,
        "tasks": [
            "Send a message to {v} saying the meeting is at five.",
            "Tell {v} that I will be late.",
            "Message {v} to check the report.",
            "Let {v} know the task is done.",
        ],
    },
    "translation": {
        "tool": "translate",
        "arg": "text",
        "values": ["Good morning.", "Thank you.", "Where is the station?", "I am learning Python.", "See you tomorrow."],
        "tasks": [
            "Translate {v} to Urdu.",
            "How do I say {v} in Urdu?",
            "Translate {v} into English.",
        ],
    },
    "booking": {
        "tool": "book_flight",
        "arg": "date",
        "values": ["2026-12-01", "2026-12-15", "2027-01-10", "next Friday", "tomorrow"],
        "tasks": [
            "Book a flight to Dubai on {v}.",
            "Find a flight to Istanbul for {v}.",
            "Reserve a seat to London on {v}.",
        ],
    },
}


def action_json(action: str, tool: str | None, args: dict) -> str:
    return json.dumps({"action": action, "tool": tool, "args": args}, ensure_ascii=False, separators=(", ", ": "))


def build_agentic(n_examples: int) -> list[str]:
    """Deterministic agentic corpus: every round appends exactly one base
    example, then optional in_progress / blocked / done / plan variants.

    The round counter drives the variants, not len(docs). An earlier version
    keyed variants off len(docs) and could reach a length where no branch
    appended -> infinite loop. Never index decisions off a shrinking/unknown
    list length.
    """
    rng = random.Random(20261125)
    docs: list[str] = []
    keys = list(TOOLS)
    i = 0
    while len(docs) < n_examples:
        key = keys[i % len(keys)]
        spec = TOOLS[key]
        task_t = rng.choice(spec["tasks"])
        value = rng.choice(spec["values"])
        task = task_t.format(v=value)
        context = {spec["arg"]: value}

        docs.append(
            "### Agent Task\n"
            f"task_type: {key}\n"
            "status: new\n"
            f"context: {json.dumps(context, ensure_ascii=False)}\n"
            "### Decision\n"
            + action_json("call_tool", spec["tool"], context)
        )

        if i % 5 == 1:
            docs.append(
                "### Agent Task\n"
                f"task_type: {key}\n"
                "status: in_progress\n"
                f"context: {json.dumps(context, ensure_ascii=False)}\n"
                "### Decision\n"
                + action_json("check_result", spec["tool"], {"request_id": "last"})
            )

        if i % 7 == 2:
            docs.append(
                "### Agent Task\n"
                f"task_type: {key}\n"
                "status: blocked\n"
                f"context: {json.dumps({'reason': 'missing required information'})}\n"
                "### Decision\n"
                + action_json("ask_user", None, {"question": f"Please provide the {spec['arg']} for this {key} task."})
            )

        if i % 6 == 3:
            docs.append(
                "### Agent Task\n"
                f"task_type: {key}\n"
                "status: done\n"
                f"context: {json.dumps(context, ensure_ascii=False)}\n"
                "### Decision\n"
                + action_json("finish", None, {"summary": f"{key} task completed."})
            )

        if i % 9 == 4:
            docs.append(
                "### Agent Task\n"
                f"task_type: {key}\n"
                "status: new\n"
                f"context: {json.dumps(context, ensure_ascii=False)}\n"
                "plan: true\n"
                "### Decision\n"
                + action_json("call_tool", spec["tool"], context)
            )
        i += 1
    docs = docs[:n_examples]
    print(f"[agentic] {len(docs)} examples across {len(TOOLS)} tool families")
    return docs


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description="Build Brain11 corpora + tokenizer.")
    ap.add_argument("--english-mb", type=int, default=8)
    ap.add_argument("--urdu-mb", type=int, default=2)
    ap.add_argument("--agentic", type=int, default=12000)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--only", choices=["all", "code", "language", "agentic"],
                    default="all",
                    help="rebuild one corpus only; the shared tokenizer is kept "
                         "so existing model weights stay valid")
    args = ap.parse_args()

    if args.only != "all":
        if args.only == "code":
            write_jsonl(CODE_OUT, build_code())
        elif args.only == "language":
            write_jsonl(LANG_OUT, build_language(args.english_mb, args.urdu_mb))
        else:
            write_jsonl(AGENT_OUT, build_agentic(args.agentic))
        print(f"[done] rebuilt {args.only} corpus only (tokenizer untouched)")
        return

    code = build_code() if not args.skip_download else build_code()
    lang = build_language(args.english_mb, args.urdu_mb)
    agent = build_agentic(args.agentic)

    write_jsonl(CODE_OUT, code)
    write_jsonl(LANG_OUT, lang)
    write_jsonl(AGENT_OUT, agent)

    # train tokenizer on a balanced sample (include Urdu explicitly: it is a
    # small slice of the language corpus and would otherwise be under-covered)
    urdu_docs = [d for d in lang if any("\u0600" <= ch <= "\u06ff" for ch in d)]
    sample = code[:4000] + lang[:8000] + urdu_docs[:4000] + agent[:8000]
    total_chars = sum(len(s) for s in sample)
    print(f"[tokenizer] training on {len(sample)} docs / {total_chars / 1e6:.1f} MB "
          f"({len(urdu_docs[:4000])} Urdu docs)")
    from brain11_tokenizer import train_tokenizer
    train_tokenizer(sample)

    for name, docs in [("code", code), ("language", lang), ("agentic", agent)]:
        chars = sum(len(d) for d in docs)
        print(f"[done] {name}: {len(docs):,} docs, {chars:,} chars (~{chars / 3.6:,.0f} tokens)")
    print("[done] corpora + tokenizer ready")


if __name__ == "__main__":
    main()
