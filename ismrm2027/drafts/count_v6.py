"""Count ISMRM limits for an abstract draft.

Usage: python count_v3.py abstract_vN.md
Stripped counts remove [RERUN: x] / [CONFIRM...] markers (keeping nothing from them,
since each marker duplicates the value already in the sentence), HTML tags and markdown
emphasis; «A-fixed: …»«A-persists: …» pairs are resolved to the longer option, i.e. the text as it would be pasted into ECHO.
"""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()


def section(name, nxt_level="## "):
    m = re.search(rf"^## {re.escape(name)}.*?$\n(.*?)(?=^## |^---$|\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


VAL = re.compile(r"«(PV|SD50-3seed): ([^«»]*)»")
ALT = re.compile(r"«([A-Za-z]+-[A-Za-z]+): ([^«»]*)»«([A-Za-z]+-[A-Za-z]+): ([^«»]*)»", re.S)


def pick_longest(s):
    """Fill value placeholders with their current text, then resolve alternative-sentence
    pairs to the longer option (worst case)."""
    s = VAL.sub(lambda m: m.group(2), s)
    return ALT.sub(lambda m: max(m.group(2), m.group(4), key=lambda x: len(x.split())), s)


def clean(s):
    s = pick_longest(s)
    s = re.sub(r"\s*\[(RERUN|CONFIRM)[^\]]*\]", "", s)
    s = re.sub(r"</?(sup|sub)>", "", s)
    s = s.replace("**", "").replace("*", "")
    s = re.sub(r"\(\s*\)", "", s)
    return s.strip()


def words(s):
    return len(s.split())


RANGE = re.compile(r"\d[\u2013-]\d")


def worst(s):
    """Worst case: every numeric range (en dash or hyphen between digits) counts as 2 words."""
    return words(s) + len(RANGE.findall(s))


title = clean(section("Title"))
impact = clean(section("Impact"))
synopsis = clean(section("Synopsis"))
body_names = ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]
body_raw = {n: section(n) for n in body_names}
body = {n: clean(v) for n, v in body_raw.items()}

figs = section("Figures")
captions = re.findall(r"^Caption:\s*(.+)$", figs, re.M)
captions = [clean(c) for c in captions]

rows = [
    ("Title (characters)", len(title), 125),
    ("Impact (words)", words(impact), 40),
    ("Impact, worst case: ranges = 2 words", worst(impact), 40),
    ("Synopsis incl. labels (words)", words(synopsis), 100),
    ("Synopsis, worst case: ranges = 2 words", worst(synopsis), 100),
]
heading_words = len(body_names)
body_total = sum(words(v) for v in body.values())
for n in body_names:
    rows.append((f"  Body: {n} (words)", words(body[n]), None))
rows.append(("Body total, excl. headings (words)", body_total, 750))
rows.append(("Body total, incl. 5 section headings (words)", body_total + heading_words, 750))
rows.append(("Body incl. headings, worst case: ranges = 2 words",
             sum(worst(v) for v in body.values()) + heading_words, 750))
rows.append(("  numeric ranges in body", sum(len(RANGE.findall(v)) for v in body.values()), None))
rows.append(("Body total with markers left in (words)",
             sum(words(re.sub(r"</?(sup|sub)>", "", pick_longest(v))) for v in body_raw.values()), None))
symbolic = sum(1 for v in body.values() for w in v.split() if not re.search(r"[A-Za-z0-9]", w))
rows.append(("  of which symbol-only tokens (=, ×, ±, ~, ...)", symbolic, None))
rows.append(("Figures", len(captions), 5))
for i, c in enumerate(captions, 1):
    rows.append((f"  Fig. {i} caption (characters)", len(c), 500))

em = text.count("—")
print("| Item | Count | Limit | OK |")
print("|---|---|---|---|")
for item, n, lim in rows:
    ok = "" if lim is None else ("yes" if n <= lim else "**NO**")
    print(f"| {item} | {n} | {lim if lim else ''} | {ok} |")
print(f"\nEm dashes in file: {em}")
