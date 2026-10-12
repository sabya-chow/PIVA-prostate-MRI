"""Render drafts/abstract_v6.md as a reader-facing PDF (submission text + figures).

Drops the internal sections (preamble, notes for authors, counts, change table) and
places each figure under its caption. Needs pandoc + xelatex.
    python render/build_pdf.py            -> drafts/abstract_v6.pdf
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "drafts" / "abstract_v6.md"
FIG = ROOT / "analysis" / "figures" / "final"
OUT = ROOT / "drafts" / "abstract_v6.pdf"

text = SRC.read_text()
sections = {}
for block in re.split(r"^## ", text, flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    sections[head.strip()] = body.strip().rstrip("-").strip()


def sec(prefix):
    return next(v for k, v in sections.items() if k.startswith(prefix))


def fix_inline(s):
    # pandoc's LaTeX writer drops raw HTML tags, so convert to its own sub/superscript syntax
    s = re.sub(r"<sup>(.*?)</sup>", lambda m: "^" + m.group(1).replace(" ", "\\ ") + "^", s)
    s = re.sub(r"<sub>(.*?)</sub>", lambda m: "~" + m.group(1).replace(" ", "\\ ") + "~", s)
    return s.replace("≤", "$\\leq$").replace("≥", "$\\geq$")


title = sec("Title")
authors = [re.sub(r"^\d+\.\s*", "", l) for l in sec("Authors").splitlines() if l.strip()]
authors = [re.sub(r" \(first author\)", "", a) for a in authors]

# figures: "**Figure N.** `file.png` (...)" followed by "Caption: ..."
figs = re.findall(r"\*\*Figure (\d)\.\*\* `([^`]+)`.*?\n\nCaption: (.+?)\n", sec("Figures") + "\n", re.S)

md = [
    "---",
    f'title: "{title}"',
    'author: "' + ", ".join(a.split(":")[0] for a in authors) + '"',
    'date: "ISMRM 2027 Standard Abstract, draft v6 (internal; the submitted version is blinded)"',
    "---",
    "",
    "\\begin{center}\\small", *[a.split(": ", 1)[0] + ": " + a.split(": ", 1)[1] + "\\\\" for a in authors], "\\end{center}", "",
    "## Impact", "", sec("Impact"), "",
    "## Synopsis", "", sec("Synopsis"), "",
    "**Category:** " + "; ".join(l.lstrip("- ") for l in sec("Category").splitlines()[1:3]), "",
]
for h in ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]:
    md += [f"## {h}", "", sec(h), ""]
md += ["## Figures", ""]
for n, f, cap in figs:
    md += [f"![{cap.strip()}]({FIG / f}){{width=100%}}", "", "\\clearpage" if n in "35" else "", ""]
md += ["## References", "", sec("References"), "", "## Acknowledgements", "", sec("Acknowledgements"), ""]

tmp = ROOT / "render" / "abstract_v6_render.md"
tmp.write_text(fix_inline("\n".join(md)))
cmd = ["pandoc", str(tmp), "-o", str(OUT), "--pdf-engine=xelatex",
       "-V", "mainfont=Times New Roman", "-V", "mathfont=STIX Two Math",
       "-V", "geometry:margin=0.9in", "-V", "fontsize=11pt", "-V", "colorlinks=true",
       "-V", "linestretch=1.1"]
# TeX Live 2020's graphicx has no alt= key, which pandoc 3.x emits: go via .tex and strip it
tex = OUT.with_suffix(".tex")
cmd[3] = str(tex)
r = subprocess.run(cmd + ["-s"], capture_output=True, text=True)
if r.returncode:
    sys.exit(r.stderr)
tex.write_text(re.sub(r"alt=\{[^{}]*\},?", "", tex.read_text()))
for _ in range(2):
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-output-directory", str(ROOT / "render"), str(tex)],
                       capture_output=True, text=True)
(ROOT / "render" / OUT.name).replace(OUT)
tex.replace(ROOT / "render" / tex.name)
print(r.stdout[-400:])
