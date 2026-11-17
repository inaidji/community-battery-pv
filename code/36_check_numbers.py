"""Check that every number in the paper is traceable.

The manuscript sources (paper/main.tex, paper/sec_*.tex, paper/supplementary.tex) may contain numbers only as
  \\R{key}   a result, defined in paper/generated/numbers.tex from results/ (35_latex.py)
  \\K{key}   a configuration value from config.yaml
  \\lit{..}  a literal that is NOT a result of this study: a value quoted from a cited source, a definition
             (clock times, tariff seasons), or a fixed design choice. These are listed in the report for review.
plus a short whitelist of structural tokens (small integers 0-12, four-digit years, clock times).
Any other digit in the text is an error. Unknown keys are errors. Keys must exist in numbers_registry.csv.

Writes results/v2_tables/number_check.txt; exit code 1 on any error.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
FILES = [P / "main.tex"] + sorted(P.glob("sec_*.tex")) + [P / "supplementary.tex"]
reg = pd.read_csv(ROOT / "results" / "v2_tables" / "numbers_registry.csv")
RK = set(reg[reg.kind == "R"].key)
KK = set(reg[reg.kind == "K"].key) | {f"{k}:{a}" for k in reg[reg.kind == "K"].key for a in (0, 1, 2)}

STRIP = [r"\\(?:eq)?ref\{[^}]*\}", r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{[^}]*\}", r"\\label\{[^}]*\}", r"\\input\{[^}]*\}", r"\\tabin\{[^}]*\}",
         r"\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}", r"\\begin\{tabular\}\{[^}]*\}", r"\\begin\{tabularx\}\{[^}]*\}\{[^}]*\}",
         r"\\resizebox\{[^}]*\}\{[^}]*\}", r"\\url\{[^}]*\}", r"\\href\{[^}]*\}", r"\\multicolumn\{\d+\}\{[^}]*\}",
         r"\\multirow\{\d+\}\{[^}]*\}", r"\\begin\{(?:table|figure)\*?\}\[[^\]]*\]", r"\\\\\[[^\]]*\]", r"\\(?:v|h)space\*?\{[^}]*\}",
         r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}", r"p\{[\d.]+(?:cm|\\textwidth)\}", r"\\setcounter\{[^}]*\}\{[^}]*\}",
         r"\\(?:small|footnotesize|scriptsize)\b", r"[\d.]+\\(?:textwidth|linewidth)", r"\\bibliographystyle\{[^}]*\}", r"\\bibliography\{[^}]*\}"]
ALLOWED = {str(i) for i in range(0, 13)}
# design grid and conventions (not results): group sizes 10/25/50, battery sizes 2.5/5/10 kWh per home, 24 h, 95 % intervals, 100 %
ALLOWED |= {"25", "50", "2.5", "24", "95", "100"}
# metric and design names: 99th percentile, H_90 / 90 % threshold, cap levels 90 / 110 % of last year's peak, price-ratio grid 1.5 / 4
ALLOWED |= {"99", "90", "110", "1.5"}
errors, lits, used = [], [], set()


def lit_spans(s):
    """Remove \\lit{...} with balanced braces; return (text without them, list of contents)."""
    out, found, i = [], [], 0
    while True:
        j = s.find("\\lit{", i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        depth, k = 1, j + 5
        while depth and k < len(s):
            depth += (s[k] == "{") - (s[k] == "}")
            k += 1
        found.append(s[j + 5:k - 1])
        i = k
    return "".join(out), found


for f in FILES:
    if not f.exists():
        continue
    text = f.read_text()
    if "\\begin{document}" in text:
        pre, text = text.split("\\begin{document}", 1)
        text = "\n" * pre.count("\n") + text
    for ln, line in enumerate(text.split("\n"), 1):
        line = re.sub(r"(?<!\\)%.*", "", line)
        for m in re.finditer(r"\\R\{([^}]*)\}", line):
            used.add(m.group(1))
            if m.group(1) not in RK:
                errors.append(f"{f.name}:{ln}: unknown result key {m.group(1)}")
        for m in re.finditer(r"\\K\{([^}]*)\}", line):
            if m.group(1) not in KK:
                errors.append(f"{f.name}:{ln}: unknown config key {m.group(1)}")
        line = re.sub(r"\\[RK]\{[^}]*\}", "", line)
        line, found = lit_spans(line)
        lits += [(f.name, ln, x) for x in found]
        for pat in STRIP:
            line = re.sub(pat, "", line)
        line = re.sub(r"\b\d{1,2}:\d\d\b", "", line)                      # clock times
        line = re.sub(r"\b15--21\b", "", line)                             # name of the 15:00-21:00 demand window
        line = re.sub(r"\b(?:19|20)\d\d(?:--\d\d)?\b", "", line)          # years, year ranges
        for m in re.finditer(r"\d+(?:\\,\d{3})*(?:\.\d+)?", line):
            if m.group(0) not in ALLOWED:
                errors.append(f"{f.name}:{ln}: bare number '{m.group(0)}'  ...{line[max(0, m.start() - 40):m.end() + 25].strip()}...")

rep = [f"files checked: {', '.join(f.name for f in FILES if f.exists())}",
       f"result keys used: {len(used)} (registry: {len(RK)})", f"literals marked \\lit: {len(lits)}", f"errors: {len(errors)}", ""]
rep += ["ERRORS"] + errors + ["", "LITERALS (not results of this study; each must be a quoted, cited or definitional value)"]
rep += [f"{a}:{b}: {c}" for a, b, c in lits]
(ROOT / "results" / "v2_tables" / "number_check.txt").write_text("\n".join(rep) + "\n")
print("\n".join(rep[:4]))
for e in errors[:60]:
    print("  ", e)
sys.exit(1 if errors else 0)
