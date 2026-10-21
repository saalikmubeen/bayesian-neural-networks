"""Check that every number in a LaTeX paper is traced in CLAIMS.md.

    python check_claims.py paper/main.tex CLAIMS.md [--min 2] [--ignore 224 256]

Extracts numbers from the paper body (skipping the preamble, comments, \\cite,
\\ref, \\label, \\includegraphics, \\url and environments' options) and reports
those that don't appear anywhere in CLAIMS.md. Numbers are normalised
(1{,}234 -> 1234, 93.9\\% -> 93.9). Small integers (< --min) are ignored, as are
years in citations. Exit code 1 if anything is untraced, so it can gate a
build. It checks provenance, not correctness: CLAIMS.md must itself point to
result files.
"""
import argparse
import re
import sys

NUM = re.compile(r'(?<![\w.])(\d{1,3}(?:\{,\}\d{3})+|\d+(?:\.\d+)?)(?![\w])')
STRIP = [r'%.*', r'\\(?:cite\w*|ref|cref|Cref|label|eqref|includegraphics|url|href|bibliography\w*|graphicspath|usepackage|documentclass|setlength|vspace|hspace)\s*(\[[^\]]*\])?\{[^}]*\}',
         r'\\begin\{[^}]*\}(\[[^\]]*\])?(\{[^}]*\})*', r'\\end\{[^}]*\}', r'width=[^,\]]*', r'\[[a-z!]+\]']


def numbers(text, min_int):
    out = {}
    for line_no, line in enumerate(text.splitlines(), 1):
        clean = line
        for pat in STRIP:
            clean = re.sub(pat, ' ', clean)
        for m in NUM.finditer(clean):
            raw = m.group(1)
            v = raw.replace('{,}', '')
            if '.' not in v and int(v) < min_int:
                continue
            if re.fullmatch(r'(19|20)\d\d', v) and ('cite' in line or 'year' in line):
                continue
            out.setdefault(v, []).append(line_no)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tex')
    ap.add_argument('claims')
    ap.add_argument('--min', type=int, default=2, help='ignore integers below this')
    ap.add_argument('--ignore', nargs='*', default=[], help='values that are settings, not claims (e.g. image sizes)')
    args = ap.parse_args()
    tex = open(args.tex).read()
    body = tex.split('\\begin{document}', 1)[-1]
    offset = tex[:len(tex) - len(body)].count('\n')
    claims = open(args.claims).read().replace(',', '').replace('{,}', '')
    claim_nums = set(re.findall(r'\d+(?:\.\d+)?', claims))
    # also accept equivalent spellings: 0.90 vs 0.9, 93.90 vs 93.9
    claim_norm = {str(float(c)) for c in claim_nums}
    found = numbers(body, args.min)
    missing = {v: ls for v, ls in found.items()
               if v not in claim_nums and str(float(v)) not in claim_norm and v not in args.ignore}
    print(f'{len(found)} distinct numbers in the paper body, {len(found) - len(missing)} traced in {args.claims}')
    if missing:
        print(f'{len(missing)} untraced (value: lines):')
        for v, ls in sorted(missing.items(), key=lambda kv: kv[1][0]):
            print(f'  {v:>10}: {", ".join(str(l + offset) for l in ls[:8])}')
        sys.exit(1)
    print('all numbers traced')


if __name__ == '__main__':
    main()
