"""Check the pre-registration hashes in experiments/E*_prereg.txt.

    python scripts/verify_prereg.py

Each protocol was hashed before its experiment ran; its "## Result" section was
filled in afterwards. For a protocol, the check therefore recomputes the hash of
the text above "## Result" followed by the original empty result template.
Other hashed files are compared as they are; shared modules extended by later
experiments are reported as changed (their hash matches the version at that
experiment's launch).
"""

import glob
import hashlib
import re

TEMPLATE = "## Result (filled after running)\n- Outcome:\n- CONFIRMATORY / EXPLORATORY\n- What it rules out / suggests:\n"
sha = lambda b: hashlib.sha256(b).hexdigest()
counts = {"protocol (pre-run text)": 0, "file unchanged": 0, "file changed later": 0, "missing": 0}
for f in sorted(glob.glob("experiments/E*_prereg.txt")):
    for line in open(f):
        m = re.match(r"([0-9a-f]{64})\s+(\S+)", line)
        if not m:
            continue
        h, path = m.groups()
        try:
            data = open(path, "rb").read()
        except FileNotFoundError:
            status = "missing"
        else:
            text = data.decode()
            if path.endswith("protocol.md") and "## Result" in text \
                    and sha((text[:text.index("## Result")] + TEMPLATE).encode()) == h:
                status = "protocol (pre-run text)"
            elif sha(data) == h:
                status = "file unchanged"
            else:
                status = "file changed later"
        counts[status] += 1
        print(f"{f.split('/')[-1]:22s} {path:60s} {status}")
print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()))
