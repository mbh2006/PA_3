"""Ad-hoc check of math-verify parsing with boxed wrapping (development helper)."""

import logging

logging.getLogger("math_verify").setLevel(logging.ERROR)
from math_verify import parse, verify


def P(s):
    return parse("\\boxed{" + s + "}", extraction_mode="first_match", parsing_timeout=None)


cases = [
    ("0.5", "\\frac{1}{2}"),
    ("2", "2.0"),
    ("3", "\\frac{6}{2}"),
    ("\\frac{3}{4}", "0.75"),
    ("x+1", "1+x"),
    ("20", "\\$40"),
    ("5", "6"),
    ("11", "11"),
    ("1,000", "1000"),
    ("2\\sqrt{3}", "\\sqrt{12}"),
]
for a, b in cases:
    pa, pb = P(a), P(b)
    r = bool(verify(pa, pb, timeout_seconds=None) or verify(pb, pa, timeout_seconds=None))
    print(f"{a!r:18} vs {b!r:14} -> {r}\n    parsed {pa[:1]} / {pb[:1]}")
