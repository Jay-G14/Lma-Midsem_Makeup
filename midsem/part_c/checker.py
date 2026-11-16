"""C1.3 answer checker: extracts the final numeric answer from a model reply and compares it with
the gold answer as an exact rational number.

0.05, 1/20, 5E-02, 5e-2, ०.०५, 5%, \\frac{1}{20} and "1 / 20" all parse to Fraction(1, 20), so they
count as the same answer. Floats that are not exactly representable (e.g. 0.3333333333 vs 1/3)
fall back to a relative tolerance of 1e-6.

Extraction order (first that yields a number wins):
  1. the last \\boxed{...}
  2. the text after the last "उत्तर" / "Answer" marker (first number in it)
  3. the last number anywhere in the reply
  4. a Hindi number word (एक ... बीस, सौ) if the reply has no digits at all
"""

import re
import unicodedata
from fractions import Fraction
from typing import Optional

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
UNICODE_FRACTIONS = {"½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4", "⅕": "1/5", "⅛": "1/8"}
HINDI_NUMBER_WORDS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "तीस": 30, "पैंतीस": 35, "सौ": 100,
}

# A number: optional sign, digits with optional thousands commas, optional decimals, optional exponent,
# optionally followed by "/ denominator" (a fraction) or "%".
NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-−]?)(?P<int>\d{1,3}(?:,\d{3})+|\d+)?(?P<dec>\.\d+)?(?:[eE](?P<exp>[-+−]?\d+))?"
    r"(?:\s*/\s*(?P<den>\d+(?:\.\d+)?))?(?P<pct>\s*%)?"
)
FRAC_LATEX = re.compile(r"\\[dt]?frac\s*\{\s*([-−]?[\d.]+)\s*\}\s*\{\s*([-−]?[\d.]+)\s*\}")
ANSWER_MARKERS = re.compile(r"(उत्तर|answer|Answer|ANSWER)")


def normalise(text: str) -> str:
    """NFKC, Devanagari digits to ASCII, unicode fractions and LaTeX \\frac to a/b, markdown removed."""
    text = unicodedata.normalize("NFKC", text).translate(DEVANAGARI_DIGITS)
    for sym, frac in UNICODE_FRACTIONS.items():
        text = text.replace(sym, f" {frac}")
    text = FRAC_LATEX.sub(lambda m: f"{m.group(1)}/{m.group(2)}", text)
    text = text.replace("\\times", "×").replace("\\,", "").replace("$", " ")
    return text.replace("**", " ").replace("−", "-")


def _to_fraction(m: re.Match) -> Optional[Fraction]:
    if m.group("int") is None and m.group("dec") is None:
        return None
    literal = (m.group("int") or "0").replace(",", "") + (m.group("dec") or "")
    if m.group("exp"):
        literal += "e" + m.group("exp").replace("−", "-")
    value = Fraction(literal)
    if m.group("den"):
        den = Fraction(m.group("den"))
        if den == 0:
            return None
        value /= den
    if m.group("pct"):
        value /= 100
    return -value if m.group("sign") in ("-", "−") else value


def numbers_in(text: str):
    """All numbers in normalised text, in order of appearance."""
    out = []
    for m in NUMBER.finditer(text):
        v = _to_fraction(m)
        if v is not None:
            out.append(v)
    return out


def parse_number(text: str) -> Optional[Fraction]:
    """The first number in a short string such as "1/20", "5E-02" or "०.०५"."""
    nums = numbers_in(normalise(text))
    return nums[0] if nums else None


def extract_answer(reply: str) -> Optional[Fraction]:
    """The model's final numeric answer, or None if it gives none."""
    text = normalise(reply)
    boxed = re.findall(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    for candidate in reversed(boxed):
        nums = numbers_in(normalise(candidate))
        if nums:
            return nums[0]
    markers = list(ANSWER_MARKERS.finditer(text))
    if markers:
        nums = numbers_in(text[markers[-1].end():])
        if nums:
            return nums[0]
    nums = numbers_in(text)
    if nums:
        return nums[-1]
    words = [HINDI_NUMBER_WORDS[w.strip("।.,!?:")] for w in text.split() if w.strip("।.,!?:") in HINDI_NUMBER_WORDS]
    return Fraction(words[-1]) if words else None


def same_answer(pred: Optional[Fraction], gold: Fraction, rel_tol: float = 1e-6) -> bool:
    """Exact rational equality, else relative tolerance (for long decimals)."""
    if pred is None:
        return False
    if pred == gold:
        return True
    return abs(float(pred) - float(gold)) <= rel_tol * max(1.0, abs(float(gold)))


def is_correct(reply: str, gold: str) -> bool:
    return same_answer(extract_answer(reply), parse_number(gold))


def _self_test():
    gold = parse_number("0.05")
    for form in ["0.05", "1/20", "5E-02", "5e-2", "०.०५", "5%", "\\frac{1}{20}", "1 / 20", "उत्तर: 0.050", "\\boxed{\\dfrac{1}{20}}"]:
        assert same_answer(extract_answer(form), gold), form
    assert not same_answer(extract_answer("0.5"), gold)
    assert extract_answer("दोनों नल मिलकर 12/5 = 2.4 घंटे में भरेंगे।\nउत्तर: 2.4 घंटे") == Fraction(12, 5)
    assert extract_answer("कुल 1,200 रुपये ब्याज") == 1200
    assert extract_answer("बहन की आयु पाँच वर्ष है।") == 5
    assert extract_answer("हल: 15 - 9 = 6, 6/2 = 3। **उत्तर: 3 किमी/घंटा**") == 3
    assert extract_answer("कोई उत्तर नहीं") is None


_self_test()
