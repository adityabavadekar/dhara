"""Numeral verifier: every number in a rendered sitrep must come from the facts table.

Works for English and Nepali output. Devanagari digits (०–९) are converted to ASCII before
matching. The allowed numbers are exactly the numerals appearing in the facts' display text and
table cells, so no number the template, or any prose filler, invents can pass.
"""

import html
import re

from dhara.sitrep.facts import ASCII, Facts

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


class SitrepVerificationError(ValueError):
    pass


def numerals(text: str) -> list[str]:
    return [m.group(0).rstrip(",") for m in NUM.finditer(text.translate(ASCII))]


def visible_text(page_html: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page_html, flags=re.S | re.I)
    t = re.sub(r"<img[^>]*>", " ", t, flags=re.I)  # embedded images carry base64, not prose
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(t)


def allowed(f: Facts) -> set[str]:
    out: set[str] = set()
    for fact in f.items.values():
        out.update(numerals(fact.text))
    for rows in f.tables.values():
        for row in rows:
            for v in row.values():
                out.update(numerals(str(v)))
    return out


def verify(page_html: str, f: Facts) -> dict:
    found = numerals(visible_text(page_html))
    ok = allowed(f)
    bad = sorted({n for n in found if n not in ok})
    if bad:
        raise SitrepVerificationError(f"numbers not backed by the facts table: {bad}")
    return {"numerals_checked": len(found), "unique": len(set(found)), "unbacked": []}
