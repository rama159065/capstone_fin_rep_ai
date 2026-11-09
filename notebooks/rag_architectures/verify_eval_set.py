"""Check each gold answer is printed in the PDF and each unanswerable question's terms are absent.

Run from the repo root: python notebooks/rag_architectures/verify_eval_set.py <path to PDF>
"""
import hashlib
import re
import sys
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from eval_set_cvx_2024 import EVAL_SET, PDF_SHA256  # noqa: E402

ABSENT = {"X01": ["bitcoin", "cryptocurrenc"], "X02": ["tesla"], "X05": ["15,625"], "X06": ["94,471"], "X08": ["14,979"],
          "X09": ["summary compensation"], "X10": ["charging station"], "X11": ["women", "female"], "X12": ["solar panel"],
          "X13": ["closing price"], "X14": ["lithium"], "X17": ["wind turbine"], "X18": ["coffee"], "X19": ["attrition"]}


def main(path):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == PDF_SHA256, "wrong PDF"
    pages = {i + 1: page.get_text().lower() for i, page in enumerate(pymupdf.open(path))}
    problems = 0
    for row in EVAL_SET:
        if row["type"] == "negative":
            present = sorted({term for term in ABSENT.get(row["id"], []) for text in pages.values() if term in text})
            if present:
                problems += 1
                print("PRESENT", row["id"], present)
            continue
        needles = [row["value"]] if row["type"] == "numeric" else row["keywords"]
        where = [p for p, text in pages.items()
                 if all(re.search(r"(?<![\d,.])" + re.escape(n.lower()) + r"(?!\d)", text) for n in needles)]
        expected = [p for p in row["pages"] if p]
        if not expected:
            print("located", row["id"], where[:8])
        if not where or not set(expected) <= set(where):
            problems += 1
            print("MISSING", row["id"], needles, "expected", expected, "found", where[:12])
    print(f"{len(EVAL_SET)} questions checked, {problems} problems")
    return problems


if __name__ == "__main__":
    sys.exit(bool(main(sys.argv[1])))
