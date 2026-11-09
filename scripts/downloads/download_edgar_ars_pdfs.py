"""Download official annual report PDFs (Form ARS) from SEC EDGAR.

Since January 2023 companies submit the glossy annual report to shareholders
on EDGAR as a PDF (form type ARS). The 10-K itself is HTML/iXBRL, so ARS is
the only official PDF source. PDFs are stored separately from the
AnnualReports.com corpus:

    data/raw/financial_report_qa/edgar/pdfs/*.pdf
    data/raw/financial_report_qa/edgar/pdf_manifest.jsonl

The manifest uses the same fields as download_financial_report_pdfs.py, so it
can be passed to index_financial_report_pdfs.py --manifest. The fiscal year
comes from the period of the latest 10-K filed on or before the ARS.

SEC requires a User-Agent with contact details and allows 10 requests/second:

    python scripts/downloads/download_edgar_ars_pdfs.py --user-agent "capstone/0.1 you@example.com"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pymupdf


DEFAULT_OUTPUT_DIR = Path("data/raw/financial_report_qa/edgar")
DEFAULT_TARGET_PDFS = 100
DEFAULT_MAX_PER_COMPANY = 4
EARLIEST_FILING_DATE = "2023-01-01"
SLEEP_SECONDS = 0.15
SEC_DEFAULT_USER_AGENT = (
    "capstone-rag-pdf/0.1 "
    "(set --user-agent or SEC_USER_AGENT with your email)"
)

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "WMT",
    "V", "XOM", "UNH", "MA", "ORCL", "COST", "HD", "PG", "NFLX", "JNJ",
    "ABBV", "BAC", "CRM", "KO", "AMD", "CVX", "PEP", "TMO", "ADBE", "WFC",
    "LLY", "MRK", "PFE", "ABT", "AMGN", "CSCO", "INTC", "IBM", "QCOM", "TXN",
    "HON", "CAT", "GE", "MMM", "DE", "LMT", "RTX", "DIS", "MCD", "NKE",
    "SBUX", "LOW", "TGT", "UPS", "GS", "MS", "C", "AXP", "T", "VZ",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)


def http_get(url: str, user_agent: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            time.sleep(SLEEP_SECONDS)
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 503} and attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"unreachable: {url}")


def get_json(url: str, user_agent: str, timeout: int) -> dict:
    return json.loads(http_get(url, user_agent, timeout).decode("utf-8"))


def filing_rows(block: dict) -> list[dict]:
    keys = ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument")
    return [dict(zip(keys, values)) for values in zip(*(block.get(key, []) for key in keys))]


def company_filings(cik: str, user_agent: str, timeout: int) -> tuple[str, list[dict]]:
    """All 10-K and ARS filings since EARLIEST_FILING_DATE, including older submission pages."""
    submissions = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json", user_agent, timeout)
    rows = filing_rows(submissions["filings"]["recent"])
    for extra in submissions["filings"].get("files", []):
        if extra.get("filingTo", "") < EARLIEST_FILING_DATE:
            continue
        rows += filing_rows(get_json(f"https://data.sec.gov/submissions/{extra['name']}", user_agent, timeout))
    wanted = [
        row for row in rows
        if row["form"] in {"10-K", "ARS"} and row["filingDate"] >= EARLIEST_FILING_DATE
    ]
    return submissions.get("name") or cik, wanted


def fiscal_year(ars: dict, filings: list[dict]) -> int | None:
    tenks = [
        row for row in filings
        if row["form"] == "10-K" and row["reportDate"] and row["filingDate"] <= ars["filingDate"]
    ]
    if not tenks:
        return None
    return int(max(tenks, key=lambda row: row["filingDate"])["reportDate"][:4])


def ars_pdf_name(cik: str, ars: dict, user_agent: str, timeout: int) -> str | None:
    if ars["primaryDocument"].lower().endswith(".pdf"):
        return ars["primaryDocument"]
    folder = ars["accessionNumber"].replace("-", "")
    index = get_json(
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/index.json", user_agent, timeout
    )
    pdfs = [item for item in index["directory"]["item"] if item["name"].lower().endswith(".pdf")]
    if not pdfs:
        return None
    return max(pdfs, key=lambda item: int(item.get("size") or 0))["name"]


def pdf_page_count(payload: bytes) -> int:
    with pymupdf.open(stream=payload, filetype="pdf") as document:
        return document.page_count


def existing_keys(manifest_path: Path) -> set[tuple[str, int]]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {(str(row["ticker"]), int(row["year"])) for row in rows}


def write_manifest_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-pdfs", type=int, default=DEFAULT_TARGET_PDFS,
                        help="Total PDFs wanted in the manifest, including ones already downloaded")
    parser.add_argument("--max-per-company", type=int, default=DEFAULT_MAX_PER_COMPANY)
    parser.add_argument("--tickers", nargs="*", help="Optional ticker list, e.g. MSFT JPM PG")
    parser.add_argument("--min-pages", type=int, default=20, help="Skip short wrapper PDFs")
    parser.add_argument("--max-mb", type=float, default=80.0, help="Skip very large PDFs")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT", SEC_DEFAULT_USER_AGENT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "@" not in args.user_agent:
        print("SEC requires contact details: pass --user-agent 'name/0.1 you@example.com' or set SEC_USER_AGENT")
        return 2
    pdf_dir = args.output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "pdf_manifest.jsonl"
    done_keys = existing_keys(manifest_path)
    total = len(done_keys)

    tickers_data = get_json("https://www.sec.gov/files/company_tickers.json", args.user_agent, args.timeout)
    ticker_map = {str(item["ticker"]).upper(): item for item in tickers_data.values()}

    for ticker in [t.upper() for t in args.tickers or TICKERS]:
        if total >= args.target_pdfs:
            break
        item = ticker_map.get(ticker)
        if not item:
            print(f"skip {ticker}: not in SEC ticker list")
            continue
        cik = f"{int(item['cik_str']):010d}"
        try:
            company, filings = company_filings(cik, args.user_agent, args.timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            print(f"skip {ticker}: submissions failed ({exc})")
            continue
        ars_filings = sorted((row for row in filings if row["form"] == "ARS"),
                             key=lambda row: row["filingDate"], reverse=True)
        if not ars_filings:
            print(f"skip {ticker}: no ARS PDF filings since {EARLIEST_FILING_DATE}")
            continue

        saved_for_company = sum(1 for key in done_keys if key[0] == ticker)
        for ars in ars_filings:
            if total >= args.target_pdfs or saved_for_company >= args.max_per_company:
                break
            year = fiscal_year(ars, filings)
            if year is None or (ticker, year) in done_keys:
                continue
            try:
                name = ars_pdf_name(cik, ars, args.user_agent, args.timeout)
                if not name:
                    print(f"skip {ticker} {year}: ARS {ars['accessionNumber']} has no PDF")
                    continue
                folder = ars["accessionNumber"].replace("-", "")
                url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{name}"
                payload = http_get(url, args.user_agent, args.timeout)
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                print(f"skip {ticker} {year}: download failed ({exc})")
                continue
            if not payload.startswith(b"%PDF"):
                print(f"skip {ticker} {year}: not a PDF")
                continue
            if len(payload) > args.max_mb * 1_000_000:
                print(f"skip {ticker} {year}: {len(payload) / 1e6:.0f} MB is over --max-mb")
                continue
            try:
                pages = pdf_page_count(payload)
            except Exception as exc:
                print(f"skip {ticker} {year}: unreadable PDF ({exc})")
                continue
            if pages < args.min_pages:
                print(f"skip {ticker} {year}: only {pages} pages")
                continue

            pdf_path = pdf_dir / safe_filename(f"EDGAR_ARS_{ticker}_{year}.pdf")
            temp_path = pdf_path.with_suffix(".part")
            temp_path.write_bytes(payload)
            temp_path.replace(pdf_path)
            write_manifest_row(manifest_path, {
                "domain": "financial_report_qa",
                "document_id": f"edgar_ars_{ticker}_{year}",
                "company": company,
                "ticker": ticker,
                "cik": cik,
                "year": year,
                "title": f"{company} annual report to shareholders {year}",
                "source_name": "SEC EDGAR",
                "source_url": url,
                "form": "ARS",
                "accession_number": ars["accessionNumber"],
                "filing_date": ars["filingDate"],
                "local_path": str(pdf_path),
                "pages": pages,
                "bytes": len(payload),
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "downloaded_at": utc_now(),
                "source_type": "annual_report_pdf",
            })
            done_keys.add((ticker, year))
            total += 1
            saved_for_company += 1
            print(f"Saved {total:03d}: {ticker} FY{year} ({pages} pages, {len(payload) / 1e6:.1f} MB)")

    print(f"{total} PDFs in {manifest_path}")
    return 0 if total >= args.target_pdfs else 1


if __name__ == "__main__":
    sys.exit(main())
