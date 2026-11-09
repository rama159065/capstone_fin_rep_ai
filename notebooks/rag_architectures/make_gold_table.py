"""Write an HTML table of the 66 gold questions, with the passage each answer comes from and each run's result.

Results come from results/results_<notebook>[__<model>].csv, as the notebooks save them.

Run from the repo root:
python notebooks/rag_architectures/make_gold_table.py data/raw/financial_report_qa/edgar/pdfs/EDGAR_ARS_CVX_2024.pdf
"""
import hashlib
import html
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
from eval_set_cvx_2024 import EVAL_SET, PDF_NAME, PDF_SHA256  # noqa: E402
from verify_eval_set import ABSENT  # noqa: E402

OUT = Path(__file__).parent / "gold_questions_cvx_2024.html"
STOP = {"what", "was", "were", "the", "and", "for", "did", "does", "how", "many", "much", "which", "who", "chevron", "chevron's", "in", "of", "to"}
KIND_LABEL = {"table": "Table", "narrative": "Narrative figure", "chart": "Chart", "text": "Text fact", "negative": "Not in report"}
WHY_UNANSWERABLE = {
    "X01": "Not disclosed: the report mentions no cryptocurrency holdings.",
    "X02": "Another company.", "X03": "Another company: the report gives no ExxonMobil results.", "X04": "Another company.",
    "X05": "Year outside the report: income figures cover 2022 to 2024.", "X06": "Year outside the report: revenue figures cover 2022 to 2024.",
    "X07": "Future year: the report covers fiscal 2024.", "X08": "Year outside the report.",
    "X09": "In the proxy statement, not the annual report.", "X10": "Not disclosed in the report.", "X11": "Not disclosed in the report.",
    "X12": "Not disclosed in the report.", "X13": "Not given: the report shows returns, not a closing price.",
    "X14": "Chevron reports no lithium business figures.", "X15": "Personal information, not in the report.",
    "X16": "Forecast: the report gives no future net income.", "X17": "Not disclosed in the report.", "X18": "Not disclosed in the report.",
    "X19": "Not disclosed in the report.", "X20": "Another company: the report gives no Shell results.",
}


def visual_lines(page):
    """Printed lines rebuilt from word positions, so a table row reads left to right."""
    rows = {}
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        rows.setdefault(round((y0 + y1) / 2 / 3), []).append((x0, word))
    return [" ".join(w for _, w in sorted(words)) for _, words in sorted(rows.items())]


def page_title(page):
    """The largest text on the page, usually its heading."""
    spans = [(s["size"], s["text"].strip()) for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
             if len(s["text"].strip()) > 3 and re.search(r"[A-Za-z]{3}", s["text"])]
    if not spans:
        return ""
    biggest = max(size for size, _ in spans)
    return " ".join(text for size, text in spans if size >= biggest - 0.5)[:80]


def printed_number(page):
    """The page number printed in the footer, if there is one."""
    height = page.rect.height
    numbers = [w[4] for w in page.get_text("words") if w[1] > height * 0.9 and re.fullmatch(r"\d{1,3}", w[4])]
    return numbers[-1] if numbers else ""


def matches(text, needle):
    return re.search(r"(?<![\d,.])" + re.escape(needle.lower()) + r"(?!\d)", text.lower())


def best(candidates, question):
    words = {w for w in re.findall(r"[a-z']{3,}", question.lower())} - STOP
    return max(candidates, key=lambda c: (len(words & set(re.findall(r"[a-z']{3,}", c.lower()))), -len(c)), default="")


def evidence(doc, row):
    """The table row, sentence or label on the first gold page that prints the answer."""
    for number in row["pages"]:
        page = doc[number - 1]
        needles = [row["value"]] if row["type"] == "numeric" else row["keywords"]
        if row["kind"] == "table":
            lines = visual_lines(page)
            found = []
            for i, line in enumerate(lines):
                if all(matches(line, n) for n in needles):
                    if not re.search(r"[A-Za-z]{3}", line):  # the label sits on its own baseline above the values
                        label = [l for l in lines[max(0, i - 2):i] if re.search(r"[A-Za-z]{3}", l) and not re.search(r"\d{2}", l)
                                 and not l.startswith("(")]  # a bracketed unit line belongs to the row above
                        line = " ".join(label[-1:] + [line])
                    found.append(line)
            location = "Table row"
        else:
            blocks = [re.sub(r"\s+", " ", b[4]).strip() for b in page.get_text("blocks") if b[6] == 0]
            found = [b for b in blocks if all(matches(b, n) for n in needles)]
            location = {"narrative": "Paragraph", "chart": "Chart label", "text": "Paragraph"}[row["kind"]]
        if found:
            text = best(found, row["question"])
            if row["kind"] == "narrative" and len(text) < 90:
                location = "Stat tile"
            if len(text) > 320:  # keep the part around the answer
                at = matches(text, needles[0]).start()
                text = ("…" if at > 150 else "") + text[max(0, at - 150):at + 170] + "…"
            section = page_title(page)
            chart = re.search(r"\bthe ([^,?]*?(?:graph|chart))", row["question"])
            if row["kind"] == "chart" and chart:
                section = chart.group(1)[0].upper() + chart.group(1)[1:]
            return number, location, text, section, printed_number(page)
    return None, "", "", "", ""


def main(pdf_path):
    data = Path(pdf_path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PDF_SHA256, f"not {PDF_NAME}"
    doc = pymupdf.open(stream=data, filetype="pdf")
    rows, missing = [], []
    for row in EVAL_SET:
        if row["type"] == "negative":
            rows.append(dict(row, answer="Not in the report: the model should refuse.", location="—", passage=WHY_UNANSWERABLE[row["id"]],
                             checked=", ".join(ABSENT.get(row["id"], [])), section="", printed="", found_page=None))
            continue
        page, location, passage, section, printed = evidence(doc, row)
        if not passage:
            missing.append(row["id"])
        answer = re.sub(r"\s*\[p\.\d+\]$", "", row["ground_truth"])
        rows.append(dict(row, answer=answer, location=location, passage=passage, section=section, printed=printed, found_page=page, checked=""))
    write(rows)
    print(f"wrote {OUT} with {len(rows)} questions; passages not found for: {missing or 'none'}")


RUN_TIMES = [  # estimates for a free Colab T4 with Qwen3-8B, scaled from local runs; not measured on Colab
    ("01 BM25 only", "10–15 min", "under 1 min", "25–40 min", "35–55 min"),
    ("02 Dense Chroma", "10–15 min", "1–2 min", "25–40 min", "35–55 min"),
    ("03 BM25 + Chroma + rerank", "10–15 min", "1–2 min", "40–60 min", "50–75 min"),
    ("04 VLM + Chroma", "12–18 min", "45–120 min (vision model reads about 70 pages)", "25–40 min", "1.5–3 h"),
    ("05 CLIP + Chroma", "12–18 min", "2–5 min", "25–40 min", "40–60 min"),
    ("06 VLM + Qdrant", "12–18 min", "2–5 min, reusing 04's transcriptions (or 45–120 min without them)", "25–40 min", "40–65 min (or 1.5–3 h)"),
    ("07 ColPali + Qdrant", "12–18 min", "20–45 min (ColPali page embeddings plus chart transcription)", "30–45 min", "1–1.75 h"),
    ("08 Portal vs RAG", "10–15 min", "under 1 min", "10–20 min", "20–35 min"),
    ("09 OCR + BM25", "10–15 min", "5–15 min (OCR of the chart pages twice, plus every page for the comparison; CPU)", "25–40 min", "40–70 min"),
    ("All nine", "", "", "", "about 7–11 hours"),
]


READINESS = [  # quick test on 8 Nov 2026: real PDF, 6 questions, local 6 GB GPU, transformers 5.22.1, torch 2.14.1
    ("04 VLM + Chroma", "Finished, exit 0", "5 of 6", "83%", "6.6 min"),
    ("05 CLIP + Chroma", "Finished, exit 0", "3 of 6", "100%", "13.8 min"),
    ("06 VLM + Qdrant", "Finished, exit 0", "5 of 6", "83%", "6.3 min (reused 04's transcriptions)"),
    ("07 ColPali + Qdrant", "Finished, exit 0", "6 of 6", "100%", "46 min (18 min of it downloading ColPali; the 6 GB GPU was full)"),
    ("08 Portal vs RAG", "Finished, exit 0, in the earlier full run", "—", "—", "not rerun: only its install line changed"),
]


def colab_guide():
    rows = "".join(f"<tr><td class='id'>{html.escape(name)}</td>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>"
                   for name, *cells in READINESS)
    return f"""<h2>Running notebooks 04–09 on Colab</h2>
<p class="lead">Each notebook is self-contained: the 66 gold questions, the parser, the answer checks and the metrics are inside it, and it imports nothing from the repository. You need only the notebook and the PDF; notebook 08 doesn't use the PDF, it downloads from the SEC.</p>
<table class="times"><thead><tr><th>Notebook</th><th>Quick test</th><th>Number match</th><th>Faithfulness</th><th>Time</th></tr></thead><tbody>{rows}</tbody></table>
<p class="lead" style="margin-top:8px">Checked on 8 Nov 2026 with the real PDF, 6 questions each, on a local 6 GB GPU with transformers 5.22.1 and torch 2.14.1. Six questions confirm the notebook runs; they are not scores. Colab installs the newest packages on the day, so a later release could still change something.</p>
<h3>1. Upload the PDF first</h3>
<p class="lead">Put <code>{PDF_NAME}</code> in Google Drive at <code>My Drive / rag_arch / {PDF_NAME}</code> before running. The folder name is <code>rag_arch</code>, all lowercase, and every notebook uses it. If the PDF isn't there, the notebook shows an upload button partway through the run and saves the file into the folder; uploading first lets you run all cells unattended. The notebook checks the file's fingerprint: with a different PDF it still answers questions but skips the scoring.</p>
<h3>2. Run each notebook</h3>
<ol class="lead">
<li>File → Upload notebook, and pick one of <code>04</code>–<code>09</code> from <code>notebooks/rag_architectures/</code>.</li>
<li>Runtime → Change runtime type → <b>T4 GPU</b>.</li>
<li>In the "Run options" cell keep <code>USE_DRIVE = True</code> and <code>DRIVE_FOLDER = "rag_arch"</code>, and set <code>QUICK_TEST = False</code> for real scores; a quick run overwrites the results file.</li>
<li>Notebook 08 only: set <code>SEC_CONTACT</code> to your name and email. The SEC requires it, and it is sent only to sec.gov. Left blank, the notebook stops and asks for it.</li>
<li>Runtime → Run all, and approve the Google Drive prompt.</li>
</ol>
<p class="lead">Run 04 before 06: 06 reuses the page transcriptions 04 saves in <code>rag_arch</code>, which saves it about 20 minutes. For 08 to compare against 01–03, first copy their CSVs from <code>notebooks/rag_architectures/results/</code> into <code>rag_arch</code>; 08 compares against every results file in that folder. A "set a HF_TOKEN" warning only means slower downloads.</p>
<h3>3. Collect the results</h3>
<p class="lead">Each notebook writes <code>results_&lt;notebook&gt;.csv</code> and <code>.json</code> to <code>rag_arch</code> and adds a row to <code>architecture_scoreboard.csv</code> there. Copy each <code>results_&lt;notebook&gt;.csv</code> into <code>notebooks/rag_architectures/results/</code>, adding <code>__qwen3-8b</code> before <code>.csv</code> if it used that model, then rerun <code>make_gold_table.py</code>. The run appears in "Results so far" and against every question.</p>
<h3>Fixes made while checking</h3>
<ul class="lead">
<li>Notebooks 04, 06 and 07 passed <code>min_pixels</code>/<code>max_pixels</code> to the vision model's processor, and transformers 5.23 removes them. Each page image is now shrunk to the same pixel budget before it is sent (82 × 62 patches, against 80 × 62 before), so it doesn't depend on the transformers version.</li>
<li>The install line had <code>pydantic&gt;=2</code> unquoted. In Colab's <code>!</code> shell the <code>&gt;</code> wrote pip's output to a stray file called <code>=2</code>. It is quoted now.</li>
</ul>"""


RESULTS = Path(__file__).parent / "results"
RUN_NAMES = {"01_bm25_only": "01 BM25", "02_dense_chroma": "02 Dense", "03_bm25_chroma_rerank": "03 Hybrid + rerank",
             "04_vlm_chroma": "04 VLM + Chroma", "05_clip_chroma": "05 CLIP", "06_vlm_qdrant": "06 VLM + Qdrant",
             "07_colpali_qdrant": "07 ColPali + Qdrant", "portal_xbrl": "08 Portal (SEC XBRL)", "09_ocr_bm25": "09 OCR + BM25"}


def load_runs():
    """Every results_<notebook>[__<model>].csv in results/. Without a model suffix the run used Qwen2.5-7B."""
    import pandas as pd
    runs = []
    for path in sorted(RESULTS.glob("results_*.csv")):
        stem = path.stem.removeprefix("results_")
        notebook, _, model = stem.partition("__")
        if notebook == "llm_only":
            continue  # refused all 66 questions under both prompts tried, so it measures the prompt, not the model
        model = model or "qwen2.5-7b"
        name = RUN_NAMES.get(notebook, notebook)
        runs.append({"key": stem, "label": name + ("" if model == "qwen2.5-7b" else f" ({model})"), "model": model,
                     "frame": pd.read_csv(path).set_index("id")})
    return runs


def outcome(frame, qid):
    """A short mark for one question in one run, and the answer it gave as hover text."""
    if qid not in frame.index:
        return ("", ""), ""
    r = frame.loc[qid]
    answered, correct = bool(r["answered"]), bool(r["correct"])
    said = str(r["llm_answer"]) if answered else f"No answer ({r['withheld_reason']})"
    if r["type"] == "negative":
        return ("ok", "refused") if correct else ("bad", "answered"), said
    if correct:
        return ("ok", "right"), said
    return ("bad", "wrong") if answered else ("none", "no answer"), said


def scoreboard(runs):
    head = "".join(f"<th>{html.escape(r['label'])}</th>" for r in runs)
    def share(frame, mask):
        part = frame[mask]
        return f"{part.correct.mean():.1%}" if len(part) else "—"
    lines = [
        ("Number match (40 questions)", lambda f: share(f, f.type == "numeric")),
        ("Table figures (26)", lambda f: share(f, f.kind == "table")),
        ("Narrative figures (6)", lambda f: share(f, f.kind == "narrative")),
        ("Chart figures (8)", lambda f: share(f, f.kind == "chart")),
        ("Text facts (6)", lambda f: share(f, f.type == "text")),
        ("Negative refusal (20), target over 95%", lambda f: share(f, f.type == "negative")),
        ("False refusals (46)", lambda f: f"{(~f[f.type != 'negative'].answered.astype(bool)).mean():.1%}"),
        ("Faithfulness, target over 98%", lambda f: f"{f[f.answered.astype(bool)].faithfulness.dropna().mean():.1%}" if f.faithfulness.notna().any() else "not comparable"),
        ("Median time per question", lambda f: f"{f.total_s.median():.1f} s"),
    ]
    body = "".join(f"<tr><td class='id'>{html.escape(name)}</td>" + "".join(f"<td class='num'>{fn(r['frame'])}</td>" for r in runs) + "</tr>"
                   for name, fn in lines)
    return f"""<h2>Results so far</h2>
<p class="lead">Each run answered the same 66 questions and was scored by the same code. Runs without a model in their name used <code>Qwen/Qwen2.5-7B-Instruct</code>; all of these ran on a local 6 GB GPU. The portal answers only figures tagged in the FY2024 10-K, and code writes its sentence from the fact, so its faithfulness isn't comparable. Each run judges faithfulness with its own model, so the 8B and 7B figures aren't comparable either. Each question's result for each run is in the last column of the question table; hover a mark to see the answer given. To add a run, copy its <code>results_&lt;notebook&gt;.csv</code> into <code>notebooks/rag_architectures/results/</code> (add <code>__qwen3-8b</code> before <code>.csv</code> if it used that model) and rerun <code>make_gold_table.py</code>.</p>
<table class="times"><thead><tr><th>Measure</th>{head}</tr></thead><tbody>{body}</tbody></table>"""


def run_times_table():
    rows = []
    for name, setup, index, evaluation, total in RUN_TIMES:
        strong = name == "All nine"
        cells = "".join(f"<td>{html.escape(c)}</td>" for c in (setup, index, evaluation))
        total_cell = f"<td class='num'><b>{html.escape(total)}</b></td>"
        rows.append(f"<tr{' class=\"total\"' if strong else ''}><td class='id'>{html.escape(name)}</td>{cells}{total_cell}</tr>")
    return f"""<h2>Estimated run time per notebook on Colab</h2>
<p class="lead">For a free Colab T4 GPU with <code>Qwen/Qwen3-8B</code> at 4-bit, all 66 questions and Google Drive on. These are estimates scaled from local runs, not Colab measurements; the vision model's speed in notebooks 04 and 06 is the largest unknown.</p>
<table class="times"><thead><tr><th>Notebook</th><th>Setup (install, download, load)</th><th>Indexing</th><th>66-question evaluation</th><th>Total</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>"""


def write(rows):
    counts = Counter(r["kind"] for r in rows)
    runs = load_runs()
    body = []
    for r in rows:
        marks = []
        for run in runs:
            (style, word), said = outcome(run["frame"], r["id"])
            if style:
                marks.append(f"<span class='mark {style}' title='{html.escape(said, quote=True)}'>{html.escape(run['label'])}: {word}</span>")
        results_cell = f"<td class='results'>{''.join(marks) or '—'}</td>"
        pages = ", ".join(str(p) for p in r["pages"]) or "—"
        printed = f"{r['printed']}" if r.get("printed") else "—"
        source = f"<div class='src'>{html.escape(r['section'])}</div>" if r.get("section") else ""
        note = f"<div class='checked'>Checked absent: {html.escape(r['checked'])}</div>" if r.get("checked") else ""
        quote = f"<blockquote>{html.escape(r['passage'])}</blockquote>" if r["type"] != "negative" else html.escape(r["passage"])
        value = html.escape(r.get("value", "")) + (f" <span class='muted'>× {r['scale']:,}</span>" if r.get("scale", 1) != 1 else "")
        body.append(f"""<tr data-kind="{r['kind']}">
<td class="id">{r['id']}</td><td><span class="tag {r['kind']}">{KIND_LABEL[r['kind']]}</span></td>
<td>{html.escape(r['question'])}</td><td class="answer">{html.escape(r['answer'])}</td><td class="num">{value or '—'}</td>
<td class="num">{pages}</td><td class="num">{printed}</td><td>{html.escape(r['location'])}{source}</td><td class="passage">{quote}{note}</td>{results_cell}</tr>""")
    buttons = "".join(f"<button data-filter='{k}'>{KIND_LABEL[k]} ({counts[k]})</button>" for k in KIND_LABEL)
    OUT.write_text(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gold questions: Chevron 2024 annual report</title>
<style>
:root {{ --ink:#1d2433; --muted:#5b6578; --line:#dde2ea; --head:#f3f5f8; --accent:#1f5fbf; }}
* {{ box-sizing:border-box; }}
body {{ font:14px/1.45 -apple-system, "Segoe UI", Calibri, Arial, sans-serif; color:var(--ink); margin:0; padding:28px; background:#fff; }}
h1 {{ font-size:22px; margin:0 0 4px; }} p.lead {{ color:var(--muted); margin:0 0 18px; max-width:1100px; }}
.summary {{ display:flex; gap:10px; flex-wrap:wrap; margin:0 0 14px; }}
.controls {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:0 0 14px; }}
button {{ border:1px solid var(--line); background:#fff; padding:6px 11px; border-radius:16px; cursor:pointer; font:inherit; color:var(--ink); }}
button.on {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
input {{ padding:7px 10px; border:1px solid var(--line); border-radius:6px; min-width:260px; font:inherit; }}
table {{ border-collapse:collapse; width:100%; }}
th, td {{ border-bottom:1px solid var(--line); padding:8px 9px; text-align:left; vertical-align:top; }}
th {{ background:var(--head); position:sticky; top:0; font-size:12px; text-transform:uppercase; letter-spacing:.03em; color:var(--muted); z-index:1; }}
td.id {{ font-weight:600; white-space:nowrap; }} td.num {{ white-space:nowrap; font-variant-numeric:tabular-nums; }}
td.answer {{ font-weight:600; min-width:150px; }} td.passage {{ max-width:430px; color:#2c3445; }}
blockquote {{ margin:0; padding:0 0 0 9px; border-left:3px solid var(--line); font-family:Consolas, "Courier New", monospace; font-size:12.5px; }}
.src, .checked, .muted {{ color:var(--muted); font-size:12px; margin-top:3px; }}
.tag {{ display:inline-block; padding:2px 8px; border-radius:10px; font-size:12px; white-space:nowrap; background:#eef1f5; }}
.tag.table {{ background:#e3edfb; }} .tag.chart {{ background:#fdebd3; }} .tag.narrative {{ background:#e6f4ea; }}
.tag.text {{ background:#efe7fa; }} .tag.negative {{ background:#f6e3e3; }}
footer {{ color:var(--muted); font-size:12px; margin-top:18px; max-width:1100px; }}
h2 {{ font-size:18px; margin:34px 0 4px; }} h3 {{ font-size:15px; margin:18px 0 4px; }} ol.lead, ul.lead {{ padding-left:20px; }} table.times {{ max-width:1100px; }} table.times th {{ position:static; }}
tr.total td {{ border-top:2px solid var(--ink); background:var(--head); }}
td.results {{ min-width:190px; }} .mark {{ display:block; font-size:12px; padding:1px 7px; margin:0 0 3px; border-radius:8px; white-space:nowrap; cursor:help; }}
.mark.ok {{ background:#e3f3e7; color:#1e6b34; }} .mark.bad {{ background:#f9e0e0; color:#9b2020; }} .mark.none {{ background:#eef1f5; color:var(--muted); }}
@media print {{ .controls {{ display:none; }} th {{ position:static; }} body {{ padding:0; }} }}
</style></head><body>
<h1>Gold questions: Chevron 2024 annual report</h1>
<p class="lead">The {len(rows)} questions used to evaluate RAG notebooks 01–08, with each answer and where it is printed in <code>{PDF_NAME}</code> (116 pages, SEC accession 0001193125-25-076832). Passages are extracted from the PDF by <code>make_gold_table.py</code>, so they show the report's own wording.</p>
{scoreboard(runs) if runs else ""}
<h2>Questions</h2>
<div class="controls"><button class="on" data-filter="all">All ({len(rows)})</button>{buttons}<input id="q" type="search" placeholder="Search questions, answers or passages"></div>
<table id="gold"><thead><tr><th>ID</th><th>Type</th><th>Question</th><th>Answer</th><th>Figure</th><th>PDF page</th><th>Printed page</th><th>Where on the page</th><th>Passage in the report</th><th>Results by run</th></tr></thead>
<tbody>
{chr(10).join(body)}
</tbody></table>
<footer><p><b>How to read it.</b> PDF page is the page in the PDF file; printed page is the number in the page footer, where there is one. Figure is the value as printed; × 1,000,000 means the report prints it in millions. For tables, the passage is the printed row read left to right, and its columns are years in the order the table shows them. For charts, the passage is the printed label; tying it to a series needs the chart itself. For unanswerable questions, the passage column gives the reason, and "checked absent" lists the terms confirmed missing from the whole PDF by <code>verify_eval_set.py</code>.</p></footer>
{colab_guide()}
{run_times_table()}
<script>
const rows = [...document.querySelectorAll("#gold tbody tr")], q = document.getElementById("q");
let kind = "all";
function apply() {{ const t = q.value.toLowerCase();
  rows.forEach(r => r.hidden = !((kind === "all" || r.dataset.kind === kind) && r.textContent.toLowerCase().includes(t))); }}
document.querySelectorAll("button").forEach(b => b.onclick = () => {{
  document.querySelectorAll("button").forEach(x => x.classList.remove("on")); b.classList.add("on"); kind = b.dataset.filter; apply(); }});
q.oninput = apply;
</script></body></html>
""", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
