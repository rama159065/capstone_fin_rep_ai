"""Generate the seven standalone Colab notebooks. Everything except the retriever is the same in each one.

Run from anywhere: python notebooks/rag_architectures/build_notebooks.py
"""
import json
from pathlib import Path

from eval_set_cvx_2024 import EVAL_SET, PDF_NAME, PDF_SHA256

HERE = Path(__file__).parent
BASE_PACKAGES = ("langchain-core langchain-community langchain-huggingface langchain-text-splitters "
                 "'pydantic>=2' pymupdf rapidocr onnxruntime sentence-transformers accelerate bitsandbytes pandas")

COLAB_OPTIONS = r'''
# @title Run options
USE_DRIVE = True  # @param {type:"boolean"}
DRIVE_FOLDER = "rag_arch"  # @param {type:"string"}
QUICK_TEST = False  # @param {type:"boolean"}

# With Drive, the PDF, VLM cache, results and scoreboard survive a disconnect and are shared by all seven notebooks.
# A quick test evaluates 6 questions and sends 3 pages to the VLM: it checks the notebook runs, so don't report its scores.
import os
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["RAG_WORK_DIR"] = f"/content/drive/MyDrive/{DRIVE_FOLDER}"
for name, value in {"EVAL_LIMIT": "6", "VLM_MAX_PAGES": "3"}.items():
    if QUICK_TEST:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)
'''

CONFIG = r'''
import gc, glob, hashlib, json, os, re, shutil, statistics, time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, computed_field, field_validator, model_validator

class Settings(BaseModel):
    """Run settings, validated once. Environment variables override the defaults."""
    model_config = ConfigDict(frozen=True)
    work_dir: Path = Path("/content")
    qwen_model: str = "Qwen/Qwen2.5-7B-Instruct"  # "Qwen/Qwen2.5-8B-Instruct" does not exist on Hugging Face
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(5, ge=1, le=20, description="Evidence passages given to Qwen")
    max_context_chars: int = Field(7000, ge=500)
    eval_limit: int = Field(0, ge=0, description="0 runs every gold question")

CFG = Settings(**{field: value for field, value in {"work_dir": os.getenv("RAG_WORK_DIR"), "qwen_model": os.getenv("QWEN_MODEL"),
                                                    "eval_limit": os.getenv("EVAL_LIMIT")}.items() if value})
CFG.work_dir.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if DEVICE == "cuda" else 0
print(f"Device: {DEVICE}, VRAM: {VRAM_GB:.1f} GB")
if DEVICE == "cpu":
    print("WARNING: no GPU. In Colab choose Runtime > Change runtime type > T4 GPU; Qwen will not load in 4-bit without it.")

def locate_pdf():
    if os.getenv("PDF_PATH"):
        return Path(os.environ["PDF_PATH"])
    found = sorted(glob.glob(str(CFG.work_dir / "*.pdf")))
    expected = [p for p in found if Path(p).name == EXPECTED_PDF]
    if expected or found:
        return Path((expected or found)[0])
    from google.colab import files
    print(f"Upload the annual report PDF ({EXPECTED_PDF} for the gold evaluation).")
    uploaded = files.upload()
    name = next(n for n in uploaded if n.lower().endswith(".pdf"))
    return Path(shutil.move(name, CFG.work_dir / name)) if Path(name).resolve().parent != CFG.work_dir.resolve() else CFG.work_dir / name

PDF_PATH = locate_pdf()
PDF_SHA = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
GOLD_VALID = PDF_SHA == EXPECTED_SHA256
print(f"PDF: {PDF_PATH} ({PDF_PATH.stat().st_size / 1e6:.1f} MB)")
print("Gold evaluation applies to this PDF." if GOLD_VALID else
      "This is not the PDF the gold questions were written for: you can ask your own questions, but the evaluation cell will be skipped.")
'''

PARSER = r'''
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

NUMERIC_CELL = re.compile(r"^(\$\s*)?\(?-?\d[\d,]*(\.\d+)?\)?\s*%?$|^[—–-]$")
YEAR_CELL = re.compile(r"^(19|20)\d{2}$")
UNIT_LINE = re.compile(r"(millions? of|billions? of|thousands? of)[^|]{0,50}", re.I)
_ocr_engine = None

class TableSegment(BaseModel):
    header: str = ""
    unit: str = ""
    rows: list[str] = Field(min_length=1)

class ParsedPage(BaseModel):
    page: int = Field(ge=1)
    narrative: str = ""
    tables: list[TableSegment] = Field(default_factory=list)
    ocr_used: bool = False
    visual: bool = False

class ChunkMetadata(BaseModel):
    page: int = Field(ge=1)
    kind: Literal["table", "text", "page"]
    ocr_used: bool = False
    chunk_id: int = -1

def chunk(text, page, kind, ocr_used=False):
    return Document(page_content=text, metadata=ChunkMetadata(page=page, kind=kind, ocr_used=ocr_used).model_dump())

def looks_garbled(text):
    """A broken font map extracts as control characters and symbols, which is not an empty page."""
    if not text.strip():
        return False
    readable = sum(c.isalnum() or c in " \n\t.,$%()-:;'\"/&’–—‑" for c in text) / len(text)
    return readable < 0.85 or len(re.findall(r"[A-Za-z]{3,}", text)) < len(text) / 40

def ocr_words(page):
    """RapidOCR lines as (x0, y0, x1, y1, text), the same shape as PyMuPDF words."""
    global _ocr_engine
    import numpy as np
    from rapidocr import RapidOCR
    _ocr_engine = _ocr_engine or RapidOCR(params={"Global.log_level": "error"})
    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False)
    result = _ocr_engine(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
    if result.boxes is None:
        return []
    ratio = 72 / 200
    return [(min(p[0] for p in box) * ratio, min(p[1] for p in box) * ratio, max(p[0] for p in box) * ratio,
             max(p[1] for p in box) * ratio, text) for box, text, score in zip(result.boxes, result.txts, result.scores) if score >= 0.6]

def layout_rows(words):
    """Rebuild printed lines from word positions; a wide gap starts a new cell, so table columns survive."""
    words = sorted(words, key=lambda w: (round(w[3]), w[0]))
    lines, seen = [], set()
    for word in words:
        key = (word[4], round(word[0] / 3), round(word[3] / 3))  # bold is often printed twice at the same spot
        if key in seen:
            continue
        seen.add(key)
        if lines and abs(lines[-1][0] - word[3]) < 3.5:
            lines[-1][1].append(word)
        else:
            lines.append([word[3], [word]])
    rows = []
    for _, line in lines:
        line.sort(key=lambda w: w[0])
        cells, current = [], [line[0]]
        for word in line[1:]:
            if word[0] - current[-1][2] > 9:
                cells.append(current)
                current = [word]
            else:
                current.append(word)
        cells.append(current)
        joined = [" ".join(w[4] for w in cell).strip() for cell in cells]
        rows.append([c for c in joined if c and c != "$"])
    return rows

def is_table_row(cells):
    return sum(bool(NUMERIC_CELL.match(c)) for c in cells) >= 2

def table_segments(rows):
    """Group table rows; keep the year header and unit with each table so a row can never lose its columns."""
    segments, current, header, unit, gap, previous = [], [], "", "", 0, None
    for cells in rows:
        text = " | ".join(cells)
        if current and " | " in current[-1] and re.fullmatch(r"\([A-Za-z ,-]+\)", text):
            label, values = current[-1].split(" | ", 1)
            current[-1] = f"{label} {text} | {values}"
            continue
        match = UNIT_LINE.search(text)
        if match and not is_table_row(cells):
            unit = match.group(0).strip(" )")
        if sum(bool(YEAR_CELL.match(c)) for c in cells) >= 2:
            header = " | ".join(cells)
            previous = None
            continue
        if is_table_row(cells):
            # A two-line label sits on a different baseline from its values: join them so the row keeps its name.
            if not any(re.search(r"[A-Za-z]{3}", c) for c in cells):
                if current and not re.search(r"\d", current[-1]) and " | " not in current[-1]:
                    text = f"{current.pop()} | {text}"
                elif not current and previous and not is_table_row(previous) and len(" ".join(previous)) < 75:
                    text = f"{' '.join(previous)} | {text}"
            current.append(text)
            gap = 0
        elif current and len(text) < 75 and gap < 2:
            current.append(text)  # sub-labels and group headings inside a table
            gap += 1
        elif current:
            segments.append(TableSegment(header=header, unit=unit, rows=current))
            current, gap = [], 0
        previous = cells
    if current:
        segments.append(TableSegment(header=header, unit=unit, rows=current))
    return [s for s in segments if sum(" | " in r and bool(re.search(r"[A-Za-z]{3}", r)) for r in s.rows) >= 2]

def has_visual(page, text):
    area = page.rect.width * page.rect.height
    picture = any(0.05 < r.width * r.height / area < 0.85 for image in page.get_images(full=True) for r in page.get_image_rects(image[0]))
    fills = [d for d in page.get_drawings() if d.get("fill") and d["rect"].width * d["rect"].height > 30]
    return picture or (15 <= len(fills) <= 900 and len(re.findall(r"\d", text)) > 10 and len(text) < 3500)

def parse_pdf(path):
    pages = []
    with pymupdf.open(path) as doc:
        for number, page in enumerate(doc, start=1):
            raw = page.get_text()
            ocr_used = len(raw.strip()) < 100 or looks_garbled(raw)
            words = ocr_words(page) if ocr_used else [w[:5] for w in page.get_text("words")]
            rows = layout_rows(words) if words else []
            segments = table_segments(rows)
            in_table = {cell for s in segments for row in s.rows for cell in row.split(" | ")}
            if ocr_used:
                narrative = "\n".join(" ".join(r) for r in rows if not is_table_row(r))
            else:
                blocks = [b[4] for b in sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0])) if b[6] == 0]
                narrative = "\n".join(x.strip() for b in blocks for x in b.splitlines()
                                      if x.strip() and not (segments and (x.strip() in in_table or NUMERIC_CELL.match(x.strip()) or x.strip() == "$")))
            narrative = re.sub(r"[ \t]+", " ", narrative).strip()
            pages.append(ParsedPage(page=number, narrative=narrative, tables=segments, ocr_used=ocr_used, visual=has_visual(page, raw)))
    return pages

def linear_rows(segment):
    """Write each value beside its year and each row under its group, so nothing depends on counting columns."""
    cells = segment.header.split(" | ")
    first_year = next((i for i, cell in enumerate(cells) if YEAR_CELL.match(cell)), None)
    years = cells[first_year:] if first_year is not None else []
    if len(years) < 2 or len(set(years)) != len(years) or not all(YEAR_CELL.match(y) for y in years):
        return segment.rows  # no year columns, or repeated ones such as one set per segment: leave as printed
    rows, group = [], ""
    for row in segment.rows:
        parts = row.split(" | ")
        split_label = re.fullmatch(r"(\$?\s*\(?[\d,.]+\)?)\s+(\(.+\))", parts[0])
        if split_label:  # a value that landed on the same baseline as its unit, under a two-line label
            parts = [f"{group} {split_label.group(2)}".strip(), split_label.group(1), *parts[1:]]
            group = ""
        if len(parts) == 1:
            group = row if not re.search(r"\d\s*$", row) or re.search(r"[A-Za-z]\d$", row) else group
            rows.append(row)
            continue
        values = parts[-len(years):]
        label = " ".join(parts[:-len(years)]).strip()
        if len(parts) <= len(years) or not all(NUMERIC_CELL.match(v) for v in values):
            rows.append(row)
            continue
        named = f"{group} > {label}" if group and not label.lower().startswith(("total", group.lower())) and not label.startswith("(") else label
        rows.append(f"{named}: " + "; ".join(f"{y}={v}" for y, v in zip(years, values)))
        if label.lower().startswith("total"):
            group = ""
    return rows

def table_chunks(segment, max_chars=1400):
    """Split long tables between rows, repeating the column header and unit in every piece."""
    head = "\n".join(x for x in [f"Units: {segment.unit}" if segment.unit else "",
                                 f"Columns: {segment.header}" if segment.header else ""] if x)
    pieces, current = [], []
    for row in linear_rows(segment):
        if current and sum(map(len, current)) + len(head) + len(row) > max_chars:
            pieces.append(head + "\n" + "\n".join(current))
            current = []
        current.append(row)
    if current:
        pieces.append(head + "\n" + "\n".join(current))
    return [p.strip() for p in pieces]

def make_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "])
    chunks = []
    for item in pages:
        for segment in item.tables:
            chunks += [chunk(text, item.page, "table", item.ocr_used) for text in table_chunks(segment)]
        if len(item.narrative) >= 40:
            chunks += [chunk(text, item.page, "text", item.ocr_used) for text in splitter.split_text(item.narrative)]
    for index, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = index
    return chunks

def page_text(item):
    tables = "\n\n".join("\n".join(table_chunks(s, 10**6)) for s in item.tables)
    return "\n\n".join(x for x in [tables, item.narrative] if x)

started = time.time()
PAGES = parse_pdf(PDF_PATH)
CHUNKS = make_chunks(PAGES)
print(f"Parsed {len(PAGES)} pages into {len(CHUNKS)} chunks in {time.time() - started:.0f}s: "
      f"{sum(p.ocr_used for p in PAGES)} pages needed OCR, {sum(bool(p.tables) for p in PAGES)} have tables, "
      f"{sum(p.visual for p in PAGES)} have pictures or charts.")
print(next(c.page_content for c in CHUNKS if c.metadata["kind"] == "table")[:400])
'''

BM25_CODE = r'''
from langchain_community.retrievers import BM25Retriever

STOPWORDS = set("a an and are at by did do does for from had has have how in is it its of on or per the to was were what when which who with".split())

def bm25_tokens(text):
    """Keep figures whole ("17,661", "10.4") so exact-number and exact-term search works."""
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,'&-][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]
'''

CHROMA_CODE = r'''
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDINGS = HuggingFaceEmbeddings(model_name=CFG.embedding_model, model_kwargs={"device": DEVICE},
                                   encode_kwargs={"normalize_embeddings": True})

def build_chroma(chunks, name):
    """A fresh in-memory collection, so rerunning a cell can never mix old and new chunks."""
    store = Chroma(collection_name=f"{name}_{time.time_ns()}", embedding_function=EMBEDDINGS, collection_metadata={"hnsw:space": "cosine"})
    store.add_documents(chunks, ids=[str(c.metadata["chunk_id"]) for c in chunks])
    return store
'''

RERANK_CODE = r'''
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

RERANK_FLOOR = -4.0  # raw logit; the best hit must clear it or the question is treated as out of scope
CROSS_ENCODER = HuggingFaceCrossEncoder(model_name=CFG.reranker_model, model_kwargs={"device": "cpu"})

def rrf_merge(result_lists, k=60):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc.metadata["chunk_id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
            docs[key] = doc
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]

def rerank(query, docs, top_k=CFG.top_k):
    """Score (question, passage) pairs. Judge only the best score, so a low-scoring table beside a good hit stays."""
    if not docs:
        return []
    scores = [float(s) for s in CROSS_ENCODER.score([(query, d.page_content) for d in docs])]
    if max(scores) < RERANK_FLOOR:
        return []
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)[:top_k]
    return [Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": s}) for s, d in ranked]
'''

QDRANT_CODE = r'''
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
'''

VLM_CODE = r'''
import base64, io
from PIL import Image

VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLM_CACHE = CFG.work_dir / f"vlm_pages_{PDF_SHA[:12]}.jsonl"   # shared by notebooks 04 and 06
VLM_MAX_NEW_TOKENS = 1200
VLM_MAX_PIXELS = 1280 * 28 * 28  # about 1,280 visual tokens per page
class VlmPage(BaseModel):
    """One cached transcription. A line that fails validation is transcribed again."""
    page: int = Field(ge=1)
    markdown: str

VLM_PROMPT = """Transcribe this annual report page into Markdown. Copy every number exactly as printed.
- Tables: a Markdown table with the printed column headers. Keep the unit line. Keep every row.
- Charts: state the chart title, then one line per labelled value saying which series, bar or year it belongs to, using the legend colours. Never estimate an unlabelled point.
- Text: copy headings and paragraphs. Ignore photographs and QR codes.
Output only the transcription."""

def vlm_targets(pages, limit=int(os.getenv("VLM_MAX_PAGES", "0"))):
    """Pages with a table, chart or picture go to the VLM; plain narrative keeps its accurate, free text layer."""
    targets = [p.page for p in pages if p.tables or p.visual or p.ocr_used]
    return targets[:limit] if limit else targets

def load_vlm():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_MODEL, quantization_config=quant, device_map={"": 0})  # 4-bit weights must stay on the GPU
    return model, AutoProcessor.from_pretrained(VLM_MODEL)

def transcribe_page(model, processor, doc, number, prompt=VLM_PROMPT):
    from qwen_vl_utils import process_vision_info
    image = Image.open(io.BytesIO(doc[number - 1].get_pixmap(dpi=150, colorspace=pymupdf.csRGB, alpha=False).tobytes("png")))
    scale = (VLM_MAX_PIXELS / (image.width * image.height)) ** 0.5
    if scale < 1:  # shrink here: the processor's min_pixels/max_pixels options are removed in transformers 5.23
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=VLM_MAX_NEW_TOKENS, do_sample=False)
    return processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

def run_vlm(targets, cache=VLM_CACHE, prompt=VLM_PROMPT):
    """Resumable: a Colab disconnect only loses the page in progress."""
    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            try:
                row = VlmPage.model_validate_json(line)
                done[row.page] = row.markdown
            except ValidationError:
                pass  # a line cut off by a disconnect
    todo = [n for n in targets if n not in done]
    print(f"VLM: {len(done)} pages cached, {len(todo)} to transcribe with {VLM_MODEL}.")
    if todo:
        model, processor = load_vlm()
        with pymupdf.open(PDF_PATH) as doc, cache.open("a", encoding="utf-8") as sink:
            for index, number in enumerate(todo, start=1):
                started = time.time()
                done[number] = transcribe_page(model, processor, doc, number, prompt)
                sink.write(VlmPage(page=number, markdown=done[number]).model_dump_json() + "\n")
                sink.flush()
                print(f"  page {number} ({index}/{len(todo)}): {len(done[number])} chars in {time.time() - started:.0f}s")
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    return done

def markdown_chunks(number, markdown):
    """Split transcribed Markdown, keeping table headers on every piece."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". ", " "])
    chunks, prose, lines, i = [], [], markdown.splitlines(), 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            head = lines[i] + "\n" + lines[i + 1]
            i += 2
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(lines[i])
                i += 1
            before = " ".join(prose[-2:])[:150]
            for start in range(0, len(rows), 12):
                chunks.append(chunk(f"{before}\n{head}\n" + "\n".join(rows[start:start + 12]), number, "table"))
        else:
            prose.append(lines[i])
            i += 1
    text = "\n".join(prose).strip()
    if len(text) >= 40:
        chunks += [chunk(t, number, "text") for t in splitter.split_text(text)]
    return chunks

def vlm_chunks(pages, transcripts):
    chunks = []
    for item in pages:
        if item.page in transcripts:
            chunks += markdown_chunks(item.page, transcripts[item.page])
        else:
            chunks += make_chunks([item])
    for index, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = index
    return chunks

def decimal_drift_report(pages, transcripts):
    """The VLM can misread digits. List figures it wrote that the page's own text layer doesn't print."""
    drift = {}
    for item in pages:
        markdown = transcripts.get(item.page)
        if not markdown or item.ocr_used:
            continue
        printed = set(re.findall(r"\d[\d,]*\.?\d*", page_text(item)))
        unseen = sorted(set(re.findall(r"\d[\d,]{2,}\.?\d*", markdown)) - printed)
        if unseen:
            drift[item.page] = unseen
    total = sum(map(len, drift.values()))
    print(f"Decimal drift check: {total} figures on {len(drift)} pages are not in the text layer.")
    for number, values in list(drift.items())[:10]:
        print(f"  page {number}: {values[:10]}")
    return drift
'''

ANSWER = r'''
from functools import partial

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(CFG.qwen_model)
if "qwen3" in CFG.qwen_model.lower():  # thinking would break the JSON reply and use up the token limit
    tokenizer.apply_chat_template = partial(tokenizer.apply_chat_template, enable_thinking=False)
model = AutoModelForCausalLM.from_pretrained(CFG.qwen_model, quantization_config=quant, device_map={"": 0})  # 4-bit weights must stay on the GPU
# No repetition_penalty: it pushes the model away from digits already in the context, which corrupts copied figures.
text_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=140, do_sample=False,
                     return_full_text=False, pad_token_id=tokenizer.eos_token_id)
for attribute in ("temperature", "top_p", "top_k"):
    setattr(model.generation_config, attribute, None)
CHAT_LLM = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=text_pipe), model_id=CFG.qwen_model, tokenizer=tokenizer)
print(f"Loaded {CFG.qwen_model}; GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
'''

GOLD_MODELS = r'''
class GoldBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str
    question: str = Field(min_length=5)
    ground_truth: str = Field(min_length=1)
    pages: list[int] = Field(default_factory=list)

class NumericGold(GoldBase):
    type: Literal["numeric"]
    kind: Literal["table", "narrative", "chart"]
    value: str
    scale: int = Field(1, gt=0)

    @field_validator("value")
    @classmethod
    def value_is_a_number(cls, value):
        Decimal(value.replace(",", ""))
        return value

    @property
    def decimal(self):
        return Decimal(self.value.replace(",", ""))

class TextGold(GoldBase):
    type: Literal["text"]
    kind: Literal["text"]
    keywords: list[str] = Field(min_length=1)

class NegativeGold(GoldBase):
    type: Literal["negative"]
    kind: Literal["negative"]

GoldQuestion = Annotated[Union[NumericGold, TextGold, NegativeGold], Field(discriminator="type")]
'''

GROUNDING = r'''
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

class LLMAnswer(BaseModel):
    """The only output accepted from Qwen. Anything else fails closed."""
    in_report: bool = Field(description="True only if the evidence states the item for the asked company and year")
    answer: Optional[str] = Field(None, description="One short sentence with the figure as printed")
    cited_pages: list[int] = Field(default_factory=list, description="PDF pages from the evidence labels")

    @model_validator(mode="after")
    def answer_matches_flag(self):
        if self.in_report and not (self.answer or "").strip():
            raise ValueError("in_report is true but there is no answer")
        if not self.in_report:
            self.answer, self.cited_pages = None, []
        return self

class GateDecision(BaseModel):
    answer: Optional[str] = None
    reason: str = "ok"
    cited_pages: list[int] = Field(default_factory=list)

Status = Literal["LLM_ANSWER", "GROUND_TRUTH_FALLBACK"]
Source = Literal["llm", "gold_set", "closest_passage", "none"]

class GroundedResult(BaseModel):
    question: str
    status: Status
    final_answer: str
    final_source: Source
    llm_answer: Optional[str] = None
    llm_raw: str = ""
    withheld_reason: str = ""
    cited_pages: list[int] = Field(default_factory=list)
    retrieved_pages: list[int] = Field(default_factory=list)
    context: str = ""
    retrieval_s: float = Field(ge=0)
    total_s: float = Field(ge=0)

    @model_validator(mode="after")
    def fallback_is_never_llm(self):
        if (self.status == "LLM_ANSWER") != (self.final_source == "llm") or (self.status == "LLM_ANSWER") != bool(self.llm_answer):
            raise ValueError("status, source and llm_answer disagree")
        return self

SYSTEM_PROMPT = """You answer questions about one company's annual report using ONLY the evidence passages.

Reply with one JSON object and nothing else, with these keys:
"in_report": true or false
"answer": one short sentence, or null
"cited_pages": the page numbers from the evidence labels, such as [12]

Rules:
1. Copy figures exactly as printed, with the unit shown in the evidence. Do not calculate, round or estimate.
2. In a table, take the value from the row that matches the item, and within that row the value after the year asked about, such as 2023=388. "Group > Item" means the item belongs to that group.
3. The evidence must state the item for the same company and the same year as the question. If it does not, set "in_report" to false and "answer" to null. Never substitute another year, company or similar item.

Examples:
Evidence: [p.4] Units: Millions of dollars
Total widgets revenue: 2024=410; 2023=388
Question: What was widgets revenue in 2023?
Answer: {{"in_report": true, "answer": "Widgets revenue was $388 million in 2023.", "cited_pages": [4]}}

Evidence: [p.4] Units: Millions of dollars
Total widgets revenue: 2024=410; 2023=388
Question: What was widgets revenue in 2019?
Answer: {{"in_report": false, "answer": null, "cited_pages": []}}

Evidence: [p.9] The board includes the chief executive of Globex Inc.
Question: What was Globex Inc.'s revenue?
Answer: {{"in_report": false, "answer": null, "cited_pages": []}}"""

ANSWER_CHAIN = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT), ("human", "Evidence:\n{context}\n\nQuestion: {question}\nAnswer:"),
]) | CHAT_LLM | StrOutputParser()
ANSWER_PARSER = PydanticOutputParser(pydantic_object=LLMAnswer)

CITATION = re.compile(r"\[?\(?\b(?:p\.|pp\.|page)\s*(\d+)\)?\]?", re.I)
NUMBER = re.compile(r"(?<![\w.,])\(?\$?\(?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\)?\s*(%|thousand|million|billion|trillion)?", re.I)
SCALES = {"thousand": Decimal(10) ** 3, "million": Decimal(10) ** 6, "billion": Decimal(10) ** 9, "trillion": Decimal(10) ** 12}
REFUSAL_WORDS = re.compile(r"\b(does|do|did) not (contain|provide|include|mention|state|specify|disclose|show)\b|"
                           r"\bnot (provided|available|mentioned|disclosed|stated|specified|found|included)\b|\bno (information|data|mention)\b|"
                           r"\bcannot (be )?(determine|determined|answer|find)\b|\bunable to\b", re.I)

def numbers_in(text):
    """(digits as printed, value without scale, scale). Bare years and page citations are not figures."""
    found = []
    for match in NUMBER.finditer(CITATION.sub(" ", text)):
        printed = match.group(1)
        value = Decimal(printed.replace(",", ""))
        if "," not in printed and "." not in printed and 1900 <= value <= 2100 and not match.group(2):
            continue
        found.append((printed, value, SCALES.get((match.group(2) or "").lower(), Decimal(1))))
    return found

def same_figure(printed, value, scale, gold, gold_scale=Decimal(1)):
    """Exact, or a correct rounding to the printed decimals with at least 3 significant digits: $17.7 billion matches 17,661 million."""
    if value == gold:
        return True
    target = gold * gold_scale / scale
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    if value == target:
        return True
    significant = len(printed.replace(",", "").replace(".", "").lstrip("0"))
    return significant >= 3 and scale != Decimal(1) and target.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP) == value

def figure_in_evidence(printed, value, scale, evidence_numbers):
    for e_printed, e_value, e_scale in evidence_numbers:
        if same_figure(printed, value, Decimal(1), e_value) or same_figure(printed, value, scale, e_value, e_scale):
            return True
        for power in (3, 6, 9):  # evidence in millions, answer in billions and the like
            if same_figure(printed, value, scale, e_value, Decimal(10) ** power):
                return True
    return False

QUESTION_STOPWORDS = {"what", "was", "were", "which", "who", "how", "many", "much", "did", "does", "the", "and", "for", "from", "with", "that",
                      "this", "has", "had", "its", "are", "chevron", "corporation", "company", "report", "year", "end", "per", "day", "total"}

def content_words(text):
    return {word.rstrip("s") for word in re.findall(r"[a-z]{3,}", text.lower()) if word not in QUESTION_STOPWORDS}

def table_cells(line):
    return [cell.strip() for cell in line.replace("Columns:", "").split("|") if cell.strip()]

def year_of(cell):
    years = re.findall(r"\b(?:19|20)\d{2}\b", cell)
    return years[-1] if len(years) == 1 and len(cell) < 30 else None

def in_year_column(header, cells, figure, year):
    """Values align with the header from the right. If the counts differ the row can't be checked, so it isn't blocked."""
    printed, value, scale = figure
    columns = header[next(i for i, cell in enumerate(header) if year_of(cell)):]
    values = [cell for cell in cells if re.fullmatch(r"[$(\s]*[\d,.]+[)%\s]*", cell)]
    if len(values) != len(columns):
        return True
    return any(year_of(column) == year and figure_in_evidence(printed, value, scale, numbers_in(cell))
               for column, cell in zip(columns, values))

def figure_placement(question, figure, docs):
    """Printed somewhere is not enough: the figure must sit beside the asked item and, in a table, under the asked year."""
    printed, value, scale = figure
    words = content_words(question)
    years = re.findall(r"\b(?:19|20)\d{2}\b", question)
    year = years[0] if len(set(years)) == 1 else None
    outcome = "figure_not_beside_item"
    for doc in docs:
        lines, header = doc.page_content.splitlines(), None
        for index, line in enumerate(lines):
            cells = table_cells(line)
            if sum(bool(year_of(cell)) for cell in cells) >= 2:
                header = cells
                continue
            if not figure_in_evidence(printed, value, scale, numbers_in(line)):
                continue
            nearby = " ".join(lines[max(0, index - 2):index + 2])
            if words and not content_words(nearby) & words:
                continue
            pairs = re.findall(r"\b((?:19|20)\d{2})=([^;]+)", line)
            if year and pairs and not any(y == year and figure_in_evidence(printed, value, scale, numbers_in(v)) for y, v in pairs):
                outcome = "figure_in_other_year_column"
                continue
            if year and header and len(cells) > 1 and not pairs and not in_year_column(header, cells, figure, year):
                outcome = "figure_in_other_year_column"
                continue
            return "ok"
    return outcome

def evidence_block(docs):
    parts, used = [], 0
    for doc in docs:
        part = f"[p.{doc.metadata['page']}] {doc.page_content}"
        if used + len(part) > CFG.max_context_chars:
            part = part[:max(0, CFG.max_context_chars - used)]
        if part:
            parts.append(part)
            used += len(part)
    return "\n\n---\n\n".join(parts)

def gate_answer(question, raw, docs):
    """Parse against the schema, then run deterministic checks on the result."""
    try:
        parsed = ANSWER_PARSER.parse(raw or "")
    except OutputParserException:
        return GateDecision(reason="invalid_llm_output")
    if not parsed.in_report:
        return GateDecision(reason="llm_refused")
    text = parsed.answer.strip()
    if REFUSAL_WORDS.search(text):
        return GateDecision(reason="llm_refused")
    pages = {d.metadata["page"] for d in docs}
    context = "\n".join(d.page_content for d in docs)
    for year in re.findall(r"\b(?:19|20)\d{2}\b", question):
        if year not in context:
            return GateDecision(reason=f"year_{year}_not_in_evidence")
    evidence_numbers = numbers_in(context) + numbers_in(question)
    asked = numbers_in(question)
    for figure in numbers_in(text):
        printed, value, scale = figure
        if not figure_in_evidence(printed, value, scale, evidence_numbers):
            return GateDecision(reason=f"unsupported_figure_{printed}")
        if not figure_in_evidence(printed, value, scale, asked):
            placement = figure_placement(question, figure, docs)
            if placement != "ok":
                return GateDecision(reason=placement)
    cited = sorted(set(parsed.cited_pages) | {int(n) for n in CITATION.findall(text)})
    if any(page not in pages for page in cited):
        return GateDecision(reason="citation_outside_evidence")
    if not cited:  # cite the first passage printing the answer's figure, otherwise the top passage
        figures = [p for p, _, _ in numbers_in(text)]
        cited = [next((d for d in docs if any(f in d.page_content for f in figures)), docs[0]).metadata["page"]]
    if not CITATION.search(text):
        text = f"{text.rstrip('.')} " + " ".join(f"[p.{page}]" for page in cited) + "."
    return GateDecision(answer=text, cited_pages=cited)

def ground_truth_for(question, docs):
    """Used only when the LLM gives no answer. Never counted as an LLM answer in the metrics."""
    gold = GOLD_BY_QUESTION.get(question.strip().lower()) if GOLD_VALID else None
    if gold:
        return gold.ground_truth, "gold_set"
    if docs:
        doc = docs[0]
        return f"The model gave no verified answer. Closest passage, not a confirmed answer [p.{doc.metadata['page']}]: {doc.page_content[:500]}", "closest_passage"
    return "No verified information on this was found in the report.", "none"

def ask(question):
    started = time.time()
    docs = RETRIEVER.invoke(question)
    retrieved = time.time()
    raw, decision = "", GateDecision(reason="no_evidence")
    context = evidence_block(docs)
    if docs:
        raw = ANSWER_CHAIN.invoke({"context": context, "question": question})
        decision = gate_answer(question, raw, docs)
    if decision.answer:
        final, source, status = decision.answer, "llm", "LLM_ANSWER"
    else:
        final, source = ground_truth_for(question, docs)
        status = "GROUND_TRUTH_FALLBACK"
    return GroundedResult(question=question, status=status, final_answer=final, final_source=source, llm_answer=decision.answer,
                          llm_raw=raw, withheld_reason="" if decision.answer else decision.reason, cited_pages=decision.cited_pages,
                          retrieved_pages=[d.metadata["page"] for d in docs], context=context,
                          retrieval_s=round(retrieved - started, 3), total_s=round(time.time() - started, 3))

def show(result):
    print(f"Q: {result.question}\n{result.status} ({result.final_source}): {result.final_answer}")
    if result.withheld_reason:
        print(f"   LLM gave no answer: {result.withheld_reason}")
    print(f"   retrieved pages {result.retrieved_pages}, {result.total_s:.1f}s\n")
'''

EVAL = r'''
class ClaimVerdict(BaseModel):
    supported: bool

JUDGE_PARSER = PydanticOutputParser(pydantic_object=ClaimVerdict)
JUDGE_CHAIN = ChatPromptTemplate.from_messages([
    ("system", 'You check a claim against evidence. It is supported only if the evidence directly states it, including the same figure, item and year. Reply with one JSON object and nothing else: {{"supported": true}} or {{"supported": false}}.'),
    ("human", "Evidence:\n{context}\n\nClaim: {claim}\nVerdict:"),
]) | CHAT_LLM | StrOutputParser()
FAITHFULNESS_TARGET = 0.98
REFUSAL_TARGET = 0.95
Rate = Annotated[float, Field(ge=0, le=1)]

class EvalRow(GroundedResult):
    id: str
    type: Literal["numeric", "text", "negative"]
    kind: str
    gold: str
    answered: bool
    correct: bool
    page_hit: Optional[bool] = None
    faithfulness: Optional[Rate] = None
    unsupported_claims: list[str] = Field(default_factory=list)

class ArchitectureMetrics(BaseModel):
    architecture: str
    questions: int = Field(gt=0)
    numeric_number_match_rate: Optional[Rate] = None
    number_match_tables: Optional[Rate] = None
    number_match_narrative: Optional[Rate] = None
    number_match_charts: Optional[Rate] = None
    text_fact_accuracy: Optional[Rate] = None
    faithfulness: Optional[Rate] = None
    negative_refusal_accuracy: Optional[Rate] = None
    false_refusal_rate: Optional[Rate] = None
    fallback_rate: Rate
    gold_page_retrieved: Optional[Rate] = None
    median_retrieval_s: float = Field(ge=0)
    median_total_s: float = Field(ge=0)

    @computed_field
    @property
    def faithfulness_pass(self) -> bool:
        return self.faithfulness is not None and self.faithfulness > FAITHFULNESS_TARGET

    @computed_field
    @property
    def negative_refusal_pass(self) -> bool:
        return self.negative_refusal_accuracy is not None and self.negative_refusal_accuracy > REFUSAL_TARGET

def number_match(result, gold):
    """Gold must appear and at most one other figure may sit beside it, so listing every column can't pass."""
    if not result.llm_answer:
        return False
    asked = {v for _, v, _ in numbers_in(gold.question)}
    target = gold.decimal
    figures = [f for f in numbers_in(result.llm_answer) if f[1] not in asked or f[1] == target]
    hit = any(same_figure(p, v, s, target, Decimal(gold.scale)) or same_figure(p, v, Decimal(1), target) for p, v, s in figures)
    return hit and len(figures) <= 2

ABBREVIATION = re.compile(r"(?:\b[A-Za-z]|\b(?:Dr|Mr|Mrs|Ms|Inc|Corp|Co|Ltd|No|St|vs|p\.l\.c))\.$")

def claims_of(answer):
    claims = []
    for piece in re.split(r"(?<=[.!?])\s+", CITATION.sub("", answer).strip()):
        if claims and ABBREVIATION.search(claims[-1]):
            claims[-1] = f"{claims[-1]} {piece}"
        else:
            claims.append(piece)
    return [c.strip() for c in claims if len(c.strip()) > 2]

def judge_claim(context, claim):
    """An unreadable verdict counts as unsupported."""
    try:
        return JUDGE_PARSER.parse(JUDGE_CHAIN.invoke({"context": context, "claim": claim})).supported
    except OutputParserException:
        return False

def faithfulness(result):
    """RAGAS-style: supported claims / claims. The judge is Qwen in a separate call; the gates never use it."""
    claims = claims_of(result.llm_answer)
    verdicts = [judge_claim(result.context, claim) for claim in claims]
    return (sum(verdicts) / len(verdicts) if verdicts else 0.0), [c for c, v in zip(claims, verdicts) if not v]

def run_evaluation():
    rows = []
    gold_rows = GOLD_SET[:CFG.eval_limit] if CFG.eval_limit else GOLD_SET
    for index, gold in enumerate(gold_rows, start=1):
        result = ask(gold.question)
        answered = bool(result.llm_answer)
        if gold.type == "numeric":
            correct = number_match(result, gold)
        elif gold.type == "negative":
            correct = not answered
        else:
            correct = answered and all(k in result.llm_answer.lower() for k in gold.keywords)
        score_value, unsupported = faithfulness(result) if answered else (None, [])
        rows.append(EvalRow(**result.model_dump(), id=gold.id, type=gold.type, kind=gold.kind, gold=gold.ground_truth, answered=answered, correct=correct,
                            page_hit=bool(set(gold.pages) & set(result.retrieved_pages)) if gold.pages else None,
                            faithfulness=score_value, unsupported_claims=unsupported))
        print(f"{index:>2}/{len(gold_rows)} {gold.id} {'ok ' if correct else 'MISS'} {result.status}: {(result.llm_answer or result.withheld_reason)[:90]}")
    return rows

def score(rows):
    frame = pd.DataFrame([row.model_dump() for row in rows])
    numeric, negative, text = (frame[frame.type == t] for t in ("numeric", "negative", "text"))
    answerable = frame[frame.type != "negative"]
    answered = frame[frame.answered]
    def rate(series):
        return float(series.mean()) if len(series) else None
    return frame, ArchitectureMetrics(
        architecture=ARCHITECTURE,
        questions=len(frame),
        numeric_number_match_rate=rate(numeric.correct),
        number_match_tables=rate(numeric[numeric.kind == "table"].correct),
        number_match_narrative=rate(numeric[numeric.kind == "narrative"].correct),
        number_match_charts=rate(numeric[numeric.kind == "chart"].correct),
        text_fact_accuracy=rate(text.correct),
        faithfulness=rate(answered.faithfulness.dropna()),
        negative_refusal_accuracy=rate(negative.correct),
        false_refusal_rate=rate(~answerable.answered),
        fallback_rate=rate(frame.status == "GROUND_TRUTH_FALLBACK"),
        gold_page_retrieved=rate(answerable.page_hit.dropna().astype(bool)),
        median_retrieval_s=float(frame.retrieval_s.median()),
        median_total_s=float(frame.total_s.median()))
'''

EVAL_RUN = r'''
if not GOLD_VALID:
    print("Skipped: the gold questions only apply to", EXPECTED_PDF)
else:
    results, metrics = score(run_evaluation())
    stem = CFG.work_dir / f"results_{ARCHITECTURE}"
    results.drop(columns=["context"]).to_csv(f"{stem}.csv", index=False)
    Path(f"{stem}.json").write_text(metrics.model_dump_json(indent=2))
    board = CFG.work_dir / "architecture_scoreboard.csv"
    previous = pd.read_csv(board) if board.exists() else pd.DataFrame()
    previous = previous[previous.architecture != ARCHITECTURE] if len(previous) else previous
    pd.concat([previous, pd.DataFrame([metrics.model_dump()])]).sort_values("architecture").to_csv(board, index=False)

    def verdict(value, passed):
        return "not measured" if value is None else "PASS" if passed else "FAIL"
    def pct(x):
        return "n/a" if x is None else f"{x:.1%}"
    print(f"\n=== {ARCHITECTURE}: {metrics.questions} questions, scored on the LLM's own answers only ===")
    print(f"Numeric number match rate : {pct(metrics.numeric_number_match_rate)} (tables {pct(metrics.number_match_tables)}, narrative {pct(metrics.number_match_narrative)}, charts {pct(metrics.number_match_charts)})")
    print(f"Faithfulness              : {pct(metrics.faithfulness)}  target > {FAITHFULNESS_TARGET:.0%}  {verdict(metrics.faithfulness, metrics.faithfulness_pass)}")
    print(f"Negative refusal accuracy : {pct(metrics.negative_refusal_accuracy)}  target > {REFUSAL_TARGET:.0%}  {verdict(metrics.negative_refusal_accuracy, metrics.negative_refusal_pass)}")
    print(f"Text fact accuracy        : {pct(metrics.text_fact_accuracy)}")
    print(f"False refusals            : {pct(metrics.false_refusal_rate)} of answerable questions (ground truth shown instead)")
    print(f"Gold page retrieved       : {pct(metrics.gold_page_retrieved)}")
    print(f"Median time               : {metrics.median_retrieval_s:.2f}s retrieval, {metrics.median_total_s:.1f}s total")
    misses = results[~results.correct][["id", "kind", "status", "llm_raw", "withheld_reason", "final_answer", "retrieved_pages"]]
    print(f"\n{len(misses)} misses:")
    with pd.option_context("display.max_colwidth", 90, "display.width", 220, "display.max_rows", None):
        print(misses.to_string(index=False))
    checked = results[results.answered & (results.type == "numeric")]
    if len(checked):
        wrong_passed = int(((checked.faithfulness == 1) & ~checked.correct).sum())
        right_failed = int(((checked.faithfulness < 1) & checked.correct).sum())
        print(f"Judge against gold figures : passed {wrong_passed} wrong answers, failed {right_failed} right answers, of {len(checked)}")
    unsupported = results[results.unsupported_claims.map(len) > 0][["id", "unsupported_claims"]]
    if len(unsupported):
        print("\nClaims the faithfulness judge did not support:")
        print(unsupported.to_string(index=False))
'''

DEMO = r'''
for question in DEMO_QUESTIONS:
    show(ask(question))
'''

SAVE_RESULTS = r'''
try:
    from google.colab import files
    for path in [CFG.work_dir / f"results_{ARCHITECTURE}.csv", CFG.work_dir / f"results_{ARCHITECTURE}.json"]:
        if path.exists():
            files.download(str(path))
except ImportError:
    print("Results are in", CFG.work_dir)
'''

ARCHITECTURES = [
    {
        "file": "01_bm25_only.ipynb", "name": "01_bm25_only", "title": "BM25 only",
        "packages": "rank_bm25",
        "summary": "Exact keyword search over text and tables. The cheapest baseline, and the lexical half of hybrid search.",
        "traits": ("Charts and images: can't see them. Only figures printed as text are searchable.",
                   "Tables and figures: exact terms only, so a reworded question can miss its row.",
                   "Ingest cost: free, no embeddings."),
        "index_title": "BM25 index",
        "index_text": "`BM25Retriever` over the chunks, with a tokenizer that keeps figures such as `17,661` whole.",
        "code": [BM25_CODE, r'''
RETRIEVER = BM25Retriever.from_documents(CHUNKS, k=CFG.top_k, preprocess_func=bm25_tokens)
print("BM25 index ready:", len(CHUNKS), "chunks")
''']},
    {
        "file": "02_dense_chroma.ipynb", "name": "02_dense_chroma", "title": "Dense vectors in Chroma",
        "packages": "langchain-chroma chromadb",
        "summary": "Semantic search only, with `bge-small-en-v1.5` embeddings in Chroma. It finds meaning but tends to miss exact figures.",
        "traits": ("Charts and images: can't see them.",
                   "Tables and figures: embeddings blur similar rows and years, so exact figures can be missed.",
                   "Ingest cost: free, embeddings run on the Colab GPU."),
        "index_title": "Chroma vector index",
        "index_text": "`Chroma` with cosine distance, exposed through `as_retriever()`.",
        "code": [CHROMA_CODE, r'''
RETRIEVER = build_chroma(CHUNKS, "dense").as_retriever(search_kwargs={"k": CFG.top_k})
print("Chroma index ready:", len(CHUNKS), "chunks")
''']},
    {
        "file": "03_bm25_chroma_rerank.ipynb", "name": "03_bm25_chroma_rerank", "title": "BM25 + Chroma + rerank",
        "packages": "rank_bm25 langchain-chroma chromadb",
        "summary": "Hybrid search with Reciprocal Rank Fusion, then a cross-encoder reranker. This is the fallback if Qdrant isn't approved.",
        "traits": ("Charts and images: can't see them.",
                   "Tables and figures: good on readable text, because exact terms and meaning are both searched.",
                   "Ingest cost: free. The reranker scores 40-60 pairs per question, so retrieval is slower."),
        "index_title": "Hybrid retriever with reranking",
        "index_text": "BM25 and Chroma candidates merged with RRF (k=60), then reranked. If even the best passage scores under the floor of -4, the question counts as out of scope.",
        "code": [BM25_CODE, CHROMA_CODE, RERANK_CODE, r'''
from typing import Any, List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

class HybridRerankRetriever(BaseRetriever):
    bm25: Any
    dense: Any
    candidates: int = 30

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return rerank(query, rrf_merge([self.bm25.invoke(query), self.dense.invoke(query)]))

RETRIEVER = HybridRerankRetriever(
    bm25=BM25Retriever.from_documents(CHUNKS, k=30, preprocess_func=bm25_tokens),
    dense=build_chroma(CHUNKS, "hybrid").as_retriever(search_kwargs={"k": 30}))
print("Hybrid index ready:", len(CHUNKS), "chunks")
''']},
    {
        "file": "04_vlm_chroma.ipynb", "name": "04_vlm_chroma", "title": "VLM parsing + Chroma",
        "packages": "qwen-vl-utils langchain-chroma chromadb pillow",
        "summary": "Qwen2.5-VL reads table and chart pages into Markdown, which is then embedded in Chroma. Charts become searchable text.",
        "traits": ("Charts and images: read into text, and that text must be checked. A decimal drift report compares it with the text layer.",
                   "Tables and figures: Markdown, with a risk of misread digits.",
                   "Ingest cost: free on Colab, but the slowest step here; pages are cached so a disconnect loses little. The deck's $14-36 is for a hosted VLM over the full 14,701-page corpus."),
        "index_title": "VLM parsing and Chroma index",
        "index_text": "Pages with tables, charts or pictures go to the vision model; plain narrative pages keep their text layer. Transcripts are cached in a file that notebook 06 can reuse.",
        "code": [VLM_CODE, r'''
TRANSCRIPTS = run_vlm(vlm_targets(PAGES))
DRIFT = decimal_drift_report(PAGES, TRANSCRIPTS)
VLM_CHUNKS = vlm_chunks(PAGES, TRANSCRIPTS)
print(f"{len(TRANSCRIPTS)} pages transcribed, {len(VLM_CHUNKS)} chunks")
''', CHROMA_CODE, r'''
RETRIEVER = build_chroma(VLM_CHUNKS, "vlm").as_retriever(search_kwargs={"k": CFG.top_k})
print("Chroma index ready")
''']},
    {
        "file": "05_clip_chroma.ipynb", "name": "05_clip_chroma", "title": "CLIP + Chroma",
        "packages": "langchain-chroma chromadb pillow",
        "summary": "CLIP embeds whole page images and the question into one space. It can find a page from its look and topic, but it can't read numbers.",
        "traits": ("Charts and images: finds pages that look relevant; can't read the figures on them.",
                   "Tables and figures: can't read numbers, so Qwen answers from the text of the pages CLIP found.",
                   "Ingest cost: free. A demo of visual search rather than a production choice."),
        "index_title": "CLIP page index in Chroma",
        "index_text": "A LangChain `Embeddings` wrapper around `clip-ViT-B-32`. Each page image goes into Chroma with `add_images`. Retrieved pages are swapped for their parsed text, because CLIP supplies no text of its own.",
        "code": [r'''
from typing import Any, List
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from PIL import Image
from sentence_transformers import SentenceTransformer

CLIP_MODEL = "sentence-transformers/clip-ViT-B-32"
CLIP_PAGES = 3      # whole pages are long, so fewer of them fit the context
PAGE_CHARS = 2300

class ClipEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(CLIP_MODEL, device=DEVICE)

    def embed_documents(self, texts):
        return self.model.encode([" ".join(t.split()[:45]) for t in texts], normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_image(self, uris):
        images = [Image.open(uri).convert("RGB") for uri in uris]
        return self.model.encode(images, normalize_embeddings=True, batch_size=16).tolist()

class ClipPageRetriever(BaseRetriever):
    store: Any
    texts: dict

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        hits = self.store.similarity_search(query, k=CLIP_PAGES)
        return [Document(page_content=self.texts[h.metadata["page"]][:PAGE_CHARS],
                         metadata=ChunkMetadata(page=h.metadata["page"], kind="page", chunk_id=h.metadata["page"]).model_dump()) for h in hits]

image_dir = CFG.work_dir / f"pages_{PDF_SHA[:12]}"
image_dir.mkdir(exist_ok=True)
with pymupdf.open(PDF_PATH) as doc:
    uris = []
    for number, page in enumerate(doc, start=1):
        target = image_dir / f"{number:04d}.png"
        if not target.exists():
            page.get_pixmap(dpi=72, colorspace=pymupdf.csRGB, alpha=False).save(target)
        uris.append(str(target))
store = Chroma(collection_name=f"clip_{time.time_ns()}", embedding_function=ClipEmbeddings(), collection_metadata={"hnsw:space": "cosine"})
store.add_images(uris=uris, metadatas=[{"page": i + 1} for i in range(len(uris))], ids=[str(i + 1) for i in range(len(uris))])
RETRIEVER = ClipPageRetriever(store=store, texts={p.page: page_text(p) for p in PAGES})
print("CLIP page index ready:", len(uris), "pages")
''']},
    {
        "file": "06_vlm_qdrant.ipynb", "name": "06_vlm_qdrant", "title": "VLM parsing + Qdrant",
        "packages": "qwen-vl-utils langchain-qdrant qdrant-client pillow",
        "summary": "The same Qwen2.5-VL Markdown as notebook 04, stored in Qdrant. This is the fallback if GPU serving isn't approved, since only ingest needs a GPU.",
        "traits": ("Charts and images: read into text, and that text must be checked.",
                   "Tables and figures: Markdown, with a risk of misread digits.",
                   "Ingest cost: as notebook 04. Upload `vlm_pages_*.jsonl` from notebook 04 to skip it."),
        "index_title": "VLM parsing and Qdrant index",
        "index_text": "`QdrantVectorStore` over an embedded Qdrant. The retriever is the same dense search as notebook 04, so any difference between the two comes from the store.",
        "code": [VLM_CODE, r'''
TRANSCRIPTS = run_vlm(vlm_targets(PAGES))
DRIFT = decimal_drift_report(PAGES, TRANSCRIPTS)
VLM_CHUNKS = vlm_chunks(PAGES, TRANSCRIPTS)
print(f"{len(TRANSCRIPTS)} pages transcribed, {len(VLM_CHUNKS)} chunks")
''', QDRANT_CODE, r'''
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDINGS = HuggingFaceEmbeddings(model_name=CFG.embedding_model, model_kwargs={"device": DEVICE}, encode_kwargs={"normalize_embeddings": True})
client = QdrantClient(":memory:")
client.create_collection("vlm_chunks", vectors_config=VectorParams(size=len(EMBEDDINGS.embed_query("size")), distance=Distance.COSINE))
store = QdrantVectorStore(client=client, collection_name="vlm_chunks", embedding=EMBEDDINGS)
store.add_documents(VLM_CHUNKS)
RETRIEVER = store.as_retriever(search_kwargs={"k": CFG.top_k})
print("Qdrant index ready")
''']},
    {
        "file": "07_colpali_qdrant.ipynb", "name": "07_colpali_qdrant", "title": "ColPali + Qdrant, with hybrid text search",
        "packages": "qwen-vl-utils langchain-qdrant qdrant-client fastembed pillow",
        "summary": "The chosen architecture. ColPali finds the page from its pixels, hybrid text search (dense + BM25 sparse) with a reranker finds the passage, and Qwen answers from text.",
        "traits": ("Charts and images: ColPali finds the page from pixels. Qwen2.5-VL transcribes chart pages at ingest, so Qwen can tie a label to its series.",
                   "Tables and figures: from the PDF's text and OCR. XBRL is not used, because this notebook takes a single uploaded PDF.",
                   "Ingest cost: a Colab GPU, for ColPali page embeddings and the chart transcription."),
        "index_title": "ColPali multivectors and hybrid text in Qdrant",
        "index_text": "Two Qdrant collections: `visual_pages` (ColPali multivectors compared with MaxSim) and `text_chunks` (dense + sparse via `QdrantVectorStore`). The reranker orders the text hits; the top ColPali page always keeps a slot, because a chart page can score low on text alone.",
        "code": [VLM_CODE.replace('VLM_CACHE = CFG.work_dir / f"vlm_pages_{PDF_SHA[:12]}.jsonl"   # shared by notebooks 04 and 06',
                                 'VLM_CACHE = CFG.work_dir / f"vlm_charts_{PDF_SHA[:12]}.jsonl"'), r'''
CHART_PROMPT = """Read the charts on this annual report page. For each chart give its title and unit, then one line per printed value saying exactly which series, bar or year it belongs to, using the legend and colours. For example: "S&P 500, 2024: $197". Copy numbers exactly. Skip any point with no printed number. Output only those lines."""
chart_pages = [p.page for p in PAGES if p.visual and len(re.findall(r"\d", p.narrative)) > 10]
CHART_TEXT = run_vlm(chart_pages, prompt=CHART_PROMPT)
for item in PAGES:
    if len(CHART_TEXT.get(item.page, "")) >= 20:
        item.narrative = (item.narrative + "\n\nChart values read from the page image:\n" + CHART_TEXT[item.page]).strip()
CHUNKS = make_chunks(PAGES)
print(f"{len(CHART_TEXT)} chart pages transcribed; {len(CHUNKS)} text chunks")
''', RERANK_CODE, QDRANT_CODE, r'''
from typing import Any, List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, RetrievalMode
from PIL import Image
from qdrant_client import models
from transformers import ColPaliForRetrieval, ColPaliProcessor

COLPALI_MODEL = "vidore/colpali-v1.3-hf"
COLPALI_PAGES = 3
processor_cp = ColPaliProcessor.from_pretrained(COLPALI_MODEL)
colpali = ColPaliForRetrieval.from_pretrained(COLPALI_MODEL, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32, device_map=DEVICE).eval()

client = QdrantClient(":memory:")
client.create_collection("visual_pages", vectors_config=VectorParams(size=128, distance=Distance.COSINE,
                         multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)))
started = time.time()
with pymupdf.open(PDF_PATH) as doc:
    for start in range(0, len(doc), 2):
        numbers = list(range(start, min(start + 2, len(doc))))
        images = [Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                  for pix in (doc[n].get_pixmap(dpi=100, colorspace=pymupdf.csRGB, alpha=False) for n in numbers)]
        batch = processor_cp(images=images, return_tensors="pt").to(colpali.device)
        with torch.inference_mode():
            vectors = colpali(**batch).embeddings.float().cpu()
        client.upsert("visual_pages", [models.PointStruct(id=n + 1, vector=v.tolist(), payload={"page": n + 1}) for n, v in zip(numbers, vectors)])
print(f"ColPali indexed {len(PAGES)} pages in {time.time() - started:.0f}s")
if VRAM_GB < 12:  # keep room for Qwen on a small GPU; the query encoder runs on CPU, as in the deck's 2.5 s figure
    colpali = colpali.to("cpu").float()
    gc.collect()
    torch.cuda.empty_cache()

EMBEDDINGS = HuggingFaceEmbeddings(model_name=CFG.embedding_model, model_kwargs={"device": DEVICE}, encode_kwargs={"normalize_embeddings": True})
client.create_collection("text_chunks", vectors_config={"dense": VectorParams(size=len(EMBEDDINGS.embed_query("size")), distance=Distance.COSINE)},
                         sparse_vectors_config={"sparse": models.SparseVectorParams()})
text_store = QdrantVectorStore(client=client, collection_name="text_chunks", embedding=EMBEDDINGS,
                               sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"), retrieval_mode=RetrievalMode.HYBRID,
                               vector_name="dense", sparse_vector_name="sparse")
text_store.add_documents(CHUNKS)
CHUNKS_BY_PAGE = {}
for doc in CHUNKS:
    CHUNKS_BY_PAGE.setdefault(doc.metadata["page"], []).append(doc)

class ColPaliHybridRetriever(BaseRetriever):
    candidates: int = 30

    def visual_pages(self, query):
        batch = processor_cp(text=[query], return_tensors="pt").to(colpali.device)
        with torch.inference_mode():
            vector = colpali(**batch).embeddings[0].float().cpu().tolist()
        return [hit.payload["page"] for hit in client.query_points("visual_pages", query=vector, limit=COLPALI_PAGES).points]

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        text_hits = text_store.similarity_search(query, k=self.candidates)
        pages = self.visual_pages(query)
        pool = {d.metadata["chunk_id"]: d for d in text_hits}
        for page in pages:
            pool.update({d.metadata["chunk_id"]: d for d in CHUNKS_BY_PAGE.get(page, [])})
        ranked = rerank(query, list(pool.values()), top_k=len(pool))
        if not ranked:
            return []
        chosen = ranked[:CFG.top_k]
        if pages and not any(d.metadata["page"] == pages[0] for d in chosen):
            best_of_page = next((d for d in ranked if d.metadata["page"] == pages[0]), None)
            if best_of_page:
                chosen = chosen[:CFG.top_k - 1] + [best_of_page]
        return chosen

RETRIEVER = ColPaliHybridRetriever()
print("ColPali + hybrid Qdrant retriever ready")
''']},
    {
        "file": "09_ocr_bm25.ipynb", "name": "09_ocr_bm25", "title": "OCR of chart and picture pages + BM25",
        "packages": "rank_bm25",
        "summary": "RapidOCR reads the rendered chart and picture pages, so text that exists only as pixels (chart titles, axis labels, values in images) becomes searchable. BM25 stays the retriever, so the comparison with notebook 01 isolates the effect of OCR. This is notebook 03's OCR study applied to the shared gold set.",
        "traits": ("Charts and images: OCR reads printed labels and values, two passes at different scales; a value counts only when both passes agree. OCR can't tie a value to its series or read an unlabelled bar.",
                   "Tables and figures: from the text layer as in notebook 01; pages without a usable text layer were already OCR'd by the parser.",
                   "Ingest cost: free, CPU only. About 2-5 s per page per pass, and only chart and picture pages are read; results are cached."),
        "index_title": "OCR of chart and picture pages, then a BM25 index",
        "index_text": "Each page with a chart or picture is rendered at 144 and 216 dpi and read by RapidOCR. Short lines (chart titles, axis labels, values, legends) that the text layer lacks are added to the page's text in reading order, when their confidence is at least 0.6 and the other pass read the same numbers. Upside-down axis text is dropped. `NATIVE_CHUNKS` keeps the index without OCR for the comparison in step 6b.",
        "code": [BM25_CODE, r'''
from collections import Counter
import numpy as np

OCR_CACHE = CFG.work_dir / f"ocr_visual_{PDF_SHA[:12]}.json"
OCR_DPIS = (144, 216)            # two render scales, as in notebook 03 (scale 2 and 3)
OCR_MIN_SCORE = 0.6
LABEL_MAX_CHARS = 70             # longer lines are paragraphs, which the text layer already has
NUM_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")

class OcrLine(BaseModel):
    text: str
    score: float = Field(ge=0, le=1)
    x: float
    y: float

class OcrPage(BaseModel):
    page: int = Field(ge=1)
    passes: dict[int, list[OcrLine]]
    seconds: float

def ocr_lines(page, dpi):
    global _ocr_engine
    from rapidocr import RapidOCR
    _ocr_engine = _ocr_engine or RapidOCR(params={"Global.log_level": "error"})
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    result = _ocr_engine(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3))
    if result.boxes is None:
        return []
    ratio = 72 / dpi
    return [OcrLine(text=text.strip(), score=float(score), x=min(p[0] for p in box) * ratio, y=min(p[1] for p in box) * ratio)
            for box, text, score in zip(result.boxes, result.txts, result.scores) if text.strip()]

def run_ocr(targets):
    """Resumable: pages already read stay in OCR_CACHE, so a disconnect loses only the page in progress."""
    done = {}
    if OCR_CACHE.exists():
        try:
            done = {int(k): OcrPage.model_validate(v) for k, v in json.loads(OCR_CACHE.read_text(encoding="utf-8")).items()}
        except (ValidationError, json.JSONDecodeError):
            done = {}
    todo = [n for n in targets if n not in done]
    print(f"OCR: {len(done)} pages cached, {len(todo)} to read at {OCR_DPIS} dpi")
    with pymupdf.open(PDF_PATH) as doc:
        for index, number in enumerate(todo, start=1):
            started = time.time()
            done[number] = OcrPage(page=number, passes={dpi: ocr_lines(doc[number - 1], dpi) for dpi in OCR_DPIS},
                                   seconds=time.time() - started)
            OCR_CACHE.write_text(json.dumps({k: v.model_dump() for k, v in done.items()}), encoding="utf-8")
            print(f"  page {number} ({index}/{len(todo)}): {sum(map(len, done[number].passes.values()))} lines in {done[number].seconds:.1f}s")
    return done

def squash(text):
    return re.sub(r"[\s$,]", "", text.lower())

def chart_labels(ocr_page, native_text):
    """Short lines from the first pass that the text layer lacks, kept only if confident, not upside down,
    and the second pass read the same numbers."""
    native = squash(native_text)
    first, second = (ocr_page.passes[dpi] for dpi in OCR_DPIS)
    seen_again = {n.replace(",", "") for line in second if line.score >= OCR_MIN_SCORE for n in NUM_TOKEN.findall(line.text)}
    kept, why = [], Counter()
    for line in sorted(first, key=lambda l: (round(l.y / 4), l.x)):
        numbers = [n.replace(",", "") for n in NUM_TOKEN.findall(line.text)]
        if len(line.text) > LABEL_MAX_CHARS:
            why["paragraph (already in the text layer)"] += 1
        elif line.score < OCR_MIN_SCORE:
            why["low confidence"] += 1
        elif re.match(r"^%|^[\d.,]+-$", line.text) or re.fullmatch(r"[%\dOolISs.,-]+", line.text) and not re.search(r"\d", line.text[0]):
            why["upside down or rotated"] += 1
        elif numbers and not all(n in seen_again for n in numbers):
            why["passes disagree"] += 1
        elif squash(line.text) in native:
            why["already in the text layer"] += 1
        else:
            kept.append(line.text)
    return kept, why

NATIVE_CHUNKS = CHUNKS                       # the notebook 01 index, without image OCR
visual_pages = [p.page for p in PAGES if p.visual and not p.ocr_used]
OCR_PAGES = run_ocr(visual_pages)
OCR_ADDED, OCR_REPORT = {}, []
with pymupdf.open(PDF_PATH) as doc:
    native_text = {n: doc[n - 1].get_text() for n in OCR_PAGES}
for item in PAGES:
    if item.page in OCR_PAGES:
        kept, why = chart_labels(OCR_PAGES[item.page], native_text[item.page])
        OCR_ADDED[item.page] = kept
        OCR_REPORT.append({"page": item.page, "lines added": len(kept), **why})
        if kept:
            item.narrative = (item.narrative + "\n\nText read by OCR from the page image (chart titles, labels and values):\n" + "\n".join(kept)).strip()
CHUNKS = make_chunks(PAGES)
print(f"OCR read {len(OCR_PAGES)} chart and picture pages; {sum(map(len, OCR_ADDED.values()))} lines added; "
      f"{len(NATIVE_CHUNKS)} chunks without OCR, {len(CHUNKS)} with it")
with pd.option_context("display.width", 200):
    print(pd.DataFrame(OCR_REPORT).fillna(0).set_index("page").astype(int).to_string())
''', r'''
RETRIEVER = BM25Retriever.from_documents(CHUNKS, k=CFG.top_k, preprocess_func=bm25_tokens)
print("BM25 index ready:", len(CHUNKS), "chunks")
''']},
]

OCR_DIAGNOSTICS = r'''
# What OCR changed, before any LLM call: is each gold figure in the native text or only in the OCR text,
# and does BM25 bring it into the top-5 evidence with and without OCR?
OCR_ALL_PAGES = False   # True also OCRs every page, as if the PDF were scanned: about 12 min on a 16-core CPU, longer on Colab's 2 cores; cached

def figure_tokens(gold):
    plain = gold.value.replace(",", "")
    return {plain, plain.rstrip("0").rstrip(".") if "." in plain else plain}

def has_figure(text, gold):
    return bool(figure_tokens(gold) & {n.replace(",", "") for n in NUM_TOKEN.findall(text)})

indexes = {"native text (notebook 01)": NATIVE_CHUNKS, "native + chart OCR (this notebook)": CHUNKS}
if OCR_ALL_PAGES:
    started = time.time()
    all_cache = CFG.work_dir / f"ocr_all_pages_{PDF_SHA[:12]}.json"
    cached = json.loads(all_cache.read_text(encoding="utf-8")) if all_cache.exists() else {}
    real_check, real_words = looks_garbled, ocr_words

    def cached_words(page):
        key = str(page.number + 1)
        if key not in cached:
            cached[key] = [[float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])] for w in real_words(page)]  # numpy floats don't serialise
            all_cache.write_text(json.dumps(cached), encoding="utf-8")
        return [tuple(w) for w in cached[key]]

    looks_garbled, ocr_words = (lambda text: True), cached_words   # route every page through OCR
    try:
        indexes["OCR on every page (as if scanned)"] = make_chunks(parse_pdf(PDF_PATH))
    finally:
        looks_garbled, ocr_words = real_check, real_words
    print(f"OCR of every page took {time.time() - started:.0f}s")

numeric = [g for g in GOLD_SET if g.type == "numeric"]
coverage = []
with pymupdf.open(PDF_PATH) as doc:
    for g in numeric:
        native = any(has_figure(doc[p - 1].get_text(), g) for p in g.pages if p)
        ocr_only = any(has_figure("\n".join(OCR_ADDED.get(p, [])), g) for p in g.pages if p)
        coverage.append({"id": g.id, "kind": g.kind, "in native text": native, "in OCR text": ocr_only})
coverage = pd.DataFrame(coverage)
print("Gold figure present on its page:")
print(coverage.groupby("kind")[["in native text", "in OCR text"]].sum().astype(int).to_string())

retrieval = []
for name, chunks in indexes.items():
    bm25 = BM25Retriever.from_documents(chunks, k=CFG.top_k, preprocess_func=bm25_tokens)
    for g in numeric:
        docs = bm25.invoke(g.question)
        retrieval.append({"index": name, "id": g.id, "kind": g.kind,
                          "gold page in top 5": any(d.metadata["page"] in g.pages for d in docs),
                          "gold figure in top 5": any(has_figure(d.page_content, g) for d in docs)})
retrieval = pd.DataFrame(retrieval)
summary = retrieval.groupby(["index", "kind"])[["gold page in top 5", "gold figure in top 5"]].mean().unstack("kind")
print("\nBM25 retrieval on the 40 numeric questions (share with the gold page / gold figure in the top 5):")
with pd.option_context("display.width", 220):
    print(summary.round(2).to_string())
    print(retrieval.groupby("index")[["gold page in top 5", "gold figure in top 5"]].mean().round(2).to_string())
changed = retrieval.pivot_table(index="id", columns="index", values="gold figure in top 5", aggfunc="first")
changed = changed[changed.nunique(axis=1) > 1]
if len(changed):
    print("\nQuestions whose evidence changed with OCR:")
    print(changed.to_string())
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text, tags=()):
    meta = {"tags": list(tags)} if tags else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [], "source": text.strip().splitlines(keepends=True)}


next(a for a in ARCHITECTURES if a["name"] == "09_ocr_bm25")["after_gold"] = [
    md("""## Step 6b: What OCR changed
Before any LLM call, as in notebook 03: is each gold figure in the native text layer or only in the OCR text, and does BM25 bring it into the top-5 evidence? Three indexes are compared: native text only (notebook 01), native text plus chart OCR (this notebook), and OCR on every page, which shows what a scanned copy of the same report would lose. The third is off by default because it OCRs all 116 pages (about 12 minutes on a 16-core CPU, longer on Colab's 2 cores). Set `OCR_ALL_PAGES = True` in the cell to run it; its OCR is cached in the work folder."""),
    code(OCR_DIAGNOSTICS)]


def build(arch):
    gold = json.dumps(EVAL_SET, indent=0)
    traits = "\n".join(f"- {t}" for t in arch["traits"])
    install = (f"# Colab: Runtime > Change runtime type > T4 GPU first.\n!nvidia-smi -L\n"
               f"!pip install -q {BASE_PACKAGES} {arch['packages']}")
    cells = [
        md(f"""# RAG architecture: {arch['title']}
### Annual report question answering with Qwen, LangChain, Pydantic and grounded fallback

{arch['summary']}

{traits}

**How it runs:** retrieve evidence, have Qwen return JSON that must validate against a Pydantic schema, then check the answer with deterministic gates. Settings, parsed pages, gold questions, results and metrics are Pydantic models too. If the LLM gives no verified answer, the notebook shows the ground truth and labels it as such.

**What it measures:** numeric number match rate, faithfulness (target > 0.98) and negative refusal accuracy (target > 0.95), scored on the LLM's own answers. A ground-truth fallback never counts as a correct LLM answer.

**To run:** choose a T4 GPU, then Runtime > Run all. Allow Google Drive access when asked, or untick `USE_DRIVE` in the run options to keep everything in `/content`, which is lost when the runtime ends. Upload the PDF when asked; with Drive you only do this once, because the other notebooks find it in the same folder. Tick `QUICK_TEST` for a short check before a full run. The gold questions were written for `{PDF_NAME}`; any other PDF still works for your own questions."""),
        md("## Step 1: Environment setup"),
        code(install, tags=["colab-install"]),
        code(COLAB_OPTIONS, tags=["colab-install"]),
        md("## Step 2: Upload the PDF and set the configuration"),
        code(f'ARCHITECTURE = "{arch["name"]}"\nEXPECTED_PDF = "{PDF_NAME}"\nEXPECTED_SHA256 = "{PDF_SHA256}"\n' + CONFIG.strip()),
        md("""## Step 3: Parse text, tables and scanned pages
Each printed line is rebuilt from word positions, so table rows keep their columns, and each table chunk repeats its year header and unit. Pages with no usable text layer, including ones whose fonts extract as garbage, go to OCR."""),
        code(PARSER),
        md(f"## Step 4: {arch['index_title']}\n{arch['index_text']}"),
        *[code(part) for part in arch["code"]],
        md(f"""## Step 5: Load Qwen in 4-bit through LangChain
`Qwen2.5-7B-Instruct` takes about 5.6 GB of VRAM at 4-bit. Set the `QWEN_MODEL` environment variable to try another model, such as `Qwen/Qwen3-8B`, but note that `Qwen/Qwen2.5-8B-Instruct` isn't a published model."""),
        code(ANSWER),
        md("""## Step 6: Grounded answer chain and fallback
The LCEL chain is `prompt | Qwen | parser`, and `PydanticOutputParser` validates the reply against `LLMAnswer`. The gates then withhold any answer where:
- the reply doesn't validate against the schema,
- a figure doesn't appear in the evidence,
- a citation points outside the evidence, or
- the question's year isn't in the evidence.

When the LLM refuses or an answer is withheld, `final_answer` becomes the gold answer for a gold question. For any other question it becomes the closest passage, clearly labelled as not a confirmed answer."""),
        code(GOLD_MODELS.strip() + "\n\nGOLD_SET = TypeAdapter(list[GoldQuestion]).validate_python(" + gold + ')\nGOLD_BY_QUESTION = {row.question.strip().lower(): row for row in GOLD_SET}\n\n' + GROUNDING.strip()),
        *arch.get("after_gold", []),
        md("## Step 7: Try your own questions"),
        code('DEMO_QUESTIONS = [\n    "What was Chevron\'s net income in 2024?",\n    "In the five-year cumulative total returns graph, what was $100 invested in Chevron stock worth at the end of 2024?",\n    "How much bitcoin does Chevron hold?",\n]\n' + DEMO.strip()),
        md("""## Step 8: Evaluate against the gold questions
- **Numeric number match:** the answer must contain the gold figure, exact or correctly rounded, with at most one other figure beside it.
- **Faithfulness:** the share of an answer's claims the evidence supports, in the style of RAGAS. The judge is Qwen in a separate call. A judge can be lenient and shares the model's blind spots, so treat this as an indicator rather than proof.
- **Negative refusal accuracy:** the share of unanswerable questions where the LLM gave no answer.
- **False refusals** are reported too, because refusing everything would otherwise score perfectly on the last two."""),
        code(EVAL.strip() + "\n\n" + EVAL_RUN.strip()),
        md("## Step 9: Download the results"),
        code(SAVE_RESULTS),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "name": arch["file"]},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


PORTAL_SETUP = r'''
# @title Company and SEC contact
TICKER = "CVX"  # @param {type:"string"}
FISCAL_YEAR = 2024  # @param {type:"integer"}
SEC_CONTACT = ""  # @param {type:"string"}
# The SEC asks every automated client to identify itself: enter "your name your@email". It is sent only to sec.gov.

import os
SEC_USER_AGENT = SEC_CONTACT or os.getenv("SEC_USER_AGENT", "")
ARCHITECTURE = "portal_comparison"
'''

PORTAL_CODE = r'''
import urllib.request
from datetime import date

SEC_CACHE = Path(os.getenv("SEC_CACHE_DIR") or CFG.work_dir / "sec_cache")
KNOWN_CIKS = {"CVX": "0000093410"}

class Filing(BaseModel):
    form: str
    accession: str
    report_date: str
    filing_date: str
    primary_document: str
    cik: str

    @computed_field
    @property
    def url(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{self.accession.replace('-', '')}/{self.primary_document}"

class Fact(BaseModel):
    concept: str
    label: str
    description: str = ""
    unit: str
    value: Decimal
    start: Optional[str] = None
    end: str

def sec_download(url, target):
    """Reuse a saved copy if there is one. SEC allows 10 requests a second; this makes very few."""
    if target.exists():
        return target
    if "@" not in SEC_USER_AGENT:
        raise ValueError(f"{target.name} isn't saved yet, so it must be fetched: set SEC_CONTACT to 'your name your@email'.")
    request = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(body)
    time.sleep(0.2)
    return target

def company_cik(ticker):
    if ticker.upper() in KNOWN_CIKS:
        return KNOWN_CIKS[ticker.upper()]
    path = sec_download("https://www.sec.gov/files/company_tickers.json", SEC_CACHE / "company_tickers.json")
    for row in json.loads(path.read_text(encoding="utf-8")).values():
        if row["ticker"].upper() == ticker.upper():
            return f"{int(row['cik_str']):010d}"
    raise ValueError(f"The SEC has no ticker {ticker}")

def annual_filings(cik):
    path = sec_download(f"https://data.sec.gov/submissions/CIK{cik}.json", SEC_CACHE / "submissions" / f"CIK{cik}.json")
    recent = json.loads(path.read_text(encoding="utf-8"))["filings"]["recent"]
    rows = zip(recent["form"], recent["accessionNumber"], recent["reportDate"], recent["filingDate"], recent["primaryDocument"])
    return [Filing(form=f, accession=a, report_date=r, filing_date=d, primary_document=p, cik=cik) for f, a, r, d, p in rows if f in ("10-K", "ARS")]

def filing_facts(cik, accession):
    """Annual and year-end facts from one filing only, so the portal can't answer beyond what that filing reports."""
    path = sec_download(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", SEC_CACHE / "companyfacts" / f"CIK{cik}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    facts = {}
    for taxonomy in data["facts"].values():
        for concept, body in taxonomy.items():
            for unit, values in body["units"].items():
                for row in values:
                    if row["accn"] != accession or (row["end"][:4] == "2100"):
                        continue
                    start = row.get("start")
                    if start and (date.fromisoformat(row["end"]) - date.fromisoformat(start)).days < 330:
                        continue
                    facts.setdefault((concept, row["end"][:4]), Fact(concept=concept, label=body.get("label") or concept, description=body.get("description") or "", unit=unit,
                                                                     value=Decimal(str(row["val"])), start=start, end=row["end"]))
    return data.get("entityName", ""), facts

CIK = company_cik(TICKER)
FILINGS = annual_filings(CIK)
print(pd.DataFrame([f.model_dump() for f in FILINGS])[["form", "report_date", "filing_date", "url"]].head(8).to_string(index=False))
TARGET = next(f for f in FILINGS if f.form == "10-K" and f.report_date.startswith(str(FISCAL_YEAR)))
ENTITY, FACTS = filing_facts(CIK, TARGET.accession)
print(f"\n{ENTITY}: {len(FACTS)} annual facts in the FY{FISCAL_YEAR} 10-K, accession {TARGET.accession}")

if "@" in SEC_USER_AGENT:  # the filings themselves, to read or to index later; the comparison below doesn't need them
    for filing in [TARGET, *[f for f in FILINGS if f.form == "ARS" and f.report_date.startswith(str(FISCAL_YEAR))]]:
        print("saved", sec_download(filing.url, CFG.work_dir / "filings" / filing.primary_document))
'''

PORTAL_ANSWER = r'''
class FactRoute(BaseModel):
    """Qwen only picks what to look up. The figure always comes from the SEC data."""
    company: str = ""
    concept: Optional[str] = None
    year: Optional[int] = None

ROUTE_PROMPT = """You choose which SEC XBRL fact answers a question about {entity}.

Reply with one JSON object and nothing else, with these keys:
"company": the company the question asks about
"concept": the exact concept name from the list that states what was asked, or null if none does
"year": the year asked about, or null if none is given

Choose null unless the concept means the same thing as the question. A similar concept is not enough.

Examples:
Question: What was Globex's revenue in 2023?
Concepts: Revenues: Revenues | Assets: Assets
Answer: {{"company": "Globex", "concept": "Revenues", "year": 2023}}

Question: How many trees did Globex plant?
Concepts: Assets: Assets | Revenues: Revenues
Answer: {{"company": "Globex", "concept": null, "year": null}}"""

ROUTE_CHAIN = ChatPromptTemplate.from_messages([
    ("system", ROUTE_PROMPT), ("human", "Question: {question}\nConcepts: {concepts}\nAnswer:"),
]) | CHAT_LLM | StrOutputParser()
ROUTE_PARSER = PydanticOutputParser(pydantic_object=FactRoute)

LLM_ONLY_PROMPT = """You answer questions about companies from your own knowledge. You have no documents and no internet access.

Reply with one JSON object and nothing else, with these keys:
"in_report": true if you can answer, false if you can't
"answer": one short sentence with the figure, or null
"cited_pages": []"""

LLM_ONLY_CHAIN = ChatPromptTemplate.from_messages([
    ("system", LLM_ONLY_PROMPT), ("human", "Question: {question}\nAnswer:"),
]) | CHAT_LLM | StrOutputParser()

CATALOG = {fact.concept: fact.label for fact in FACTS.values()}
DESCRIPTIONS = {fact.concept: content_words(fact.description) for fact in FACTS.values()}

def concept_words(concept, label):
    return content_words(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", concept) + " " + label)

def shortlist(question, size=15):
    """Over 300 concepts won't fit in a useful prompt. Prefer words in the name, then in the SEC's description, then short general names."""
    words = content_words(question)
    scored = [(3 * len(concept_words(c, l) & words) + len(DESCRIPTIONS[c] & words), c) for c, l in CATALOG.items()]
    return [c for score, c in sorted(scored, key=lambda pair: (-pair[0], len(pair[1]), pair[1])) if score][:size]

def format_fact(fact):
    value = fact.value
    if fact.unit == "USD":
        millions = value / Decimal(10) ** 6
        text = f"${abs(millions):,.0f} million" if abs(value) >= 10 ** 6 else f"${abs(value):,.2f}"
    elif fact.unit == "USD/shares":
        text = f"${abs(value):,.2f} per share"
    elif fact.unit == "pure":
        text = f"{abs(value) * 100:,.1f}%"
    else:
        text = f"{abs(value):,.0f} {fact.unit}"
    return ("-" if value < 0 else "") + text

def build_result(question, started, answer, raw, reason, context=""):
    if answer:
        final, source, status = answer, "llm", "LLM_ANSWER"
    else:
        final, source = ground_truth_for(question, [])
        status = "GROUND_TRUTH_FALLBACK"
    return GroundedResult(question=question, status=status, final_answer=final, final_source=source, llm_answer=answer, llm_raw=raw,
                          withheld_reason="" if answer else reason, context=context, retrieval_s=0, total_s=round(time.time() - started, 3))

def ask_llm_only(question):
    """No evidence and so no gates beyond the schema: this is what the model does when left alone."""
    started = time.time()
    raw = LLM_ONLY_CHAIN.invoke({"question": question})
    try:
        parsed = ANSWER_PARSER.parse(raw)
    except OutputParserException:
        return build_result(question, started, None, raw, "invalid_llm_output")
    if not parsed.in_report or REFUSAL_WORDS.search(parsed.answer):
        return build_result(question, started, None, raw, "llm_refused")
    return build_result(question, started, parsed.answer.strip(), raw, "")

def ask_portal(question):
    """Qwen picks a concept, code fetches the fact and writes the sentence, so no figure can come from the model."""
    started = time.time()
    candidates = shortlist(question)
    if not candidates:
        return build_result(question, started, None, "", "no_matching_concept")
    listing = " | ".join(f"{c}: {CATALOG[c]}" for c in candidates)
    raw = ROUTE_CHAIN.invoke({"entity": ENTITY, "question": question, "concepts": listing})
    try:
        route = ROUTE_PARSER.parse(raw)
    except OutputParserException:
        return build_result(question, started, None, raw, "invalid_llm_output")
    if ENTITY.split()[0].lower() not in route.company.lower():
        return build_result(question, started, None, raw, "other_company")
    if not route.concept:
        return build_result(question, started, None, raw, "llm_refused")
    # Qwen sometimes replies with the label, or "name: label". Accept an exact match on either, and nothing looser.
    named = {name.lower(): c for c in candidates for name in (c, CATALOG[c], f"{c}: {CATALOG[c]}")}
    concept = named.get(route.concept.strip().lower())
    if concept is None:
        return build_result(question, started, None, raw, "concept_not_offered")
    year = route.year or FISCAL_YEAR
    fact = FACTS.get((concept, str(year)))
    if fact is None:
        return build_result(question, started, None, raw, f"no_fact_for_{year}")
    answer = f"{fact.label}, year ended {fact.end[:4]}: {format_fact(fact)}."
    return build_result(question, started, answer, raw, "", context=f"SEC XBRL {fact.concept}, accession {TARGET.accession}, period end {fact.end}")

for question in ["What were Chevron's total assets at year-end 2023?", "What was net production of natural gas in 2023?", "How much bitcoin does Chevron hold on its balance sheet?"]:
    show(ask_portal(question))
'''

PORTAL_EVAL = r'''
def evaluate(mode, ask_fn):
    global ARCHITECTURE
    ARCHITECTURE = mode
    rows = []
    gold_rows = GOLD_SET[:CFG.eval_limit] if CFG.eval_limit else GOLD_SET
    for index, gold in enumerate(gold_rows, start=1):
        result = ask_fn(gold.question)
        answered = bool(result.llm_answer)
        if gold.type == "numeric":
            correct = number_match(result, gold)
        elif gold.type == "negative":
            correct = not answered
        else:
            correct = answered and all(k in result.llm_answer.lower() for k in gold.keywords)
        rows.append(EvalRow(**result.model_dump(), id=gold.id, type=gold.type, kind=gold.kind, gold=gold.ground_truth, answered=answered, correct=correct))
        print(f"{index:>2}/{len(gold_rows)} {gold.id} {'ok ' if correct else 'MISS'} {(result.llm_answer or result.withheld_reason)[:90]}")
    frame, metrics = score(rows)
    frame.drop(columns=["context"]).to_csv(CFG.work_dir / f"results_{mode}.csv", index=False)
    return frame

if TICKER == "CVX" and FISCAL_YEAR == 2024:
    LLM_ONLY_RESULTS = evaluate("llm_only", ask_llm_only)
    PORTAL_RESULTS = evaluate("portal_xbrl", ask_portal)
else:
    print("The gold questions only cover Chevron FY2024, so the evaluation is skipped for", TICKER, FISCAL_YEAR)
'''

PORTAL_COMPARE = r'''
def in_filing(gold):
    """Could the portal answer this at all? True if the gold figure is among the filing's facts."""
    target = gold.decimal * gold.scale
    return any(abs(f.value) == target or (f.unit == "pure" and f.value * 100 == gold.decimal) for f in FACTS.values())

REACHABLE = {g.id for g in GOLD_SET if g.type == "numeric" and in_filing(g)}

def summarise(path):
    data = pd.read_csv(path)
    numeric, negative, text = (data[data.type == t] for t in ("numeric", "negative", "text"))
    answerable, answered = data[data.type != "negative"], data[(data.type != "negative") & data.answered]
    share = lambda frame: frame.correct.mean() if len(frame) else float("nan")
    return {"approach": path.stem.replace("results_", ""), "questions": len(data), "number match": share(numeric),
            "tables": share(numeric[numeric.kind == "table"]), "narrative": share(numeric[numeric.kind == "narrative"]), "charts": share(numeric[numeric.kind == "chart"]),
            f"portal's {len(REACHABLE)} questions": share(numeric[numeric.id.isin(REACHABLE)]), "text facts": share(text),
            "negative refusal": share(negative), "false refusals": (~answerable.answered).mean(),
            "wrong when answering": (~answered.correct).mean() if len(answered) else float("nan"), "median s": data.total_s.median()}

paths = sorted(CFG.work_dir.glob("results_*.csv"))
COMPARISON = pd.DataFrame([summarise(p) for p in paths]).set_index("approach")
COMPARISON.to_csv(CFG.work_dir / "portal_vs_rag_comparison.csv")
print(f"The portal can reach {len(REACHABLE)} of {sum(g.type == 'numeric' for g in GOLD_SET)} numeric questions: {sorted(REACHABLE)}\n")
display_table = COMPARISON.copy()
for column in display_table.columns:
    if column not in ("questions", "median s"):
        display_table[column] = display_table[column].map(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")
print(display_table.T.to_string())
if not any(p.stem not in ("results_llm_only", "results_portal_xbrl") for p in paths):
    print("\nNo RAG results here yet. Run notebooks 01 to 07 with the same work folder, then rerun this cell.")

portal = pd.read_csv(CFG.work_dir / "results_portal_xbrl.csv").set_index("id")
for path in paths:
    if path.stem in ("results_llm_only", "results_portal_xbrl"):
        continue
    other = pd.read_csv(path).set_index("id")
    shared = portal.index.intersection(other.index)
    print(f"\n{path.stem.replace('results_', '')}: portal right and RAG wrong {sorted(i for i in shared if portal.correct[i] and not other.correct[i])}"
          f"\n  RAG right and portal wrong {sorted(i for i in shared if other.correct[i] and not portal.correct[i] and other.type[i] != 'negative')}")
'''

PORTAL_SAVE = r'''
try:
    from google.colab import files
    for name in ["portal_vs_rag_comparison.csv", "results_llm_only.csv", "results_portal_xbrl.csv"]:
        if (CFG.work_dir / name).exists():
            files.download(str(CFG.work_dir / name))
except ImportError:
    print("Results are in", CFG.work_dir)
'''


def build_portal():
    gold = json.dumps(EVAL_SET, indent=0)
    cells = [
        md("""# Portal without RAG, compared with RAG
### Annual report questions answered three ways, scored the same way

| Approach | Where the figure comes from |
|---|---|
| **LLM only** | Qwen's memory, with no documents and no internet |
| **Portal (SEC EDGAR XBRL)** | The SEC's data for one 10-K. Qwen only picks which fact to look up; code fetches it and writes the sentence |
| **RAG** | The annual report PDF, from the results of notebooks 01 to 07 |

**Keeping it fair:**
- The portal is limited to the FY2024 10-K, otherwise it could answer questions about years the report doesn't cover.
- Every approach uses the same 66 gold questions and the same scoring code, and the same Qwen model unless you change `QWEN_MODEL`.
- These questions were written from the annual report PDF, so they favour RAG. The table therefore also scores every approach on just the questions the portal can reach.
- Faithfulness isn't compared: LLM only has no evidence to check against, and the portal's sentence is built from the fact by code.

**To run:** choose a T4 GPU and enter your contact in `SEC_CONTACT`, as the SEC requires. To include RAG, use the same `DRIVE_FOLDER` as notebooks 01 to 07, then Runtime > Run all."""),
        md("## Step 1: Environment setup"),
        code(f"# Colab: Runtime > Change runtime type > T4 GPU first.\n!nvidia-smi -L\n!pip install -q {BASE_PACKAGES}", tags=["colab-install"]),
        code(COLAB_OPTIONS, tags=["colab-install"]),
        code(PORTAL_SETUP),
        code(CONFIG.split("def locate_pdf")[0].strip() + "\n\nGOLD_VALID = True  # this notebook reads no PDF; the gold questions are for Chevron FY2024"),
        md("## Step 2: Load Qwen in 4-bit through LangChain"),
        code(ANSWER),
        md("## Step 3: Gold questions and scoring helpers\nThe same code the seven RAG notebooks use."),
        code(GOLD_MODELS.strip() + "\n\nGOLD_SET = TypeAdapter(list[GoldQuestion]).validate_python(" + gold + ')\nGOLD_BY_QUESTION = {row.question.strip().lower(): row for row in GOLD_SET}\n\n' + GROUNDING.strip()),
        code(EVAL),
        md("## Step 4: Fetch the company's filings and facts from SEC EDGAR\n`submissions` lists the 10-K and annual report filings, and `companyfacts` holds the tagged XBRL figures. Downloads are saved and reused."),
        code(PORTAL_CODE),
        md("## Step 5: Answer without RAG\nIn the portal approach Qwen chooses a concept and code does the lookup, so a wrong answer means the wrong concept was chosen, not that a figure was invented."),
        code(PORTAL_ANSWER),
        md("## Step 6: Evaluate LLM only and the portal"),
        code(PORTAL_EVAL),
        md("""## Step 7: Compare with RAG
- **Number match:** the share of numeric questions answered with the gold figure.
- **Portal's questions:** the same score on only the questions whose figure is in the 10-K's XBRL. Use it to compare accuracy where the portal has the data.
- **Wrong when answering:** of the answers given to answerable questions, the share that were wrong. This is the risk of trusting an answer.
- **False refusals:** answerable questions that got no answer."""),
        code(PORTAL_COMPARE),
        md("## Step 8: Download the results"),
        code(PORTAL_SAVE),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "name": "08_portal_vs_rag.ipynb"},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    for arch in ARCHITECTURES:
        notebook = build(arch)
        for index, cell in enumerate(notebook["cells"]):
            cell["id"] = f"cell-{index:02d}"
        (HERE / arch["file"]).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", arch["file"], len(notebook["cells"]), "cells")
    notebook = build_portal()
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"
    (HERE / "08_portal_vs_rag.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote 08_portal_vs_rag.ipynb", len(notebook["cells"]), "cells")
