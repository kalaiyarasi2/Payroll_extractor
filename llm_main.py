"""
gpt_payroll_extractor.py
------------------------
GPT-powered payroll stub extractor using the OpenAI API.
- Chunks input by page (--- Page N --- delimiters)
- Processes pages in parallel using ThreadPoolExecutor
- Falls back gracefully; validates JSON before accepting
- Outputs the same CSV columns as the original regex script

Usage:
    python gpt_payroll_extractor.py <input.txt> [output.csv] [--workers N] [--batch B] [--model M]

Defaults:
    output  = payroll_output_gpt.csv
    workers = 8                      (parallel threads)
    batch   = 5                      (pages per GPT call — reduces API round trips)
    model   = gpt-4o                 (best accuracy; use gpt-4o-mini for lower cost)

Requirements:
    pip install openai

API key:
    Set env var  : export OPENAI_API_KEY=sk-...
    Or CLI flag  : --api-key sk-...
"""

import re
import csv
import sys
import json
import time
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv


# ── OpenAI client (initialised in main() after args parsed) ──────────────────
client = None
MODEL  = "gpt-4o"

# ── CSV layout (matches original regex script exactly) ────────────────────────
CSV_COLUMNS = [
    "EE Name", "Week Ending",
    "Regular Time Hours", "Regular Pay", "Total Regular",
    "Overtime Hours", "Overtime Pay", "Total OT Pay",
    "Federal", "State", "City", "Medicare", "SS",
    "Tax", "Tax Hours",
    "Per Diem Days", "Per Diem Rate", "Total Per Diem",
    "Check",
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM = """You are a precise payroll data extractor.
You will receive one or more pay stub pages (separated by --- Page N --- markers).
For EACH page extract these fields and return a JSON array (one object per page):

{
  "EE Name":             "Employee full name, Title Case",
  "Week Ending":         "Date as found (e.g. 2/15/26 or 02/15/26)",
  "Regular Time Hours":  "numeric string or empty",
  "Regular Pay":         "hourly rate as numeric string or empty",
  "Total Regular":       "total regular pay as numeric string or empty",
  "Overtime Hours":      "numeric string, 0 if absent",
  "Overtime Pay":        "OT hourly rate as numeric string or empty",
  "Total OT Pay":        "total OT pay, 0.00 if absent",
  "Federal":             "federal tax amount or empty",
  "State":               "state tax amount or empty",
  "City":                "city tax amount or empty",
  "Medicare":            "medicare amount or empty",
  "SS":                  "social security amount or empty",
  "Tax":                 "total tax dollar amount withheld (e.g. 308.00)",
  "Tax Hours":           "tax RATE (decimal like 0.20) not hours",
  "Per Diem Days":       "number of per diem days, 0 if absent",
  "Per Diem Rate":       "per diem daily rate (e.g. 100), 100 if standard",
  "Total Per Diem":      "total per diem pay, 0.00 if absent",
  "Check":               "final take-home / total check amount"
}

Rules:
- Numbers must NOT contain $ signs or commas.
- "Tax" = the dollar amount withheld (e.g. TAX 0.20 $308.00 → Tax=308.00).
- "Tax Hours" = the rate / percentage (e.g. 0.20 from above).
- Regular Pay = the hourly rate column (e.g. $55.00 → 55.00).
- Total Regular = hours × rate product (e.g. $1,540.00 → 1540.00).
- Same logic for Overtime Pay (rate) vs Total OT Pay (product).
- If a field is truly absent, use empty string "" (not null, not "N/A").
- Return ONLY the raw JSON array, no markdown, no explanation.
"""

# ── Page splitter ─────────────────────────────────────────────────────────────
PAGE_RE = re.compile(r"---\s*Page\s*\d+\s*---", re.IGNORECASE)

def split_pages(text: str) -> list[tuple[int, str]]:
    """Return list of (page_number, page_text) tuples."""
    parts   = PAGE_RE.split(text)
    numbers = [int(m) for m in re.findall(r"---\s*Page\s*(\d+)\s*---", text, re.IGNORECASE)]
    pages   = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        page_num = numbers[i - 1] if i > 0 and (i - 1) < len(numbers) else i
        pages.append((page_num, part))
    return pages

# ── GPT call (with retry) ─────────────────────────────────────────────────────
def call_llm(batch_text: str, page_nums: list[int], retries: int = 3) -> list[dict]:
    """Send a batch of page text to GPT, return list of extracted records."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model       = MODEL,
                temperature = 0,          # deterministic extraction
                max_tokens  = 2000,
                response_format = {"type": "json_object"},   # JSON mode — no fences
                messages    = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": batch_text},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # JSON mode may return {"records": [...]} or bare [...] — handle both
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # unwrap first list value found
                records = next(
                    (v for v in parsed.values() if isinstance(v, list)),
                    [parsed],   # single record wrapped in dict
                )
            elif isinstance(parsed, list):
                records = parsed
            else:
                records = [parsed]

            # Fill missing keys with ""
            cleaned = []
            for rec in records:
                entry = {col: str(rec.get(col, "")).strip() for col in CSV_COLUMNS}
                cleaned.append(entry)
            return cleaned

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            wait = 2 ** attempt
            print(f"  [warn] Parse error on pages {page_nums} (attempt {attempt+1}): {e}. Retrying in {wait}s…")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [warn] API error on pages {page_nums} (attempt {attempt+1}): {e}. Retrying in {wait}s…")
            time.sleep(wait)

    print(f"  [error] Failed pages {page_nums} after {retries} attempts. Skipping.")
    return []

# ── Batch builder ─────────────────────────────────────────────────────────────
def make_batches(pages: list[tuple[int, str]], batch_size: int) -> list[tuple[list[int], str]]:
    """Group pages into batches. Returns (page_nums_list, combined_text)."""
    batches = []
    for i in range(0, len(pages), batch_size):
        chunk = pages[i : i + batch_size]
        nums  = [p[0] for p in chunk]
        text  = "\n\n".join(f"--- Page {n} ---\n{t}" for n, t in chunk)
        batches.append((nums, text))
    return batches

# ── CSV writer ────────────────────────────────────────────────────────────────
def write_csv(records: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\n✓ Wrote {len(records)} record(s) to {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GPT payroll extractor")
    parser.add_argument("input",            help="Input .txt file")
    parser.add_argument("output", nargs="?", default="payroll_output_gpt.csv")
    parser.add_argument("--workers", type=int,  default=8,      help="Parallel threads (default 8)")
    parser.add_argument("--batch",   type=int,  default=5,      help="Pages per GPT call (default 5)")
    parser.add_argument("--model",             default="gpt-4o", help="OpenAI model (default gpt-4o)")
    parser.add_argument("--api-key", dest="api_key", default=None,
                        help="OpenAI API key (overrides OPENAI_API_KEY env var)")
    args = parser.parse_args()

    load_dotenv() # Load environment variables from .env file

    # Initialise client
    global client, MODEL
    MODEL = args.model
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    client = OpenAI()   # reads OPENAI_API_KEY automatically

    # Read input
    with open(args.input, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    pages   = split_pages(text)
    batches = make_batches(pages, args.batch)

    print(f"Input  : {args.input}")
    print(f"Model  : {MODEL}")
    print(f"Pages  : {len(pages)}")
    print(f"Batches: {len(batches)}  (batch size={args.batch})")
    print(f"Workers: {args.workers}")
    print(f"Output : {args.output}\n")

    # ── Parallel extraction ────────────────────────────────────────────────────
    all_records: list[tuple[int, dict]] = []
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(call_llm, batch_text, page_nums): page_nums
            for page_nums, batch_text in batches
        }

        done = 0
        for future in as_completed(future_map):
            page_nums = future_map[future]
            records   = future.result()
            done     += 1
            print(f"  [{done:>3}/{len(batches)}] Pages {page_nums} → {len(records)} record(s)")

            for i, rec in enumerate(records):
                order_key = page_nums[0] * 100 + i
                all_records.append((order_key, rec))

    elapsed = time.perf_counter() - t0
    print(f"\nExtraction complete in {elapsed:.1f}s")

    # Re-sort by original page order
    all_records.sort(key=lambda x: x[0])
    final = [r for _, r in all_records]

    if not final:
        print("No records extracted. Check input format.")
        sys.exit(1)

    write_csv(final, args.output)
    print(f"Total records : {len(final)}")
    print(f"Time taken    : {elapsed:.1f}s  ({elapsed/len(pages)*1000:.0f} ms/page avg)")


if __name__ == "__main__":
    main()