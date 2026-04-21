"""
payroll_extractor_fixed.py
--------------------------
Fixed version: handles all label variants found in the actual pay stub text.

Key fixes vs original:
  1. Regular time: also matches REG TIME / REG TIME HRS
  2. Overtime:     also matches OVERTIME (no HOURS suffix)
  3. Tax:          also matches TAX (singular) in addition to TAXES
  4. Week ending:  extracted from PROJECTION CHECK line  (Week Ending / WEEK ENDING)
  5. Total check:  also matches TOTAL PAY / TOTAL  (not just TOTAL CHECK)
  6. Per diem:     flexible X-connector handles DIEMX / DIEMXI / DIEM. / DIEM) / DIEM X
"""

import re
import csv
import sys
import glob
import os


# ---------------------------------------------------------------------------
# Page splitter
# ---------------------------------------------------------------------------

PAGE_SPLIT = re.compile(r"---\s*Page\s*\d+\s*---", re.IGNORECASE)


def split_pages(text: str) -> list[str]:
    parts = PAGE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(match, group=1, default=""):
    if match is None:
        return default
    try:
        v = match.group(group)
        return v.strip() if v else default
    except IndexError:
        return default


def _clean(val: str) -> str:
    """Remove commas from numbers."""
    return val.replace(",", "")


# ---------------------------------------------------------------------------
# Patterns — compiled once
# ---------------------------------------------------------------------------

# Regular time: REGULAR TIME HOURS | REG TIME HRS | REG TIME
_REG = re.compile(
    r"(?:REGULAR\s+TIME\s+HOURS?|REG\s+TIME\s+HRS?|REG\s+TIME)"
    r"\s+([\d.]+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# Overtime: OVERTIME HOURS | OVERTIME (hours value optional — may be 0 or blank)
_OT = re.compile(
    r"(?:OVERTIME\s+HOURS?|OVERTIME)"
    r"\s*([\d.]*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# Tax: TAXES | TAX  (followed by rate then dollar amount)
# NOTE: TAX(?:ES)? not TAXES? — the latter means TAXE + optional S
_TAX = re.compile(
    r"TAX(?:ES)?\s+([\d.]+)\s+\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# Week ending — on the PROJECTION CHECK line
# e.g. "PROJECTION CHECK Week Ending 01/11/2026"
#      "PROJECTION CHECK   WEEK ENDING 01/11/26"
_WEEK = re.compile(
    r"(?:PROJECTION\s+CHECK\s+)?WEEK?\s+ENDING\s+(\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE,
)

# Per diem — very flexible connector between DIEM and DAY
# Handles: DIEM X DAY / DIEMX DAY / DIEMXI DAY / DIEM. X DAY / DIEM) X DAY etc.
_PD = re.compile(
    r"\$([\d,]+\.?\d*)\s+PER\s+DIEM[^D\n]{0,6}DAY\s+"   # $rate PER DIEM...DAY
    r"([\d.]*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# Total check: TOTAL CHECK | TOTAL PAY | TOTAL  (but NOT SUBTOTAL)
# Use word-boundary after TOTAL to avoid SUBTOTAL
_TOTAL = re.compile(
    r"(?<!\w)TOTAL\s+(?:CHECK|PAY)?\s*\$?([\d,]+\.?\d*)",
    re.IGNORECASE,
)

# Employee name
_NAME = re.compile(
    r"EMPLOYEE\s+NAME\s*:\s*([A-Za-z][A-Za-z ',.\-]+?)(?=\n|Hours|Hrs|Total|$)",
    re.IGNORECASE,
)

# Optional deductions
_FEDERAL  = re.compile(r"FEDERAL\s+\$?([\d,]+\.?\d*)",                      re.IGNORECASE)
_STATE    = re.compile(r"\bSTATE\s+\$?([\d,]+\.?\d*)",                      re.IGNORECASE)
_CITY     = re.compile(r"\bCITY\s+\$?([\d,]+\.?\d*)",                       re.IGNORECASE)
_MEDICARE = re.compile(r"MEDICARE\s+\$?([\d,]+\.?\d*)",                      re.IGNORECASE)
_SS       = re.compile(r"(?:(?<!\w)SS(?!\w)|SOCIAL\s+SECURITY)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Per-page extractor
# ---------------------------------------------------------------------------

def extract_record(page: str) -> dict:
    rec = {}

    # Employee name
    m = _NAME.search(page)
    rec["EE Name"] = _g(m).strip().title() if m else ""

    # Week ending
    m = _WEEK.search(page)
    rec["Week Ending"] = _g(m)

    # Regular time
    m = _REG.search(page)
    rec["Regular Time Hours"] = _clean(_g(m, 1))
    rec["Regular Pay"]        = _clean(_g(m, 2))
    rec["Total Regular"]      = _clean(_g(m, 3))

    # Overtime
    m = _OT.search(page)
    rec["Overtime Hours"] = _clean(_g(m, 1)) or "0"
    rec["Overtime Pay"]   = _clean(_g(m, 2))
    rec["Total OT Pay"]   = _clean(_g(m, 3)) or "0.00"

    # Tax
    m = _TAX.search(page)
    rec["Tax Hours"] = _clean(_g(m, 1))   # tax rate
    rec["Tax"]       = _clean(_g(m, 2))

    # Per diem
    m = _PD.search(page)
    if m:
        rec["Per Diem Rate"]  = _clean(_g(m, 1))
        rec["Per Diem Days"]  = _clean(_g(m, 2))
        # group 3 = rate printed mid-line again, group 4 = total
        rec["Total Per Diem"] = _clean(_g(m, 4)) or _clean(_g(m, 3))
    else:
        rec["Per Diem Rate"]  = "100"
        rec["Per Diem Days"]  = "0"
        rec["Total Per Diem"] = "0.00"

    # Total check — take the LAST match to avoid hitting subtotals
    matches = list(_TOTAL.finditer(page))
    # Filter out SUBTOTAL hits (the regex lookbehind handles most; double-check)
    matches = [x for x in matches if "sub" not in page[max(0,x.start()-3):x.start()].lower()]
    rec["Check"] = _clean(_g(matches[-1]) if matches else None)

    # Optional deductions
    for pat, key in [(_FEDERAL,"Federal"),(_STATE,"State"),(_CITY,"City"),
                     (_MEDICARE,"Medicare"),(_SS,"Ss")]:
        m = pat.search(page)
        rec[key] = _clean(_g(m)) if m else ""

    return rec


# ---------------------------------------------------------------------------
# CSV layout
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "EE Name", "Week Ending",
    "Regular Time Hours", "Regular Pay", "Total Regular",
    "Overtime Hours", "Overtime Pay", "Total OT Pay",
    "Federal", "State", "City", "Medicare", "Ss",
    "Tax", "Tax Hours",
    "Per Diem Days", "Per Diem Rate", "Total Per Diem",
    "Check",
]

HEADER_DISPLAY = {c: c for c in CSV_COLUMNS}
HEADER_DISPLAY["Ss"] = "SS"


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def process_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    pages = split_pages(text)
    records = []
    for page in pages:
        rec = extract_record(page)
        if rec.get("EE Name") or rec.get("Check"):
            records.append(rec)
    return records


def write_csv(records: list[dict], output_path: str) -> None:
    display_headers = [HEADER_DISPLAY.get(c, c) for c in CSV_COLUMNS]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(display_headers) + "\n")
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writerows(records)
    print(f"  → Wrote {len(records)} record(s) to {output_path}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    input_pattern = args[0]
    output_csv    = args[1] if len(args) > 1 else "payroll_output.csv"

    files = sorted(glob.glob(input_pattern))
    if not files:
        if os.path.isfile(input_pattern):
            files = [input_pattern]
        else:
            print(f"No files matched: {input_pattern}")
            sys.exit(1)

    all_records = []
    for fp in files:
        print(f"Processing: {fp}")
        recs = process_file(fp)
        print(f"  Found {len(recs)} stub(s)")
        all_records.extend(recs)

    if not all_records:
        print("No records extracted. Check input file format.")
        sys.exit(1)

    write_csv(all_records, output_csv)
    print(f"\nDone. Total records: {len(all_records)}")


if __name__ == "__main__":
    main()