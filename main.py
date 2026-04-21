"""
payroll_extractor.py
--------------------
Extracts payroll fields from plain-text pay stub files (one or more pages/stubs
per file) and writes a CSV matching the spreadsheet layout shown in the screenshot.

CSV columns (matching screenshot):
  EE Name, Week Ending, Regular Time Hours, Regular Pay, Total Regular,
  Overtime Hours, Overtime Pay, Total OT Pay, Federal, State, City,
  Medicare, SS, Tax, Tax Hours, Per Diem Days, Per Diem Rate, Total Per Diem, Check

Usage:
  python payroll_extractor.py <input_file_or_glob> [output.csv]

Examples:
  python payroll_extractor.py stubs.txt
  python payroll_extractor.py stubs.txt output.csv
  python payroll_extractor.py "*.txt" all_employees.csv
"""

import re
import csv
import sys
import glob
import os

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _money(pattern_body: str) -> str:
    """Wrap a label pattern so it captures an optional $-prefixed decimal number."""
    return rf"{pattern_body}\s*\$?([\d,]+\.?\d*)"


def _opt_money(pattern_body: str) -> str:
    """Same as _money but the value is optional (may be absent or zero)."""
    return rf"{pattern_body}\s*\$?([\d,]*\.?\d*)"


# Each pattern returns (group_index_map, compiled_regex) tuples.
# We search the full page text (case-insensitive, DOTALL).

PATTERNS = {
    "employee_name": re.compile(
        r"EMPLOYEE\s+NAME[:\s]+([A-Za-z][A-Za-z ',.\-]+?)(?=\n|Hours|$)",
        re.IGNORECASE,
    ),
    "week_ending": re.compile(
        r"Week\s+Ending\s+(\d{1,2}/\d{1,2}/\d{2,4})",
        re.IGNORECASE,
    ),
    # Regular time: hours  rate  total  — three consecutive dollar amounts
    # Pattern: REGULAR TIME HOURS  <hrs>  $<rate>  $<total>
    "regular_block": re.compile(
        r"REGULAR\s+TIME\s+HOURS\s+([\d.]+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # Overtime block — hours might be missing (0/blank)
    "overtime_block": re.compile(
        r"OVERTIME\s+HOURS\s*([\d.]*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # Tax line: TAXES  <rate_decimal>  $<amount>
    "tax_block": re.compile(
        r"TAXES\s+([\d.]+)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # Per diem: $100 PER DIEM X DAY  <days>  $<rate>  $<total>
    "per_diem_block": re.compile(
        r"\$([\d,]+\.?\d*)\s+PER\s+DIEM\s+X\s+DAY\s+([\d.]*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # Alternate per diem without leading $: PER DIEM X DAY  <days>  $<rate>  $<total>
    "per_diem_alt": re.compile(
        r"PER\s+DIEM\s+X\s+DAY\s+([\d.]*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # Total check
    "total_check": re.compile(
        r"TOTAL\s+CHECK\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    ),
    # Optional deduction lines (Federal / State / City / Medicare / SS)
    "federal":   re.compile(r"FEDERAL\s+\$?([\d,]+\.?\d*)",   re.IGNORECASE),
    "state":     re.compile(r"STATE\s+\$?([\d,]+\.?\d*)",     re.IGNORECASE),
    "city":      re.compile(r"CITY\s+\$?([\d,]+\.?\d*)",      re.IGNORECASE),
    "medicare":  re.compile(r"MEDICARE\s+\$?([\d,]+\.?\d*)",  re.IGNORECASE),
    "ss":        re.compile(r"(?:SS|SOCIAL\s+SECURITY)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE),
}


# ---------------------------------------------------------------------------
# Page splitter
# ---------------------------------------------------------------------------

PAGE_SPLIT = re.compile(r"---\s*Page\s*\d+\s*---", re.IGNORECASE)


def split_pages(text: str) -> list[str]:
    """Split text on '--- Page N ---' markers; fall back to the whole text."""
    parts = PAGE_SPLIT.split(text)
    pages = [p.strip() for p in parts if p.strip()]
    return pages if pages else [text.strip()]


# ---------------------------------------------------------------------------
# Per-page extractor
# ---------------------------------------------------------------------------

def _g(match, group=1, default=""):
    """Safely retrieve a regex group, returning default if no match."""
    if match is None:
        return default
    try:
        return match.group(group).strip()
    except IndexError:
        return default


def extract_record(page_text: str) -> dict:
    """Extract one payroll record from a single page/stub text block."""

    rec = {}

    # --- Employee Name ---
    m = PATTERNS["employee_name"].search(page_text)
    rec["EE Name"] = _g(m).title()

    # --- Week Ending ---
    m = PATTERNS["week_ending"].search(page_text)
    rec["Week Ending"] = _g(m)

    # --- Regular Time ---
    m = PATTERNS["regular_block"].search(page_text)
    rec["Regular Time Hours"] = _g(m, 1)
    rec["Regular Pay"]        = _g(m, 2)   # hourly rate
    rec["Total Regular"]      = _g(m, 3)   # gross regular pay

    # --- Overtime ---
    m = PATTERNS["overtime_block"].search(page_text)
    rec["Overtime Hours"] = _g(m, 1) or "0"
    rec["Overtime Pay"]   = _g(m, 2)       # OT hourly rate
    rec["Total OT Pay"]   = _g(m, 3) or "0.00"

    # --- Tax ---
    m = PATTERNS["tax_block"].search(page_text)
    rec["Tax Hours"] = _g(m, 1)   # tax rate (e.g. 0.15)
    rec["Tax"]       = _g(m, 2)   # dollar amount withheld

    # --- Per Diem ---
    m = PATTERNS["per_diem_block"].search(page_text)
    if m:
        rec["Per Diem Rate"]   = _g(m, 1)   # rate per day (e.g. 100)
        rec["Per Diem Days"]   = _g(m, 2)
        # group 3 = rate again printed mid-line, group 4 = total
        rec["Total Per Diem"]  = _g(m, 4) or _g(m, 3)
    else:
        m2 = PATTERNS["per_diem_alt"].search(page_text)
        rec["Per Diem Days"]  = _g(m2, 1) or "0"
        rec["Per Diem Rate"]  = _g(m2, 2) or "100"
        rec["Total Per Diem"] = _g(m2, 3) or "0.00"

    # --- Optional deductions (may be absent in simpler stubs) ---
    for key in ("federal", "state", "city", "medicare", "ss"):
        m = PATTERNS[key].search(page_text)
        col = key.title()
        rec[col] = _g(m) if m else ""

    # --- Total Check ---
    m = PATTERNS["total_check"].search(page_text)
    rec["Check"] = _g(m)

    return rec


# ---------------------------------------------------------------------------
# CSV column order (matches screenshot)
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "EE Name",
    "Week Ending",
    "Regular Time Hours",
    "Regular Pay",
    "Total Regular",
    "Overtime Hours",
    "Overtime Pay",
    "Total OT Pay",
    "Federal",
    "State",
    "City",
    "Medicare",
    "Ss",           # SS column
    "Tax",
    "Tax Hours",
    "Per Diem Days",
    "Per Diem Rate",
    "Total Per Diem",
    "Check",
]

# Map internal key "Ss" back to display header "SS"
HEADER_DISPLAY = {c: c for c in CSV_COLUMNS}
HEADER_DISPLAY["Ss"] = "SS"


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_file(filepath: str) -> list[dict]:
    """Read a text file and return a list of extracted payroll records."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    pages = split_pages(text)
    records = []
    for page in pages:
        if not page:
            continue
        rec = extract_record(page)
        # Skip completely empty records (no name, no check amount)
        if rec.get("EE Name") or rec.get("Check"):
            records.append(rec)
    return records


def write_csv(records: list[dict], output_path: str) -> None:
    """Write records to a CSV file."""
    display_headers = [HEADER_DISPLAY.get(c, c) for c in CSV_COLUMNS]
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )
        # Write display headers manually
        fh.write(",".join(display_headers) + "\n")
        writer.writerows(records)
    print(f"  → Wrote {len(records)} record(s) to {output_path}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    input_pattern = args[0]
    output_csv    = args[1] if len(args) > 1 else "payroll_output.csv"

    # Expand glob (e.g. "*.txt")
    files = sorted(glob.glob(input_pattern))
    if not files:
        # Maybe it's a literal path
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