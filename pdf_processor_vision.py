import os
import re
import csv
import json
import time
import base64
import argparse
import fitz  # PyMuPDF
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
CSV_COLUMNS = [
    "EE Name", "Week Ending", "Regular Time Hours", "Regular Pay", "Total Regular",
    "Overtime Hours", "Overtime Pay", "Total OT Pay", "Federal", "State", "City",
    "Medicare", "SS", "Tax", "Tax Hours", "Per Diem Days", "Per Diem Rate", 
    "Total Per Diem", "Check"
]

# --- Regex Patterns (from main_fixed.py) ---
_REG = re.compile(r"(?:REGULAR\s+TIME\s+HOURS?|REG\s+TIME\s+HRS?|REG\s+TIME)\s+([\d.]+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_OT = re.compile(r"(?:OVERTIME\s+HOURS?|OVERTIME)\s*([\d.]*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_TAX = re.compile(r"TAX(?:ES)?\s+([\d.]+)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_WEEK = re.compile(r"(?:PROJECTION\s+CHECK\s+)?WEEK?\s+ENDING\s+(\d{1,2}/\d{1,2}/\d{2,4})", re.IGNORECASE)
_PD = re.compile(r"\$([\d,]+\.?\d*)\s+PER\s+DIEM[^D\n]{0,6}DAY\s+([\d.]*)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_TOTAL = re.compile(r"(?<!\w)TOTAL\s+(?:CHECK|PAY)?\s*\$?([\d,]+\.?\d*)", re.IGNORECASE)
_NAME = re.compile(r"EMPLOYEE\s+NAME\s*:\s*([A-Za-z][A-Za-z ',.\-]+?)(?=\n|Hours|Hrs|Total|$)", re.IGNORECASE)
_FEDERAL  = re.compile(r"FEDERAL\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_STATE    = re.compile(r"\bSTATE\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_CITY     = re.compile(r"\bCITY\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_MEDICARE = re.compile(r"MEDICARE\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)
_SS       = re.compile(r"(?:(?<!\w)SS(?!\w)|SOCIAL\s+SECURITY)\s+\$?([\d,]+\.?\d*)", re.IGNORECASE)

SYSTEM_PROMPT = """You are a highly accurate payroll data extractor specializing in vision tasks.
You will see an image of a pay stub page. Some information might be handwritten or present in notes.
Your task is to:
1. Extract the structured payroll data into a JSON format.
2. Provide a full text transcription of the page for audit purposes.

### Extraction Rules:
- EE Name: Employee full name.
- Week Ending: Date of the week ending.
- Regular Time Hours: The number of regular hours.
- Regular Pay: This is the HOURLY RATE for regular time (often found under the "RATE" header). DO NOT LEAVE THIS EMPTY IF A RATE IS VISIBLE.
- Total Regular: This is the TOTAL regular pay (Hours x Rate).
- Overtime Hours: The number of overtime hours.
- Overtime Pay: This is the HOURLY RATE for overtime (often found under the "RATE" header). DO NOT LEAVE THIS EMPTY IF A RATE IS VISIBLE.
- Total OT Pay: This is the TOTAL overtime pay (Hours x Rate).
- Federal, State, City, Medicare, SS: Tax amounts.
- Tax: Total dollar amount withheld for taxes.
- Tax Hours: This represents the tax RATE/PERCENTAGE (e.g., 0.15 or 0.30) if available.
- Per Diem Days, Per Diem Rate, Total Per Diem: Extract amounts.
- Check: The final net check amount. 
  ***CRITICAL***: Look closely for handwritten numbers or calculations at the bottom of the page (often below the printed 'TOTAL CHECK' or 'PROJECTION CHECK' lines). If a handwritten summation or total is present, use that final handwritten value as the 'Check' amount. It overrides the printed value.
- HANDWRITTEN DATA: Pay close attention to handwritten notes or corrections anywhere on the page. If a handwritten note updates a value (e.g., hours or check amount), use the handwritten value.

### Important Column Mapping:
In the PDF, you will see a column titled "RATE". 
- The first value under "RATE" always maps to "Regular Pay".
- The second value under "RATE" (on the Overtime line) always maps to "Overtime Pay".
Ensure you capture these as individual hourly rates ($xx.xx), not the totals.

### Output Format:
Return a JSON object with two keys:
1. "data": An object containing the extracted fields matching the CSV columns.
2. "transcription": A string containing the full text seen on the page, including any notes.

JSON example:
{
  "data": {
    "EE Name": "John Doe",
    "Week Ending": "3/22/2026",
    "Regular Time Hours": "40",
    "Regular Pay": "35.00",
    "Total Regular": "1400.00",
    ...
  },
  "transcription": "Electrical Source LLC... [Full text here]"
}

Return ONLY the raw JSON. No markdown fences.
"""

MODEL = "gpt-4o"

def _g(match, group=1, default=""):
    if match is None: return default
    try:
        v = match.group(group)
        return v.strip() if v else default
    except IndexError: return default

def _clean(val):
    if not val: return ""
    return val.replace(",", "").replace("$", "").strip()

def regex_extract(text):
    rec = {col: "" for col in CSV_COLUMNS}
    
    # Employee name
    m = _NAME.search(text)
    rec["EE Name"] = _g(m).strip().title() if m else ""

    # Week ending
    m = _WEEK.search(text)
    rec["Week Ending"] = _g(m)

    # Regular time
    m = _REG.search(text)
    rec["Regular Time Hours"] = _clean(_g(m, 1))
    rec["Regular Pay"]        = _clean(_g(m, 2))
    rec["Total Regular"]      = _clean(_g(m, 3))

    # Overtime
    m = _OT.search(text)
    rec["Overtime Hours"] = _clean(_g(m, 1)) or "0"
    rec["Overtime Pay"]   = _clean(_g(m, 2))
    rec["Total OT Pay"]   = _clean(_g(m, 3)) or "0.00"

    # Tax
    m = _TAX.search(text)
    rec["Tax Hours"] = _clean(_g(m, 1))
    rec["Tax"]       = _clean(_g(m, 2))

    # Per diem
    m = _PD.search(text)
    if m:
        rec["Per Diem Rate"]  = _clean(_g(m, 1))
        rec["Per Diem Days"]  = _clean(_g(m, 2))
        rec["Total Per Diem"] = _clean(_g(m, 4)) or _clean(_g(m, 3))
    else:
        rec["Per Diem Rate"]  = "100"
        rec["Per Diem Days"]  = "0"
        rec["Total Per Diem"] = "0.00"

    # Total check
    matches = list(_TOTAL.finditer(text))
    # Filter out SUBTOTAL hits
    matches = [x for x in matches if "sub" not in text[max(0, x.start()-3):x.start()].lower()]
    rec["Check"] = _clean(_g(matches[-1]) if matches else "")

    # Optional deductions
    for pat, key in [(_FEDERAL, "Federal"), (_STATE, "State"), (_CITY, "City"),
                     (_MEDICARE, "Medicare"), (_SS, "SS")]:
        m = pat.search(text)
        rec[key] = _clean(_g(m)) if m else ""

    return rec

def has_handwriting_indicators(text):
    # Detect if there are notes/handwriting
    keywords = ["miss", "owe", "note", "hand", "written", "adj", "calc", "handwritten"]
    lower_text = text.lower()
    
    # Check for keywords
    if any(kw in lower_text for kw in keywords):
        return True
    
    # Check for text after standard block
    lines = text.split('\n')
    found_end = False
    extra_content = []
    for line in lines:
        if found_end:
            if line.strip() and len(line.strip()) > 3:
                extra_content.append(line)
        if "PROJECTION CHECK" in line.upper() or "TOTAL CHECK" in line.upper():
            found_end = True
            
    if len(extra_content) > 1: # If more than 1 meaningful line after summary
        return True
        
    return False

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def extract_page_as_image(pdf_path, page_num, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img_data = pix.tobytes("png")
    doc.close()
    return img_data

def process_page(client, pdf_path, page_num):
    try:
        # 1. Try Fast Text Extraction (Best for digital PDFs)
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num)
        text = page.get_text()
        doc.close()
        
        # 2. Run Regex Extraction
        regex_res = regex_extract(text)
        
        # 3. Decision Logic: Use GPT if data is missing or handwriting is suspected
        is_missing = not regex_res.get("EE Name") or not regex_res.get("Check") or not regex_res.get("Regular Pay")
        has_notes = has_handwriting_indicators(text)
        
        # If it's a scan (very little text), text will be empty
        is_scan = len(text.strip()) < 100
        
        if is_missing or has_notes or is_scan:
            reason = []
            if is_missing: reason.append("missing data")
            if has_notes: reason.append("handwritten notes")
            if is_scan: reason.append("likely scan")
            
            print(f"Page {page_num + 1}: Using GPT ({', '.join(reason)})")
            
            img_bytes = extract_page_as_image(pdf_path, page_num)
            base64_image = encode_image(img_bytes)

            response = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Process page {page_num + 1} of this payroll document."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=2000
            )

            content = response.choices[0].message.content.strip()
            result = json.loads(content)
            result["method"] = "gpt-vision"
            return page_num, result
        else:
            # print(f"Page {page_num + 1}: Using Fast Regex extraction")
            return page_num, {
                "data": regex_res, 
                "transcription": text,
                "method": "regex-fast"
            }
    except Exception as e:
        print(f"Error processing page {page_num + 1}: {e}")
        return page_num, None

def main():
    parser = argparse.ArgumentParser(description="PDF to CSV with Vision extraction")
    parser.add_argument("pdf_input", nargs="?", help="Path to the PDF file")
    parser.add_argument("--output_csv", default="payroll_extracted.csv", help="Output CSV filename")
    parser.add_argument("--output_txt", default="audit_transcription.txt", help="Output audit text filename")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--max_pages", type=int, default=None, help="Limit number of pages to process")
    args = parser.parse_args()

    # If pdf_input is not provided, prompt the user
    if not args.pdf_input:
        print("\n--- Payroll Data Extractor ---")
        args.pdf_input = input("Please enter the path to the PDF file: ").strip()
        # Remove quotes if the user wrapped the path in them
        if (args.pdf_input.startswith('"') and args.pdf_input.endswith('"')) or \
           (args.pdf_input.startswith("'") and args.pdf_input.endswith("'")):
            args.pdf_input = args.pdf_input[1:-1]
            
    if not args.pdf_input:
        print("Error: No PDF input provided.")
        return

    if not os.path.exists(args.pdf_input):
        print(f"Error: File not found: {args.pdf_input}")
        return

    # Ensure directories exist
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("extracted_text", exist_ok=True)

    # Automatically set output names based on PDF name if not specified
    base_name = os.path.splitext(os.path.basename(args.pdf_input))[0]
    if args.output_csv == "payroll_extracted.csv":
        args.output_csv = os.path.join("outputs", f"{base_name}_vision.csv")
    if args.output_txt == "audit_transcription.txt":
        args.output_txt = os.path.join("extracted_text", f"{base_name}_audit.txt")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    client = OpenAI(api_key=api_key)

    doc = fitz.open(args.pdf_input)
    num_pages = len(doc)
    doc.close()

    if args.max_pages:
        num_pages = min(num_pages, args.max_pages)

    print(f"Processing {num_pages} pages from {args.pdf_input}...")
    
    results = [None] * num_pages
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_page, client, args.pdf_input, i): i for i in range(num_pages)}
        
        for future in as_completed(futures):
            idx, res = future.result()
            results[idx] = res
            status = f" (Method: {res.get('method', 'unknown')})" if res else " (Failed)"
            print(f"Completed page {idx + 1}/{num_pages}{status}")

    # Prepare data for files
    all_data = []
    full_transcription = []

    for i, res in enumerate(results):
        if res and "data" in res:
            record = res["data"]
            # Ensure all columns exist and handle None values
            cleaned_record = {}
            for col in CSV_COLUMNS:
                val = record.get(col, "")
                if val is None:
                    val = ""
                cleaned_record[col] = str(val).strip()
            all_data.append(cleaned_record)
            
            transcription = res.get("transcription", "")
            full_transcription.append(f"--- Page {i + 1} ---\n{transcription}\n")
        else:
            print(f"Warning: Missing data for page {i + 1}")

    # Save CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_data)

    # Save Text
    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(full_transcription))

    print(f"\nDone!")
    print(f"CSV saved to: {args.output_csv}")
    print(f"Audit text saved to: {args.output_txt}")

if __name__ == "__main__":
    main()
