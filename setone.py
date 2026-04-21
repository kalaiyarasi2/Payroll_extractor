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
- HANDWRITTEN DATA: Pay close attention to handwritten notes or corrections. If a handwritten note updates a value, use the handwritten value.

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
        return page_num, result
    except Exception as e:
        print(f"Error processing page {page_num + 1}: {e}")
        return page_num, None

def main():
    parser = argparse.ArgumentParser(description="PDF to CSV with Vision extraction")
    parser.add_argument("pdf_input", help="Path to the PDF file")
    parser.add_argument("--output_csv", default="payroll_extracted.csv", help="Output CSV filename")
    parser.add_argument("--output_txt", default="audit_transcription.txt", help="Output audit text filename")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--max_pages", type=int, default=None, help="Limit number of pages to process")
    args = parser.parse_args()

    # Automatically set output names based on PDF name if not specified
    base_name = os.path.splitext(os.path.basename(args.pdf_input))[0]
    if args.output_csv == "payroll_extracted.csv":
        args.output_csv = f"{base_name}_vision.csv"
    if args.output_txt == "audit_transcription.txt":
        args.output_txt = f"{base_name}_audit.txt"

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
            print(f"Completed page {idx + 1}/{num_pages}")

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
