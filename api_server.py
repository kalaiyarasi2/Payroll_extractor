import os
import csv
import uuid
import sys
import subprocess
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Payroll Extractor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Universal logging (poc_db) ─────────────────────────────────────────────────
try:
    import importlib.util as _ilu
    _WORKSPACE_DIR = Path(__file__).resolve().parent.parent
    _poc_db_path = _WORKSPACE_DIR / "database" / "poc_db.py"
    if _poc_db_path.exists():
        _spec = _ilu.spec_from_file_location("poc_db", str(_poc_db_path))
        _poc_db = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_poc_db)
        _log_uni = _poc_db.log_universal
    else:
        _log_uni = None
except Exception as e:
    print(f"[Payroll] Universal logging init failed: {e}")
    _log_uni = None
# ───────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

@app.post("/process-pdf")
@app.post("/api/process-pdf")
async def process_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    job_id = str(uuid.uuid4())
    pdf_path = TEMP_DIR / f"{job_id}_{file.filename}"
    csv_path = TEMP_DIR / f"{job_id}_output.csv"
    txt_path = TEMP_DIR / f"{job_id}_output.txt"
    
    processed_by = request.headers.get("X-User-Email") or request.headers.get("x-user-email") or "SYSTEM"
    
    if _log_uni:
        _log_uni(
            module="Payroll Extractor",
            action="extract",
            file_name=file.filename,
            status="STARTED",
            details="Starting payroll PDF extraction",
            processed_by=processed_by
        )
        
    try:
        contents = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(contents)
            
        script_path = BASE_DIR / "pdf_processor_vision.py"
        
        command = [
            sys.executable,
            str(script_path),
            str(pdf_path),
            "--output_csv", str(csv_path)
        ]
        
        # Inject existing env to ensure OPENAI_API_KEY is available
        env = os.environ.copy()
        process = subprocess.run(command, capture_output=True, text=True, env=env)
        
        if process.returncode != 0:
            if _log_uni:
                _log_uni(
                    module="Payroll Extractor", action="extract", file_name=file.filename,
                    status="FAILED", details=f"Subprocess failed: {process.stderr[:200]}", processed_by=processed_by
                )
            raise HTTPException(status_code=500, detail=f"Extraction failed: {process.stderr}")
            
        if not csv_path.exists():
            if _log_uni:
                _log_uni(
                    module="Payroll Extractor", action="extract", file_name=file.filename,
                    status="FAILED", details="CSV output not found.", processed_by=processed_by
                )
            raise HTTPException(status_code=500, detail="Output CSV was not created. Logs: " + process.stdout[-200:])
            
        # Parse CSV into JSON array
        results = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                
        if _log_uni:
            _log_uni(
                module="Payroll Extractor",
                action="extract",
                file_name=file.filename,
                status="SUCCESS",
                details=f"Extracted {len(results)} rows",
                processed_by=processed_by
            )
            
        # Cleanup
        try:
            pdf_path.unlink()
            csv_path.unlink()
            if txt_path.exists():
                txt_path.unlink()
        except Exception:
            pass
            
        return JSONResponse(results)
        
    except HTTPException:
        raise
    except Exception as e:
        if _log_uni:
            _log_uni(
                module="Payroll Extractor", action="extract", file_name=file.filename,
                status="FAILED", details=str(e), processed_by=processed_by
            )
        raise HTTPException(status_code=500, detail=str(e))
