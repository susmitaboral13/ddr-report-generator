"""
DDR Report Generator — Groq API Backend
Free, fast, no credit card required.
Model: llama3-8b-8192 via Groq (free tier)
"""

import os, io, re, base64, json, logging, time, fitz
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from typing import Optional
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ddr")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"
MAX_FILE_MB  = 20
MAX_TEXT_CHARS = 4000

app = FastAPI(title="DDR Report Generator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache: dict = {}

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Extraction ─────────────────────────────────────────────────────────────────
def ocr_page(page):
    if not OCR_AVAILABLE:
        return ""
    try:
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return pytesseract.image_to_string(img).strip()
    except:
        return ""

def extract_pdf(data: bytes, label: str) -> dict:
    doc = fitz.open(stream=data, filetype="pdf")
    pages, images = [], []
    for pnum, page in enumerate(doc, 1):
        text = page.get_text("text").strip()
        if len(text) < 50:
            ocr = ocr_page(page)
            if len(ocr) > len(text):
                text = ocr
        if text:
            pages.append(f"[Page {pnum}]\n{text}")
        for idx, ref in enumerate(page.get_images(full=True)):
            try:
                bi  = doc.extract_image(ref[0])
                pil = Image.open(io.BytesIO(bi["image"])).convert("RGB")
                if pil.width < 100 or pil.height < 100:
                    continue
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=75)
                images.append({
                    "caption": f"{label} — Page {pnum}, Image {idx+1}",
                    "base64":  base64.b64encode(buf.getvalue()).decode(),
                    "media_type": "image/jpeg",
                })
            except:
                pass
    doc.close()
    return {"text": "\n\n".join(pages) or "[No text extracted]", "images": images}

def extract_image_file(data: bytes, filename: str, label: str) -> dict:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return {
        "text": f"[{label}: image file '{filename}']",
        "images": [{"caption": f"{label} — {filename}",
                    "base64": base64.b64encode(buf.getvalue()).decode(),
                    "media_type": "image/jpeg"}]
    }

# ── Cross-reference ────────────────────────────────────────────────────────────
AREAS = [r'\b(roof|ceiling|wall|floor|bathroom|kitchen|bedroom|electrical|plumbing|foundation|basement|window|door|balcony|garage|attic|staircase)\b']

def find_areas(text):
    found = set()
    for pat in AREAS:
        for m in re.finditer(pat, text.lower()):
            found.add(m.group(0))
    return found

def crossref(insp, therm):
    common = find_areas(insp["text"]) & find_areas(therm["text"])
    if common:
        return f"Areas in BOTH documents (cross-reference these): {', '.join(sorted(common))}"
    return "Cross-reference all findings by context."

# ── Prompt ─────────────────────────────────────────────────────────────────────
def build_prompt(insp, therm, xref):
    it = insp["text"][:MAX_TEXT_CHARS]
    tt = therm["text"][:MAX_TEXT_CHARS]
    return f"""You are a property inspector. Read both documents and write a Detailed Diagnostic Report (DDR) as JSON.

INSPECTION REPORT:
{it}

THERMAL REPORT:
{tt}

CROSS-REFERENCE: {xref}

RULES:
- Link thermal readings to matching visual observations
- Never invent facts not in the documents
- Missing info = "Not Available"
- Plain English only
- Each finding in exactly one section

Return ONLY valid JSON, nothing else:
{{
  "propertyName": "from documents or Not Available",
  "reportDate": "from documents",
  "inspectorName": "from documents or Not Available",
  "sections": [
    {{"id":"summary","number":"01","title":"Property Issue Summary","content":"Overview of all issues from both reports combined.","images":[]}},
    {{"id":"observations","number":"02","title":"Area-wise Observations","content":"**Area:** observation + thermal reading for each area.","images":[]}},
    {{"id":"rootcause","number":"03","title":"Probable Root Cause","content":"Likely cause for each issue in simple language.","images":[]}},
    {{"id":"severity","number":"04","title":"Severity Assessment","content":"Issue | HIGH or MEDIUM or LOW | reason. One per line.","images":[]}},
    {{"id":"actions","number":"05","title":"Recommended Actions","content":"1. Most urgent. 2. Next. etc.","images":[]}},
    {{"id":"notes","number":"06","title":"Additional Notes","content":"Other relevant findings.","images":[]}},
    {{"id":"missing","number":"07","title":"Missing or Unclear Information","content":"What data was missing or unclear.","images":[]}}
  ],
  "conflicts": "Any contradictions between reports, or: None identified",
  "disclaimer": "AI-generated report. Verify with a certified inspector before taking action."
}}"""

# ── Groq API call ──────────────────────────────────────────────────────────────
async def call_groq(prompt: str) -> dict:
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not set. Get a free key at console.groq.com")

    for attempt in range(1, 4):
        try:
            log.info(f"Groq attempt {attempt}, prompt: {len(prompt)} chars")
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type":  "application/json",
                    },
                    json={
                        "model":       GROQ_MODEL,
                        "messages":    [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens":  3000,
                    }
                )

            if resp.status_code == 200:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"```$",          "", raw, flags=re.MULTILINE)
                raw = raw.strip()
                start, end = raw.find("{"), raw.rfind("}")
                if start != -1 and end != -1:
                    raw = raw[start:end+1]
                result = json.loads(raw)
                if "sections" not in result:
                    raise ValueError("Missing sections")
                return result

            elif resp.status_code == 429:
                log.warning("Rate limited, waiting...")
                time.sleep(10)
                continue
            else:
                raise HTTPException(502, f"Groq error {resp.status_code}: {resp.text[:200]}")

        except json.JSONDecodeError as e:
            log.warning(f"JSON parse fail attempt {attempt}: {e}")
            if attempt < 3:
                time.sleep(2)
        except httpx.TimeoutException:
            log.warning(f"Timeout attempt {attempt}")
            if attempt < 3:
                time.sleep(3)

    raise HTTPException(502, "Failed to generate report after 3 attempts. Try again.")

# ── PDF export ─────────────────────────────────────────────────────────────────
def make_pdf(report: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    s_title = ParagraphStyle("T",  parent=styles["Title"],   fontSize=20, spaceAfter=6)
    s_meta  = ParagraphStyle("M",  parent=styles["Normal"],  fontSize=9,  textColor=colors.grey, spaceAfter=10)
    s_tag   = ParagraphStyle("TG", parent=styles["Normal"],  fontSize=8,  textColor=colors.HexColor("#2a7d4a"), spaceAfter=2)
    s_h2    = ParagraphStyle("H2", parent=styles["Heading2"],fontSize=13, spaceBefore=14, spaceAfter=4)
    s_body  = ParagraphStyle("B",  parent=styles["Normal"],  fontSize=10, leading=15, spaceAfter=6)
    s_disc  = ParagraphStyle("D",  parent=styles["Normal"],  fontSize=8,  textColor=colors.grey, spaceBefore=10)
    SEV = {"HIGH": "#cc0000", "MEDIUM": "#cc6600", "LOW": "#007700"}
    story = []
    story.append(Paragraph("Detailed Diagnostic Report (DDR)", s_title))
    story.append(Paragraph(
        f"Property: {report.get('propertyName','N/A')} | Date: {report.get('reportDate','N/A')} | Inspector: {report.get('inspectorName','N/A')}",
        s_meta))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc"), spaceAfter=12))
    for sec in report.get("sections", []):
        story.append(Paragraph(f"Section {sec['number']}", s_tag))
        story.append(Paragraph(sec["title"], s_h2))
        content = sec.get("content", "Not Available")
        if sec["id"] == "severity" and "|" in content:
            rows = [r for r in content.split("\n") if "|" in r]
            if rows:
                tdata = [["Issue", "Severity", "Reason"]]
                for row in rows:
                    cells = [c.strip() for c in row.split("|") if c.strip()]
                    if len(cells) >= 2:
                        sev = cells[1] if len(cells) > 1 else ""
                        col = next((v for k, v in SEV.items() if k in sev.upper()), "#000000")
                        tdata.append([Paragraph(cells[0], s_body),
                                      Paragraph(f'<font color="{col}">{sev}</font>', s_body),
                                      Paragraph(cells[2] if len(cells) > 2 else "", s_body)])
                t = Table(tdata, colWidths=[55*mm, 28*mm, 77*mm])
                t.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8f5e9")),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                    ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cccccc")),
                    ("VALIGN",(0,0),(-1,-1),"TOP"),
                ]))
                story.append(t)
                story.append(Spacer(1, 8))
                continue
        for line in content.split("\n"):
            if line.strip():
                story.append(Paragraph(line.strip(), s_body))
        story.append(Spacer(1, 4))
    cf = report.get("conflicts", "")
    if cf and cf != "None identified":
        story.append(Paragraph("Data Conflicts", s_h2))
        story.append(Paragraph(cf, s_body))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceBefore=10))
    story.append(Paragraph(report.get("disclaimer", ""), s_disc))
    doc.build(story)
    return buf.getvalue()

# ── Validation ─────────────────────────────────────────────────────────────────
async def validate(f: UploadFile, label: str) -> bytes:
    data = await f.read()
    if len(data) / 1024 / 1024 > MAX_FILE_MB:
        raise HTTPException(400, f"{label} too large (max {MAX_FILE_MB}MB)")
    return data

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model":         GROQ_MODEL,
        "api_key_set":   bool(GROQ_API_KEY),
        "pdf_available": PDF_AVAILABLE,
        "ocr_available": OCR_AVAILABLE,
    }

@app.post("/generate-report")
async def generate_report(
    inspection_file: Optional[UploadFile] = File(None),
    thermal_file:    Optional[UploadFile] = File(None),
):
    if not inspection_file and not thermal_file:
        raise HTTPException(400, "Upload at least one document.")

    if inspection_file:
        ib = await validate(inspection_file, "Inspection Report")
        fn = (inspection_file.filename or "").lower()
        insp = extract_pdf(ib, "Inspection Report") if fn.endswith(".pdf") \
               else extract_image_file(ib, inspection_file.filename, "Inspection Report")
    else:
        insp = {"text": "[Not provided]", "images": []}

    if thermal_file:
        tb = await validate(thermal_file, "Thermal Report")
        fn = (thermal_file.filename or "").lower()
        therm = extract_pdf(tb, "Thermal Report") if fn.endswith(".pdf") \
                else extract_image_file(tb, thermal_file.filename, "Thermal Report")
    else:
        therm = {"text": "[Not provided]", "images": []}

    xref   = crossref(insp, therm)
    prompt = build_prompt(insp, therm, xref)
    report = await call_groq(prompt)

    # Attach images
    all_imgs = {img["caption"]: img for img in insp["images"] + therm["images"]}
    for sec in report.get("sections", []):
        resolved = []
        for cap in sec.get("images", []):
            img = all_imgs.get(cap)
            resolved.append({"caption": cap,
                              "base64":  img["base64"] if img else None,
                              "media_type": img["media_type"] if img else None})
        sec["images"] = resolved

    rid = str(int(time.time()))
    _cache[rid] = report
    report["report_id"] = rid
    return report

@app.get("/download-pdf/{report_id}")
async def download_pdf(report_id: str):
    report = _cache.get(report_id)
    if not report:
        raise HTTPException(404, "Report not found. Generate one first.")
    if not PDF_AVAILABLE:
        raise HTTPException(501, "Install reportlab for PDF export.")
    pdf = make_pdf(report)
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="DDR_Report_{report_id}.pdf"'})

# Serve frontend
_fe = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(_fe):
    app.mount("/", StaticFiles(directory=_fe, html=True), name="static")