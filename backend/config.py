import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "tickets.db"
CSV_PATH = DATA_DIR / "failures.csv"

VSTORE_DIR = BASE_DIR / "faiss_store"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Increased from 12 to 30 for better data coverage
TOP_N_DOCS = 30
TOP_K = 3
DATE_COL = "date"

COLUMN_ALIASES = {
    "date": ["date"],
    "part_number": ["part number", "partnumber"],
    "serialnumber": ["serialnumber", "serial_number", "serial no", "serial no."],
    "typename": ["typename", "type name"],
    "stationnumber": ["stationnumber", "station number"],
    "stationdescription": ["stationdescription", "station description"],
    "failure_code": ["failure code", "faulire code", "fault code"],
    "failure_description": ["failure description", "desc", "issue", "problem"],
    "defect": ["defect"],
    "failure_details": ["failuredetails", "failure details", "details"],
    "action_code": ["action code", "action"],
    "material_code": ["material code"],
    "material_desc": ["material desc", "material description"],
    "partclass": ["partclass", "part class"],
}